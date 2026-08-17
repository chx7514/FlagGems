# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import pytest
import torch

import flag_gems

from . import accuracy_utils as utils
from . import conftest as cfg

device = flag_gems.device

# Small attention configs (batch, num_heads, seq_len, kv_len).
# FlashAttention only supports fp16/bf16 with head_dim a multiple of 8.
NO_DROPOUT_INPLACE_CONFIGS = [
    (1, 2, 64, 64),
    (2, 4, 128, 128),
    (1, 8, 256, 256),
    (1, 2, 17, 1030),
]
NO_DROPOUT_INPLACE_HEAD_DIMS = [64, 128]


def make_input(batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device):
    # FlashAttention expects the BSHD layout: (batch, seqlen, num_heads, head_dim).
    q_shape = (batch, q_seq_len, num_head, head_size)
    kv_shape = (batch, kv_seq_len, num_head, head_size)
    q = torch.empty(q_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    k = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    v = torch.empty(kv_shape, dtype=dtype, device=device).uniform_(-0.05, 0.05)
    return q, k, v


def torch_ref(q, k, v, scale, is_causal):
    """Reference implementation: aten::_flash_attention_forward with dropout_p=0.0."""
    (
        out,
        lse,
        seed,
        offset,
        debug_softmax,
    ) = torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        0.0,  # dropout_p = 0.0 (no dropout)
        is_causal,
        False,
        scale=scale,
    )
    return out, lse, seed, offset, debug_softmax


def gems_impl(q, k, v, scale, is_causal):
    """FlagGems implementation (no dropout, in-place into ``q``)."""
    q_clone = q.clone()
    with flag_gems.use_gems():
        (
            out,
            lse,
            seed,
            offset,
            debug_softmax,
        ) = flag_gems._flash_attention_forward_no_dropout_inplace(
            q_clone,
            k,
            v,
            None,
            None,
            q.shape[-3],
            k.shape[-3],
            is_causal,
            False,
            scale=scale,
        )
    # In-place semantics: output aliases the query tensor.
    assert out.data_ptr() == q_clone.data_ptr(), "output must alias the query tensor"
    return out, lse, seed, offset, debug_softmax


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(cfg.TO_CPU, reason="Unsupported in CPU mode")
@pytest.mark.flash_attention_forward_no_dropout_inplace
@pytest.mark.parametrize(
    ["batch", "num_head", "q_seq_len", "kv_seq_len"],
    NO_DROPOUT_INPLACE_CONFIGS,
)
@pytest.mark.parametrize("head_size", NO_DROPOUT_INPLACE_HEAD_DIMS)
@pytest.mark.parametrize("is_causal", [False, True])
# FlashAttention only supports fp16/bf16; filter from FLOAT_DTYPES.
@pytest.mark.parametrize(
    "dtype", [d for d in utils.FLOAT_DTYPES if d in (torch.float16, torch.bfloat16)]
)
def test_flash_attention_forward_no_dropout_inplace_impl(
    batch, num_head, q_seq_len, kv_seq_len, head_size, is_causal, dtype
):
    q, k, v = make_input(
        batch, num_head, q_seq_len, kv_seq_len, head_size, dtype, device
    )
    ref_q = utils.to_reference(q, False)
    ref_k = utils.to_reference(k, False)
    ref_v = utils.to_reference(v, False)
    scale = float(1.0 / np.sqrt(head_size))

    torch_out, torch_lse, _, _, _ = torch_ref(ref_q, ref_k, ref_v, scale, is_causal)
    gems_out, gems_lse, _, _, _ = gems_impl(q, k, v, scale, is_causal)

    utils.gems_assert_close(gems_out, torch_out, dtype)
    utils.gems_assert_close(gems_lse, torch_lse, torch.float)
