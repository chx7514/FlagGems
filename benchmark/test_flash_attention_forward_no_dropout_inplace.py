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

import math

import pytest
import torch

import flag_gems

from . import base, consts

device = flag_gems.device

# (batch, num_heads, q_seq_len, kv_seq_len, head_dim) configs.
FLASH_FWD_CONFIGS = [
    (1, 2, 512, 512, 64),
    (1, 8, 1024, 1024, 128),
    (2, 4, 512, 512, 64),
    (1, 2, 1024, 2048, 64),
]


def torch_flash_attention_forward_no_dropout_inplace(
    q,
    k,
    v,
    scale,
    is_causal,
    return_debug_mask=False,
    **extra_kwargs,
):
    """Reference: aten::_flash_attention_forward with dropout_p=0.0."""
    return torch.ops.aten._flash_attention_forward(
        q,
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        0.0,  # dropout_p = 0.0 (no dropout)
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )


def gems_flash_attention_forward_no_dropout_inplace(
    q,
    k,
    v,
    scale,
    is_causal,
    return_debug_mask=False,
    **extra_kwargs,
):
    """FlagGems Triton implementation (no dropout, in-place into ``q``)."""
    # ``do_bench`` reuses the same tensors across iterations, so clone ``q`` to
    # preserve the original data between runs (the kernel writes in-place).
    return flag_gems._flash_attention_forward_no_dropout_inplace(
        q.clone(),
        k,
        v,
        None,
        None,
        q.shape[-3],
        k.shape[-3],
        is_causal,
        return_debug_mask,
        scale=scale,
        **extra_kwargs,
    )


def flash_attention_forward_no_dropout_inplace_input_fn(config, dtype, device):
    batch, num_head, q_seq_len, kv_seq_len, head_size = config
    q = torch.empty(
        (batch, q_seq_len, num_head, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (batch, kv_seq_len, num_head, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty(
        (batch, kv_seq_len, num_head, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    scale = float(1.0 / math.sqrt(head_size))

    # BSHD layout; no dropout; non-causal for the default benchmark configs.
    yield q, k, v, scale, False, False, {}


class FlashAttentionForwardNoDropoutInplaceBenchmark(base.GenericBenchmark):
    def set_shapes(self, shape_file_path=None):
        # Use the configs defined in FLASH_FWD_CONFIGS directly, since this
        # operator has no entry in the shared core-shapes yaml file.
        self.shapes = [tuple(c) for c in FLASH_FWD_CONFIGS]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
@pytest.mark.skipif(flag_gems.device == "cpu", reason="Unsupported in CPU mode")
@pytest.mark.flash_attention_forward_no_dropout_inplace
def test_flash_attention_forward_no_dropout_inplace_impl():
    bench = FlashAttentionForwardNoDropoutInplaceBenchmark(
        op_name="flash_attention_forward_no_dropout_inplace",
        torch_op=torch_flash_attention_forward_no_dropout_inplace,
        input_fn=flash_attention_forward_no_dropout_inplace_input_fn,
        # FlashAttention only supports fp16/bf16; filter from FLOAT_DTYPES.
        dtypes=[d for d in consts.FLOAT_DTYPES if d in (torch.float16, torch.bfloat16)],
    )
    bench.set_gems(gems_flash_attention_forward_no_dropout_inplace)
    bench.run()
