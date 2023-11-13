# coding=utf-8
# Copyright 2023 Meta AI, EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
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
"""Flax LLaMA model."""
from functools import partial
from typing import Optional, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
from flax.core.frozen_dict import FrozenDict, freeze, unfreeze
from flax.linen import combine_masks, make_causal_mask
from flax.linen.attention import dot_product_attention_weights
from flax.traverse_util import flatten_dict, unflatten_dict
from jax import lax

from transformers.modeling_flax_outputs import FlaxBaseModelOutput, FlaxCausalLMOutput
from transformers.modeling_flax_utils import ACT2FN, FlaxPreTrainedModel, append_call_sample_docstring
from transformers.utils import add_start_docstrings, add_start_docstrings_to_model_forward, logging
from configuration_mistral import MistralConfig
import math

logger = logging.get_logger(__name__)

_CONFIG_FOR_DOC = "MistralConfig"


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
# Propably not used in the flax implementation
"""
def _get_unpad_data(padding_mask):
    seqlens_in_batch = jnp.sum(padding_mask, axis=-1, dtype=np.int32)
    indices = jnp.ravel(jnp.nonzero(jnp.ravel(padding_mask)))
    max_seqlen_in_batch = jnp.max(seqlens_in_batch)
    cu_seqlens = jnp.cumsum(seqlens_in_batch, axis=0, dtype=np.int32)
    pad_width = [(0, 0)] * (cu_seqlens.ndim - 1) + [(1, 0)]
    cu_seqlens = jnp.pad(cu_seqlens, pad_width)
    return (indices,
            cu_seqlens,
            max_seqlen_in_batch
            )
"""


def _make_causal_mask(
    input_ids_shape: jax.numpy.size,
    dtype: jax.numpy.dtype,
    past_key_values_length: int = 0
):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = jnp.full((tgt_len, tgt_len), fill_value=jnp.finfo(dtype).min)
    mask_cond = jnp.reshape(jnp.arange(mask.shape[-1]), (1, -1))
    mask = mask.at[jnp.where(mask_cond < (mask_cond.T+1))].set(0)
    mask = mask.astype(dtype)

    if past_key_values_length > 0:
        mask = jnp.concatenate([jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

    return jax.lax.broadcast_in_dim(mask[None, None, :, :],
                                   shape=(bsz, 1, tgt_len, tgt_len + past_key_values_length),
                                   broadcast_dimensions=(0, 1, 2, 3))


def _make_sliding_window_causal_mask(
    input_ids_shape: jax.numpy.size,
    dtype: jax.numpy.dtype,
    past_key_values_length: int = 0,
    sliding_window: int = 4096
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = jnp.full((tgt_len, tgt_len), fill_value=1)

    mask = jnp.tril(tensor, k=0)
    # make the mask banded to account for sliding window
    mask = jnp.triu(mask, k=-sliding_window)
    mask = jnp.log(mask).astype(dtype)

    if past_key_values_length > 0:
        mask = jnp.concatenate([jnp.zeros((tgt_len, past_key_values_length), dtype=dtype), mask], axis=-1)

    return jax.lax.broadcast_in_dim(mask[None, None, :, :],
                                   shape=(bsz, 1, tgt_len, tgt_len + past_key_values_length),
                                   broadcast_dimensions=(0, 1, 2, 3))


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(
        mask: jax.numpy.array,
        dtype: jax.numpy.dtype,
        tgt_len: Optional[int] = None
):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.shape
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = jax.lax.broadcast_in_dim(mask[:, None, None, :],
                                   shape=(bsz, 1, tgt_len, src_len),
                                   broadcast_dimensions=(0, 1, 2, 3))

    inverted_mask = 1 - expanded_mask

    inverted_mask = inverted_mask.at[inverted_mask.astype(bool)].set(jnp.finfo(dtype).min)
    return inverted_mask


# Inverse dim formula to find dim based on number of rotations
def _yarn_find_correction_dim(
        num_rotations,
        dim,
        base=10000,
        max_position_embeddings=2048
):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
        low_rot,
        high_rot,
        dim,
        base=10000,
        max_position_embeddings=2048
):
    low = math.floor(_yarn_find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(_yarn_find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case


def _yarn_linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (jnp.arange(dim, dtype=jnp.float32) - min) / (max - min)
    ramp_func = jnp.clip(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.07 * math.log(scale) + 1.0


class MistralRMSNorm(nn.Module):

    hidden_size: float
    eps: float = 1e-6

    def setup(self):
        # all the methods below for initializing the parameters work
        #self.weight = self.param('weight', lambda key: jnp.full((self.hidden_size,), 1.0))
        #self.weight = self.param('weight', lambda key, shape: jnp.full(shape, 1.0), (self.hidden_size,))
        self.weight = self.param('weight', nn.initializers.constant(1.0), (self.hidden_size,))
        self.variance_epsilon = self.eps
        return

    def __call__(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.astype(jnp.float32)
        variance = jnp.mean(jnp.power(hidden_states, 2), axis=-1, keepdims=True)
        hidden_states = hidden_states * jax.lax.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.astype(input_dtype)


class MistralRotaryEmbedding(nn.Module):

    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000

    def setup(self):
        self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))
        self._set_cos_sin_cache(
            seq_len=self.max_position_embeddings, dtype=jnp.float32
        )

    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = jnp.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = jnp.einsum("i,j->ij", t, jax.lax.stop_gradient(self.inv_freq))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = jnp.cos(emb).astype(dtype)
        self.sin_cached = jnp.sin(emb).astype(dtype)

    def __call__(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(
                seq_len=seq_len, dtype=jnp.float32
            )
        return (
            jax.lax.stop_gradient(self.cos_cached[:seq_len].astype(x.dtype)),
            jax.lax.stop_gradient(self.sin_cached[:seq_len].astype(x.dtype)),
        )


class MistralLinearScalingRotaryEmbedding(MistralRotaryEmbedding):
    """MistralRotaryEmbedding extended with linear scaling. Credits to the Reddit user /u/kaiokendev"""

    scaling_factor: float = 1

    def setup(self):
        super().setup()
        return
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len
        t = jnp.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        t = t / self.scaling_factor
        freqs = jnp.einsum("i,j->ij", t, jax.lax.stop_gradient(self.inv_freq))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = jnp.cos(emb).astype(dtype)
        self.sin_cached = jnp.sin(emb).astype(dtype)



class MistralDynamicNTKScalingRotaryEmbedding(MistralRotaryEmbedding):
    """MistralRotaryEmbedding extended with Dynamic NTK scaling. Credits to the Reddit users /u/bloc97 and /u/emozilla"""

    scaling_factor: float = 1

    def setup(self):
        super().setup()
        return
    def _set_cos_sin_cache(self, seq_len, dtype):
        self.max_seq_len_cached = seq_len

        if seq_len > self.max_position_embeddings:
            base = self.base * (
                (self.scaling_factor * seq_len / self.max_position_embeddings) - (self.scaling_factor - 1)
            ) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim))

        t = jnp.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)

        freqs = jnp.einsum("i,j->ij", t, jax.lax.stop_gradient(self.inv_freq))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.cos_cached = jnp.cos(emb).astype(dtype)
        self.sin_cached = jnp.sin(emb).astype(dtype)


class MistralYaRNScaledRotaryEmbedding(nn.Module):
    """MistralRotaryEmbedding extended with YaRN. See: https://arxiv.org/abs/2309.00071"""
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000
    scale: float = 1.0
    original_max_position_embeddings: int = 2048
    extrapolation_factor: float = 1.0
    attn_factor: float = 1.0
    beta_fast: float = 1.0
    beta_slow: float = 2.0
    finetuned: bool = False

    def setup(self):

        self.yarn()

        self.max_seq_len_cached = self.max_position_embeddings
        t = jnp.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = jnp.einsum("i,j->ij", t, jax.lax.stop_gradient(self.inv_freq))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.dtype = jnp.float32
        self.cos_cached = (jnp.cos(emb) * self.mscale).astype(self.dtype)
        self.sin_cached = (jnp.sin(emb) * self.mscale).astype(self.dtype)

    def __call__(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len
            t = jnp.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = jnp.einsum("i,j->ij", t, jax.lax.stop_gradient(self.inv_freq))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = jnp.concatenate((freqs, freqs), axis=-1)
            self.cos_cached = (jnp.cos(emb) * self.mscale).astype(self.dtype)
            self.sin_cached = (jnp.sin(emb) * self.mscale).astype(self.dtype)
        return (
            jax.lax.stop_gradient(self.cos_cached[:seq_len].astype(x.dtype)),
            jax.lax.stop_gradient(self.sin_cached[:seq_len].astype(x.dtype)),
        )

    def yarn(self):
        pos_freqs = self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim)
        inv_freq_etrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (self.scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast,
                                                self.beta_slow,
                                                self.dim,
                                                self.base,
                                                self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).astype(jnp.float32)) * self.extrapolation_factor
        self.inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_etrapolation * inv_freq_mask
        self.mscale = float(_yarn_get_mscale(self.scale) * self.attn_factor)


class MistralDynamicYaRNScaledRotaryEmbedding(nn.Module):
    """MistralRotaryEmbedding extended with YaRN. See: https://arxiv.org/abs/2309.00071"""
    dim: int
    max_position_embeddings: int = 2048
    base: float = 10000
    original_max_position_embeddings: int = 2048
    extrapolation_factor: float = 1.0
    attn_factor: float = 1.0
    beta_fast: float = 1.0
    beta_slow: float = 2.0
    finetuned: bool = False

    def setup(self):

        if self.finetuned:
            self.yarn(self.max_position_embeddings / self.original_max_position_embeddings)
        else:
            self.inv_freq = 1.0 / (self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float) / self.dim))

        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = self.max_position_embeddings
        t = jnp.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
        freqs = jnp.einsum("i,j->ij", t, jax.lax.stop_gradient(self.inv_freq))
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = jnp.concatenate((freqs, freqs), axis=-1)
        self.dtype = jnp.float32

        self.cos_cached = (jnp.cos(emb) * self.mscale).astype(self.dtype)
        self.sin_cached = (jnp.sin(emb) * self.mscale).astype(self.dtype)

    def __call__(self, x, seq_len):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        if seq_len > self.max_seq_len_cached:
            self.max_seq_len_cached = seq_len

            self.yarn(seq_len / self.max_position_embeddings)

            t = jnp.arange(self.max_seq_len_cached, dtype=self.inv_freq.dtype)
            freqs = jnp.einsum("i,j->ij", t, jax.lax.stop_gradient(self.inv_freq))
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = jnp.concatenate((freqs, freqs), axis=-1)
            self.cos_cached = (jnp.cos(emb) * self.mscale).astype(self.dtype)
            self.sin_cached = (jnp.sin(emb) * self.mscale).astype(self.dtype)
        return (
            jax.lax.stop_gradient(self.cos_cached[:seq_len].astype(x.dtype)),
            jax.lax.stop_gradient(self.sin_cached[:seq_len].astype(x.dtype)),
        )

    def yarn(self, scale):
        pos_freqs = self.base ** (jnp.arange(0, self.dim, 2).astype(jnp.float32) / self.dim)
        inv_freq_etrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scale * pos_freqs)

        low, high = _yarn_find_correction_range(self.beta_fast,
                                                self.beta_slow,
                                                self.dim,
                                                self.base,
                                                self.original_max_position_embeddings)
        inv_freq_mask = (1 - _yarn_linear_ramp_mask(low, high, self.dim // 2).astype(jnp.float32)) * self.extrapolation_factor
        self.inv_freq = inv_freq_interpolation * (1 - inv_freq_mask) + inv_freq_etrapolation * inv_freq_mask
        self.mscale = float(_yarn_get_mscale(scale) * self.attn_factor)



































