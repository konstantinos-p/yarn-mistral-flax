import numpy as np
from unittest import TestCase
import jax

from modeling_mistral_yarn_flax import _make_causal_mask as _make_causal_mask_flax
from modeling_mistral_yarn import _make_causal_mask as _make_causal_mask_torch

from modeling_mistral_yarn_flax import _make_sliding_window_causal_mask as _make_sliding_window_causal_mask_flax
from modeling_mistral_yarn import _make_sliding_window_causal_mask as _make_sliding_window_causal_mask_torch

from modeling_mistral_yarn_flax import _yarn_linear_ramp_mask as _yarn_linear_ramp_mask_flax
from modeling_mistral_yarn import _yarn_linear_ramp_mask as _yarn_linear_ramp_mask_torch

from modeling_mistral_yarn_flax import _expand_mask as _expand_mask_flax
from modeling_mistral_yarn import _expand_mask as _expand_mask_torch

from modeling_mistral_yarn_flax import MistralRMSNorm as MistralRMSNorm_flax
from modeling_mistral_yarn import MistralRMSNorm as MistralRMSNorm_torch

from modeling_mistral_yarn_flax import MistralRotaryEmbedding as MistralRotaryEmbedding_flax
from modeling_mistral_yarn import MistralRotaryEmbedding as MistralRotaryEmbedding_torch

from modeling_mistral_yarn_flax import MistralLinearScalingRotaryEmbedding as MistralLinearScalingRotaryEmbedding_flax
from modeling_mistral_yarn import MistralLinearScalingRotaryEmbedding as MistralLinearScalingRotaryEmbedding_torch

from modeling_mistral_yarn_flax import MistralDynamicNTKScalingRotaryEmbedding as MistralDynamicNTKScalingRotaryEmbedding_flax
from modeling_mistral_yarn import MistralDynamicNTKScalingRotaryEmbedding as MistralDynamicNTKScalingRotaryEmbedding_torch

import torch
import jax.numpy as jnp


class Test(TestCase):
    def test__get_unpad_data(self):
        self.fail()


class Test(TestCase):
    def test__make_causal_mask(self):
        """
        Compare the causal mask function between the pytorch and the flax implementation
        """
        # torch version of _make_causal_mask
        t = torch.empty(10, 10)
        input_ids_shape = t.size()
        dtype = torch.float32
        device = torch.device("cpu")
        past_key_values_length = 10

        outputs_pytorch = _make_causal_mask_torch(
            input_ids_shape=input_ids_shape,
            dtype=dtype,
            device=device,
            past_key_values_length=past_key_values_length)

        # flax version of _make_causal_mask
        input_ids_shape = (10, 10)
        dtype = jnp.dtype('float32')
        past_key_values_length = 10

        outputs_flax = _make_causal_mask_flax(
            input_ids_shape=input_ids_shape,
            dtype=dtype,
            past_key_values_length=past_key_values_length)

        np_array = np.asarray(outputs_flax)
        flax_to_torch_ten = torch.from_numpy(np_array)

        tol = torch.max(torch.abs(flax_to_torch_ten - outputs_pytorch))

        self.assertTrue(torch.allclose(flax_to_torch_ten, outputs_pytorch, atol=1e-03))


class Test(TestCase):
    def test__make_sliding_window_causal_mask(self):
        """
        Compare the sliding_window_causal mask function between the pytorch and the flax implementation
        """
        # torch version of _make_causal_mask
        t = torch.empty(10, 10)
        input_ids_shape = t.size()
        dtype = torch.float32
        device = torch.device("cpu")
        past_key_values_length = 10
        sliding_window = 5

        outputs_pytorch = _make_sliding_window_causal_mask_torch(
            input_ids_shape=input_ids_shape,
            dtype=dtype,
            device=device,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window)

        # flax version of _make_causal_mask
        input_ids_shape = (10, 10)
        dtype = jnp.dtype('float32')
        past_key_values_length = 10
        sliding_window = 5

        outputs_flax = _make_sliding_window_causal_mask_flax(
            input_ids_shape=input_ids_shape,
            dtype=dtype,
            past_key_values_length=past_key_values_length,
            sliding_window=sliding_window)

        np_array = np.asarray(outputs_flax)
        flax_to_torch_ten = torch.from_numpy(np_array)

        self.assertTrue(torch.allclose(flax_to_torch_ten, outputs_pytorch, atol=1e-03))


class Test(TestCase):
    def test__expand_mask(self):
        self.fail()


class Test(TestCase):
    def test__yarn_linear_ramp_mask(self):
        outputs_pytorch = _yarn_linear_ramp_mask_torch(0, 1, 10)
        outputs_flax = _yarn_linear_ramp_mask_flax(0, 1, 10)

        np_array = np.asarray(outputs_flax)
        flax_to_torch_ten = torch.from_numpy(np_array)

        self.assertTrue(torch.allclose(flax_to_torch_ten, outputs_pytorch, atol=1e-03))


class Test(TestCase):
    def test__expand_mask(self):
        mask = torch.zeros((12, 22))
        dtype = torch.float32
        tgt_len = 10

        outputs_pytorch = _expand_mask_torch(
            mask=mask,
            dtype=dtype,
            tgt_len=tgt_len
        )

        mask = jnp.zeros((12, 22))
        dtype = jnp.float32
        tgt_len = 10

        outputs_flax = _expand_mask_flax(
            mask=mask,
            dtype=dtype,
            tgt_len=tgt_len
        )

        np_array = np.asarray(outputs_flax)
        flax_to_torch_ten = torch.from_numpy(np_array)

        self.assertTrue(torch.allclose(flax_to_torch_ten, outputs_pytorch, atol=1e-03))


class TestMistralRMSNorm(TestCase):
    def test_setup(self):
        flax_layer = MistralRMSNorm_flax(hidden_size=4096)
        params = flax_layer.init(jax.random.key(0), jnp.ones((1, 13, 4096)))
        flax_input = jax.random.normal(jax.random.key(0), shape=(1, 13, 4096))
        flax_outputs = flax_layer.apply(params, flax_input)

        torch_layer = MistralRMSNorm_torch(hidden_size=4096)
        np_input = np.asarray(flax_input)
        torch_input = torch.from_numpy(np_input)
        torch_outputs = torch_layer(torch_input)

        np_array = np.asarray(flax_outputs)
        flax_to_torch_ten = torch.from_numpy(np_array)

        self.assertTrue(torch.allclose(flax_to_torch_ten, torch_outputs, atol=1e-03))


class TestMistralRotaryEmbedding(TestCase):
    def test_setup(self):
        flax_layer = MistralRotaryEmbedding_flax(dim=50)
        params = flax_layer.init(jax.random.key(0), jnp.ones((20, 13, 10, 40)), 10)
        flax_input = jax.random.normal(jax.random.key(0), shape=(20, 13, 10, 40))
        flax_outputs = flax_layer.apply(params, flax_input, 10)

        torch_layer = MistralRotaryEmbedding_torch(dim=50)
        np_input = np.asarray(flax_input)
        torch_input = torch.from_numpy(np_input)
        torch_outputs = torch_layer(torch_input, 10)

        np_array = np.asarray(flax_outputs)
        flax_to_torch_ten = torch.from_numpy(np_array)

        cond1 = torch.allclose(flax_to_torch_ten[0], torch_outputs[0], atol=1e-03)
        cond2 = torch.allclose(flax_to_torch_ten[1], torch_outputs[1], atol=1e-03)

        self.assertTrue(cond1 and cond2)


class TestMistralLinearScalingRotaryEmbedding(TestCase):
    def test_setup(self):
        flax_layer = MistralLinearScalingRotaryEmbedding_flax(dim=50, scaling_factor=2)
        params = flax_layer.init(jax.random.key(0), jnp.ones((20, 13, 10, 40)), 10)
        flax_input = jax.random.normal(jax.random.key(0), shape=(20, 13, 10, 40))
        flax_outputs = flax_layer.apply(params, flax_input, 10)

        torch_layer = MistralLinearScalingRotaryEmbedding_torch(dim=50, scaling_factor=2)
        np_input = np.asarray(flax_input)
        torch_input = torch.from_numpy(np_input)
        torch_outputs = torch_layer(torch_input, 10)

        np_array = np.asarray(flax_outputs)
        flax_to_torch_ten = torch.from_numpy(np_array)

        cond1 = torch.allclose(flax_to_torch_ten[0], torch_outputs[0], atol=1e-03)
        cond2 = torch.allclose(flax_to_torch_ten[1], torch_outputs[1], atol=1e-03)
        # !!! There is a possible bug in the original implementation cos,cos is returned instead of cos,sin
        self.assertTrue(cond1 and cond2)


class TestMistralDynamicNTScalingRotaryEmbedding(TestCase):
    def test_setup(self):
        flax_layer = MistralDynamicNTKScalingRotaryEmbedding_flax(dim=50, scaling_factor=2)
        params = flax_layer.init(jax.random.key(0), jnp.ones((20, 13, 10, 40)), 10)
        flax_input = jax.random.normal(jax.random.key(0), shape=(20, 13, 10, 40))
        flax_outputs = flax_layer.apply(params, flax_input, 10)

        torch_layer = MistralDynamicNTKScalingRotaryEmbedding_torch(dim=50, scaling_factor=2)
        np_input = np.asarray(flax_input)
        torch_input = torch.from_numpy(np_input)
        torch_outputs = torch_layer(torch_input, 10)

        np_array = np.asarray(flax_outputs)
        flax_to_torch_ten = torch.from_numpy(np_array)

        cond1 = torch.allclose(flax_to_torch_ten[0], torch_outputs[0], atol=1e-03)
        cond2 = torch.allclose(flax_to_torch_ten[1], torch_outputs[1], atol=1e-03)
        self.assertTrue(cond1 and cond2)
