from unittest import TestCase
from modeling_mistral_yarn import MistralForCausalLM
from transformers import AutoTokenizer


class TestMistralForCausalLM(TestCase):
    def test_forward(self):

        model = MistralForCausalLM.from_pretrained('/Users/pkonstan/PycharmProjects/yarn-flax/Yarn-Mistral-7b-128k/',
                                                   device_map="auto",
                                                   offload_folder="offload",
                                                   offload_state_dict=True)
        tokenizer = AutoTokenizer.from_pretrained('/Users/pkonstan/PycharmProjects/yarn-flax/Yarn-Mistral-7b-128k/')

        prompt = "Hey, are you conscious? Can you talk to me?"
        inputs = tokenizer(prompt, return_tensors="pt")

        # Generate
        generate_ids = model.generate(inputs.input_ids, max_length=30)
        a = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

        self.assertTrue(True)
