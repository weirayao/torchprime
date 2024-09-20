import unittest

from transformers import AutoConfig, LlamaForCausalLM as HfLlamaForCausalLM

import torch
import torch_xla.core.xla_model as xm

from torchprime.models.llama import LlamaForCausalLM


class TestYourModule(unittest.TestCase):
    def test_forward(self):
        config = AutoConfig.from_pretrained(
            "meta-llama/Meta-Llama-3-8B",
            num_hidden_layers=1,
            num_attention_heads=8,
            hidden_size=8,
            intermediate_size=16,
            vocab_size=128,
        )
        config.flash_attention = False

        device = xm.xla_device()
        torch.manual_seed(
            42)  # Such that the two models are initialized the same way
        hf_model = HfLlamaForCausalLM(config).to(device)
        torch.manual_seed(
            42)  # Such that the two models are initialized the same way
        model = LlamaForCausalLM(config).to(device)

        input_sizes = [8, 128, 256]
        for input_size in input_sizes:
            input = torch.randint(128, ((2, input_size // 2))).to(device)
            hf_output = hf_model(input, labels=input)
            llama_output = model(input, labels=input)
            self.assertTrue(torch.allclose(hf_output.logits, llama_output.logits, atol=1e-6),
                            "logits are not equal")
            self.assertTrue(torch.allclose(hf_output.loss, llama_output.loss, atol=1e-6),
                            "loss is not equal")


if __name__ == '__main__':
    unittest.main()
