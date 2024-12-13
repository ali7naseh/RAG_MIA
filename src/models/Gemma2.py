import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .Model import Model


class Gemma(Model):
    def __init__(self, config, model_name: str):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16,
        )

    def query(self, msg, max_output_tokens=None):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to("cuda")
        num_input_tokens = input_ids.shape[1]

        outputs = self.model.generate(input_ids,
            max_new_tokens=self.max_output_tokens if max_output_tokens is None else max_output_tokens,
            early_stopping=True)

        output_tokens = outputs[0][num_input_tokens:]
        result = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return result


class Gemma2_2B(Gemma):
    def __init__(self, config):
        super().__init__(config, model_name="google/gemma-2-2b-it")


class Gemma2_9B(Gemma):
    def __init__(self, config):
        super().__init__(config, model_name="google/gemma-2-9b-it")
