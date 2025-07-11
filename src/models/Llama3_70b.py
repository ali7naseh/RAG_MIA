import torch
# from transformers import LlamaTokenizer, LlamaForCausalLM
from transformers import AutoModelForCausalLM, AutoTokenizer
from .Model import Model

class Llama3_70b(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        api_pos = int(config["api_key_info"]["api_key_use"])
        hf_token = config["api_key_info"]["api_keys"][api_pos]

        # self.tokenizer = LlamaTokenizer.from_pretrained(self.name, use_auth_token=hf_token)
        # self.model = LlamaForCausalLM.from_pretrained(self.name, torch_dtype=torch.float16, use_auth_token=hf_token).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained('/datasets/ai/llama3/huggingface/llama-3-70b-instruct')
        self.model = AutoModelForCausalLM.from_pretrained('/datasets/ai/llama3/huggingface/llama-3-70b-instruct', device_map="auto", torch_dtype=torch.float16 )

    def query(self, msg, max_output_tokens=None):
        input_ids = self.tokenizer(msg, return_tensors="pt").input_ids.to("cuda")
        num_input_tokens = input_ids.shape[1]

        outputs = self.model.generate(input_ids,
            max_new_tokens=self.max_output_tokens if max_output_tokens is None else max_output_tokens,
            early_stopping=True)

        output_tokens = outputs[0][num_input_tokens:]
        result = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        #out = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #result = out[len(msg):]
        return result