from transformers import AutoModelForCausalLM, AutoTokenizer
from .Model import Model


class Phi4(Model):
    def __init__(self, config):
        super().__init__(config)
        self.max_output_tokens = int(config["params"]["max_output_tokens"])
        self.device = config["params"]["device"]
        self.max_output_tokens = config["params"]["max_output_tokens"]

        model_name = "microsoft/phi-4"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            trust_remote_code=True
        )

    def to(self, device):
        pass

    def query(self, msg: str, max_output_tokens=None):
        chat_template_extra_tokens = 4 # Phi-4 uses chat-template and will add 4 tokens

        # Phi-4 expects chat-template
        chat = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": msg}
        ]
        wrapped_msg = self.tokenizer.apply_chat_template(chat, tokenize=False)
        inputs = self.tokenizer(wrapped_msg, return_tensors="pt", max_length=12296, truncation=True)
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}
        num_input_tokens = inputs['input_ids'].shape[1]

        max_new_tokens = self.max_output_tokens if max_output_tokens is None else max_output_tokens
        max_new_tokens += chat_template_extra_tokens

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            early_stopping=True
        )
        output_tokens = outputs[0][num_input_tokens:]
        result = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
        return result
