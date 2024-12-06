from .Retriever import HFRetriever
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch


class GTE(HFRetriever):
    def __init__(self, eval_dataset: str):
        super().__init__("gte", eval_dataset)
        model_name = "Alibaba-NLP/gte-large-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,
                                               trust_remote_code=True).to("cuda").eval()

    @torch.no_grad()
    def _encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True,
                                padding=True, max_length=8192).to("cuda")
        outputs = self.model(**inputs)
        embedding = F.normalize(outputs.last_hidden_state[:, 0], p=2, dim=1)  # CLS pooling and normalization
        return embedding.cpu().numpy()
