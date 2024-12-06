from .Retriever import HFRetriever
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch


class BGE(HFRetriever):
    def __init__(self, eval_dataset: str):
        super().__init__("bge", eval_dataset)
        model_name = "BAAI/bge-large-en-v1.5"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to('cuda').eval()
    
    # Function for encoding queries
    @torch.no_grad()
    def _encode(self, text: str):
        inputs = self.tokenizer(text, return_tensors="pt", padding=True,
                                truncation=True, max_length=512).to("cuda")
        outputs = self.model(**inputs)
        embedding = normalize_embeddings(outputs.last_hidden_state[:, 0])  # CLS pooling
        return embedding.cpu().numpy()


def normalize_embeddings(embeddings):
    return torch.nn.functional.normalize(embeddings, p=2, dim=1)
