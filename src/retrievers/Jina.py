from .Retriever import HFReranker
from transformers import AutoModelForSequenceClassification
from typing import List
import torch


class JinaReranker(HFReranker):
    def __init__(self, eval_dataset: str):
        super().__init__("jina", eval_dataset)
        model_name = "jinaai/jina-reranker-v2-base-multilingual"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                                                        torch_dtype="auto",
                                                                        use_flash_attn=False, # Enable if you have flash-attention setup on your machine
                                                                        trust_remote_code=True).to("cuda").eval()

    @torch.no_grad()
    def encode(self, text: str, passages: List[str]) -> List[float]:
        sentence_pairs = [[text, doc] for doc in passages]
        scores = self.model.compute_score(sentence_pairs, max_length=2048)
        return scores
