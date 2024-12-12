from .Retriever import HFReranker
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List
import torch


class MixedBreadReranker(HFReranker):
    def __init__(self, eval_dataset: str):
        super().__init__("mixed_bread", eval_dataset)
        model_name = "mixedbread-ai/mxbai-rerank-large-v1"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name,
                                               torch_dtype="auto",
                                               trust_remote_code=True).to("cuda").eval()

    @torch.no_grad()
    def encode(self, text: str, passages: List[str]) -> List[float]:
        inputs = self.tokenizer(
            [text] * len(passages),  # Repeat query for each passage
            passages,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        outputs = self.model(**inputs)
        # Apply sigmoid to get probabilities
        scores = torch.sigmoid(outputs.logits.detach().squeeze(-1))
        return scores.cpu().numpy().tolist()
