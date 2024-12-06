import faiss
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from .Retriever import HFRetriever
import os


class Contriever(HFRetriever):
    def __init__(self, eval_dataset: str):
        super().__init__("contriever", eval_dataset)
        model_name = "facebook/contriever"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to('cuda').eval()

    # Query encoding function
    @torch.no_grad()
    def _encode(self, text: str):

        # Mean pooling function for query encoding
        def mean_pooling(token_embeddings, mask):
            token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
            sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
            return sentence_embeddings
    
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512).to("cuda")
        outputs = self.model(**inputs)
        embedding = mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        embedding = torch.nn.functional.normalize(embedding, p=2, dim=1)
        return embedding.cpu().numpy()
