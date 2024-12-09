from .Retriever import HFRetriever
from transformers import AutoTokenizer, AutoModel
import torch.nn.functional as F
import torch
import json


class NVEmbed(HFRetriever):
    def __init__(self, eval_dataset: str):
        super().__init__("nvembed", eval_dataset)
        model_name = "nvidia/NV-Embed-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name,
                                               trust_remote_code=True,
                                               torch_dtype=torch.float16).to("cuda").eval()
        # Load instruction mapping
        with open("instructions.json", 'r') as f:
            instruction_mapping = json.load(f)
        
        self.instruction_mapping = instruction_mapping.get(self.eval_dataset.lower(), None)
        if self.instruction_mapping is None:
            raise ValueError(f"No instruction mapping found for dataset {self.eval_dataset}")

    @torch.no_grad()
    def encode_query(self, query: str):
        # Check for instruction
        query_prefix = "Instruct: " + self.instruction_mapping["query"]+"\nQuery: "
        max_length = 1024 #32768
        embeddings = self.model.encode(query, instruction=query_prefix, max_length=max_length)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()

    @torch.no_grad()
    def encode_passage(self, passage: str):
        # Check for instruction
        max_length = 1024 #32768
        embeddings = self.model.encode(passage, max_length=max_length)
        embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings.cpu().numpy()
