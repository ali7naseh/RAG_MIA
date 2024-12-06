import faiss
import torch
import pickle
from transformers import AutoTokenizer, AutoModel
from .Retriever import Retriever
import os


class Contriever(Retriever):
    def __init__(self, eval_dataset: str):
        super().__init__("contriever", eval_dataset)
        model_name = "facebook/contriever"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to('cuda').eval()
        
        # Paths for FAISS index and document IDs
        index_folder = os.path.join("./datasets", eval_dataset, 'contriever', 'indexes', f"{eval_dataset}-index")
        faiss_index_path = os.path.join(index_folder, "corpus_index.faiss")
        doc_ids_path = os.path.join(index_folder, "doc_ids.pkl")

        # Load FAISS index
        print(f"Loading FAISS index from {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)

        # Load document IDs
        print(f"Loading document IDs from {doc_ids_path}")
        with open(doc_ids_path, "rb") as f:
            self.doc_ids = pickle.load(f)

    # Mean pooling function for query encoding
    def mean_pooling(self, token_embeddings, mask):
        token_embeddings = token_embeddings.masked_fill(~mask[..., None].bool(), 0.0)
        sentence_embeddings = token_embeddings.sum(dim=1) / mask.sum(dim=1)[..., None]
        return sentence_embeddings

    # Query encoding function
    @torch.no_grad()
    def encode_query(self, query, tokenizer, model, device):
        inputs = tokenizer(query, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = model(**inputs)
        query_embedding = self.mean_pooling(outputs.last_hidden_state, inputs['attention_mask'])
        query_embedding = torch.nn.functional.normalize(query_embedding, p=2, dim=1)
        return query_embedding.cpu().numpy()

    def search_question(self, question: str, k: int):
        try:
            query_embedding = self.encode_query(question, self.tokenizer, self.model, torch.device("cuda"))
            distances, indices = self.faiss_index.search(query_embedding, k)
            retrieved_ids = [self.doc_ids[i] for i in indices[0]]
        except Exception as e:
            return None
        return retrieved_ids
