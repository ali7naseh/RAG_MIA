import os
import faiss
import pickle
from typing import List


class Retriever:
    def __init__(self, name, eval_dataset: str):
        self.name = name
        self.eval_dataset = eval_dataset
        self.model = None
    
    def _encode(self, text: str):
        raise NotImplementedError("_encode method must be implemented in subclass")

    def encode_query(self, query: str):
        # Child class can override this method if needed
        return self._encode(query)
    
    def encode_passage(self, passage: str):
        # Child class can override this method if needed
        return self._encode(passage)

    def search_question(self, question: str, k: int):
        raise NotImplementedError("search_question method must be implemented in subclass")


class HFRetriever(Retriever):
    def __init__(self, name, eval_dataset: str):
        super().__init__(name, eval_dataset)
        self.faiss_index = None
        self.doc_ids = None

    def faiss_load(self):
        # Paths for FAISS index and document IDs
        index_folder = os.path.join("./datasets", self.eval_dataset, self.name.lower(), 'indexes', f"{self.eval_dataset}-index")
        faiss_index_path = os.path.join(index_folder, "corpus_index.faiss")
        doc_ids_path = os.path.join(index_folder, "doc_ids.pkl")

        # Load FAISS index
        print(f"Loading FAISS index from {faiss_index_path}")
        self.faiss_index = faiss.read_index(faiss_index_path)

        # Load document IDs
        print(f"Loading document IDs from {doc_ids_path}")
        with open(doc_ids_path, "rb") as f:
            self.doc_ids = pickle.load(f)

    def search_question(self, question: str, k: int):
        # Lazy-load FAISS index and document IDs
        if self.faiss_index is None:
            self.faiss_load()

        try:
            query_embedding = self.encode_query(question)
            distances, indices = self.faiss_index.search(query_embedding, k)
            retrieved_ids = [self.doc_ids[i] for i in indices[0]]
        except Exception as e:
            return None
        return retrieved_ids


class HFReranker(HFRetriever):
    def encode(self, text: str, passages: List[str]) -> List[float]:
        raise NotImplementedError("_encode method must be implemented in subclass")
    
    def faiss_load(self):
        pass

    def reranked_topk(self, question: str,
                      passages: List[str],
                      k: int,
                      batch_size: int=16):
        scores = []
        for i in range(0, len(passages), batch_size):
            batch_passages = passages[i:i + batch_size]
            batch_scores = self.encode(question, batch_passages)
            scores.extend(batch_scores)
        # Pick the top-k passages and return
        top_k_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        return top_k_indices
