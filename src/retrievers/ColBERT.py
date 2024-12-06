from ragatouille import RAGPretrainedModel
from .Retriever import Retriever
import os


class ColBERT(Retriever):
    def __init__(self, eval_dataset: str):
        super().__init__("colbert", eval_dataset)
        # Load the RAG model with ColBERT index
        index_path = os.path.join("./datasets", self.eval_dataset, 'colbert', 'indexes', f"{self.eval_dataset}-index")
        self.model = RAGPretrainedModel.from_index(index_path)

    def search_question(self, question: str, k: int):
        # Perform the search for each question
        results = self.model.search(question, k=k)

        # Collect the document IDs from the search results
        doc_ids = [result['document_id'] for result in results]
        return doc_ids
