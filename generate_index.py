"""
    Generate and cache passage embeddings
"""
import os
import json
import pickle
import torch
import numpy as np
import faiss
import argparse
import yaml
from tqdm import tqdm

from src.retrievers import create_retriever
from src.retrievers.Retriever import Retriever


# Batch Query Encoding Function
@torch.no_grad()
def encode_queries(queries, model: Retriever, batch_size: int=32):
    """
    Encode a batch of queries using GTE.
    """
    embeddings = []
    for i in range(0, len(queries), batch_size):
        batch_queries = queries[i:i + batch_size]

        query_embeddings = model.encode_query(batch_queries)
        embeddings.append(query_embeddings)

    return np.concatenate(embeddings, axis=0)


def test(model: Retriever, k: int):
    # Test Queries
    queries = [
        "Is dexmedetomidine used as part of enhanced recovery after surgery (ERAS) protocols to reduce opioid consumption in the Post anesthesia care unit (PACU)?",
        "Does the 2016 American Society of Anesthesiologists (ASA) guidelines recommend a multimodal opioid-sparing approach for postoperative pain management?",
        "Does persistent postsurgical pain (PPP) have an incidence rate of up to 50%?",
        "Can dexmedetomidine be used as an adjuvant in epidurals to enhance local anesthetic sparing effects?",
        "Is local infiltration of IV dexmedetomidine associated with earlier discharge from the Post anesthesia care unit (PACU)?",
        "Is dexmedetomidine known to provide analgesic effects through a different mechanism of action compared to opioids?",
        "Does dexmedetomidine improve hemodynamic stability when used at the end of a surgical procedure?",
        "Is dexmedetomidine associated with better pain response than ropivacaine when initiated at the end of surgery?",
        "Can dexmedetomidine be used during nerve blocks to reduce postoperative pain?",
        "Is the perioperative use of dexmedetomidine known to significantly improve postoperative outcomes in ERAS protocols?",
        "Does the incorporation of dexmedetomidine into ERAS protocols contribute to reduced opioid consumption in the Post anesthesia care unit (PACU)?",
        "Is dexmedetomidine classified as a selective alpha(2) agonist with unique analgesic properties compared to opioids?",
        "Are regional nerve blocks combined with dexmedetomidine used to improve postoperative outcomes in ERAS protocols?",
        "Does local infiltration of IV dexmedetomidine correlate with improved discharge times from the PACU?",
        "Is the use of dexmedetomidine as an adjuvant in epidurals intended to enhance local anesthetic sparing effects?"
    ]

    # Paths
    dataset_folder = f"./datasets/{dataset}"
    index_folder = os.path.join(dataset_folder, retriever_name, "indexes", f"{dataset}-index")
    faiss_index_path = os.path.join(index_folder, "corpus_index.faiss")
    doc_ids_path = os.path.join(index_folder, "doc_ids.pkl")
    corpus_file = os.path.join(dataset_folder, "corpus.json")

    # Load FAISS index and document IDs
    print("Loading FAISS index...")
    faiss_index = faiss.read_index(faiss_index_path)

    print("Loading document IDs...")
    with open(doc_ids_path, "rb") as f:
        doc_ids = pickle.load(f)

    # Load corpus for mapping document IDs to content
    print("Loading corpus...")
    with open(corpus_file, "r") as f:
        corpus = json.load(f)

    # Encode queries in a batch
    print("Encoding queries...")
    query_embeddings = encode_queries(queries, model)

    # Perform batch retrieval
    print(f"Performing search for top-{k} results for each query...")
    distances, indices = faiss_index.search(query_embeddings, k)

    # Map results back to corpus
    print("\nSearch Results:")
    for query_idx, query in enumerate(queries):
        print(f"Query {query_idx + 1}: {query}")
        print(f"Top-{k} results:")
        for rank, idx in enumerate(indices[query_idx], start=1):
            doc_id = doc_ids[idx]
            doc_content = corpus[doc_id]
            print(f"  Rank {rank}:")
            print(f"    Doc ID: {doc_id}")
            print(f"    Title: {doc_content.get('title', '')}")
            print(f"    Text Snippet: {doc_content.get('text', '')[:200]}...")
        print()


# Document encoding function
@torch.no_grad()
def encode_documents(corpus, model, batch_size=32):
    """
    Encode documents using provided retriever
    """
    doc_ids = list(corpus.keys())
    texts = [f"{corpus[doc_id]['title']} {corpus[doc_id]['text']}" for doc_id in doc_ids]
    embeddings = []

    print("Encoding documents...")
    for i in tqdm(range(0, len(texts), batch_size), desc="Encoding batches", unit="batch"):
        batch_texts = texts[i:i + batch_size]

        passage_embedding = model.encode_passage(batch_texts)
        embeddings.append(passage_embedding)

    embeddings = np.concatenate(embeddings, axis=0)  # Combine all batches into a single array
    return doc_ids, embeddings


def main(retriever_name: str, dataset: str, batch_size: int=32):
    # Paths
    dataset_folder = f"./datasets/{dataset}"
    corpus_file = os.path.join(dataset_folder, "corpus.json")
    selected_indices_file = os.path.join(dataset_folder, "selected_indices.json")
    index_folder = os.path.join(dataset_folder, retriever_name, "indexes", f"{dataset}-index")
    os.makedirs(index_folder, exist_ok=True)

    # Model setup
    ret_model = create_retriever(retriever_name, dataset)

    # Load corpus
    print("Loading corpus...")
    with open(corpus_file, "r") as f:
        corpus = json.load(f)

    # Load selected indices
    print("Loading selected indices...")
    with open(selected_indices_file, "r") as f:
        selected_indices = json.load(f)

    # Exclude non-member documents
    non_mem_indices = set(selected_indices["non_mem_indices"])
    filtered_corpus = {doc_id: content for doc_id, content in corpus.items() if doc_id not in non_mem_indices}
    print(f"Filtered corpus size: {len(filtered_corpus)} documents")

    # Encode documents
    doc_ids, doc_embeddings = encode_documents(filtered_corpus, ret_model, batch_size=batch_size)

    # Save document embeddings using FAISS
    dimension = doc_embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)  # Inner product index
    index.add(doc_embeddings)

    # Save FAISS index and document IDs
    faiss_index_path = os.path.join(index_folder, "corpus_index.faiss")
    doc_ids_path = os.path.join(index_folder, "doc_ids.pkl")

    faiss.write_index(index, faiss_index_path)
    with open(doc_ids_path, "wb") as f:
        pickle.dump(doc_ids, f)

    print(f"Indexing completed. Index saved to: {index_folder}")

    # Test the retriever
    test(ret_model, k=3)


def load_config(config_path):
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config['test_params']


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the main script with a specified config file.")
    parser.add_argument('--config', type=str, required=True, help='Path to the configuration YAML file.')
    args = parser.parse_args()

    test_params = load_config(args.config)

    # TODO: Handle ColBERT

    retriever_name = test_params['retriever']
    dataset = test_params['eval_dataset']

    # Generate and cache passage embeddings
    batch_size = 32
    main(retriever_name=retriever_name, dataset=dataset, batch_size=batch_size)
