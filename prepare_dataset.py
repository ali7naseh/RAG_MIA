from beir import util
import os
import json
import random

num=1000

datasets = ['nfcorpus',
            # 'trec-covid',
            # 'fiqa'
            ]

# for dataset in datasets:
#     url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{}.zip".format(dataset)
#     out_dir = os.path.join(os.getcwd(), "datasets")
#     data_path = os.path.join(out_dir, dataset)
#     if not os.path.exists(data_path):
#         data_path = util.download_and_unzip(url, out_dir)

# os.system('rm datasets/*.zip')

# Function to randomly select 200 mem and 200 non-mem indices from a JSONL dataset
def select_mem_non_mem_indices(dataset_path):
    corpus_file = os.path.join(dataset_path, "corpus.jsonl")
    

    doc_ids = []
    with open(corpus_file, 'r') as f:
        for line in f:
            doc = json.loads(line)
            doc_ids.append(doc['_id'])  # Assuming the ID is stored in the '_id' field
    
    random.shuffle(doc_ids)
    mem_indices = random.sample(doc_ids, num)
    remaining_doc_ids = list(set(doc_ids) - set(mem_indices))
    non_mem_indices = random.sample(remaining_doc_ids, num)
    
    return mem_indices, non_mem_indices


for dataset in datasets:
    dataset_path = os.path.join("datasets", dataset)
    
    mem_indices, non_mem_indices = select_mem_non_mem_indices(dataset_path)
    # Save the indices to a file
    output_file = os.path.join(dataset_path, "selected_indices.json")
    with open(output_file, "w") as f:
        json.dump({"mem_indices": mem_indices, "non_mem_indices": non_mem_indices}, f, indent=4)

    print(f"Saved {num} mem and {num} non-mem indices for {dataset} in {output_file}")