# RAG_MIA

This project implements membership inference attacks (MIA) on Retrieval-Augmented Generation (RAG).

## Setup

Create the Conda environment using the provided `environment.yml` file:

`conda env create -f environment.yml`

Also make sure you have SimpleParsing installed

`pip install git+https://github.com/lebrice/SimpleParsing.git`

## Running the MIA Pipeline

Start by preparing the relevant datasets

`python prepare_dataset.py`

Then, generate passage embeddings:

`python generate_index.py --config ./configs/nfcorpus.yml`

To run the attack pipeline, use the following command:

`python run_mia.py --config ./configs/nfcorpus.yml`

After the script runs, it generates a `result file` (see details below).

## Configuration Parameters

The configuration file (e.g., `./configs/nfcorpus.yml`) contains the following parameters:

- M: The number of selected members and non-members.
- N: Total number of questions generated for each target document.
- top_k: The number of questions selected from the N generated questions for the attack.
- from_ckpt: If true, starts from the previously generated result file instead of running the attack again.
- post_filter: Generates a result file from a superset without rerunning the entire attack process. For example, if `mia-nfcorpus-llama3_70b-Top15-M250-N15.json` exists, you can generate `mia-nfcorpus-llama3_70b-Top10-M250-N15.json` by setting top_k: 10 and post_filter: true.

## Workflow Overview

The run_mia.py script orchestrates the following steps, executing the full attack pipeline through main_mia.py:

### 1. Member and Non-Member Selection

The script selects members and non-members based on the `selected_indices.json` file (*please download it from this repository to ensure consistency*). It initializes a result file in the `result/target_docs` directory with the following format:
`mia-[dataset]-[model]-Top[top_k]-M[M]-N[N].json`

Example file content:
```json
{
    "MED-906": {
        "text": "Annatto dye is ...",
        "title": "Anaphylaxis to annatto dye: a case report.",
        "mem": "yes"
    },
    ...
}
```


### 2. Question Generation

Questions are generated for each document using target_data_with_questions.json (provided by Ali) and stored in the result file:

```json
{
    "MED-906": {
        ...
        "all_questions": "Is annatto dye used as a food coloring in cheeses?\nCan annatto dye cause urticaria as an adverse reaction?..."
    },
    ...
}
```

### 3. Selecting Top Questions

The script selects the top k questions (e.g., top-2) for each document:
```json
{
    "MED-906": {
        ...
        "questions": [
            "Is annatto dye a potential cause of anaphylaxis?",
            "Is annatto dye an orange-yellow color?"
        ],
        ...
    },
    ...
}
```
### 4. Generating Ground-Truth Answers
The ground-truth answers for each question are stored in the result file:
```json
{
    "MED-906": {
        ...
        "answers": [
            "Yes.",
            "Yes."
        ],
        ...
    },
    ...
}
```
### 5. Retrieving Documents

Documents are retrieved for each question using a retriever (e.g., [ColBERT](https://github.com/stanford-futuredata/ColBERT)). The retrieved document IDs are recorded as shown below:

```json
{
    "MED-906": {
        "text": "Annatto dye is ...",
        "title": "Anaphylaxis to annatto dye: a case report.",
        "mem": "yes",
        "retrieved_doc_ids": [
            ["MED-906", "MED-2355", "MED-2356"],
            ["MED-906", "MED-2618", "MED-2616"]
        ]
    },
    ...
}
```
### 6. Querying the RAG Model
Responses from the RAG model are saved:
```json
{
    "MED-906": {
        ...
        "llm_responses": [
            "Yes.",
            "Yes.\n\n"
        ]
    },
    ...
}
```

## Post-Processing

After completing the pipeline, various analyses can be performed on the result file:
1. Compare the `answers` with the `llm_responses` to compute the accuracy of RAG's responses and evaluate the MIA's effectiveness.
2. Observe whether the target document is retrieved by the corresponding questions.

Example analysis is provided in main.ipynb.