# DrugSpace: A literature-based drug embedding resource for biomedical research

## Tutorial

### Environment / Installation

```bash
# create and activate a fresh environment
conda create -n your_env python=3.12
conda activate your_env

# install required packages
pip install -r requirements.txt
```

### Quick start

```python
from llm2vec import LLM2Vec
import torch
import torch.nn.functional as F

l2v = LLM2Vec.from_pretrained(
    base_model_name_or_path="cczzzyyy/DrugSpace-mntp-8B",
    peft_model_name_or_path="cczzzyyy/DrugSpace-full-lora",
    enable_bidirectional=True,
    device_map="cuda" if torch.cuda.is_available() else "cpu",
    torch_dtype=torch.bfloat16,
)

texts = [
    "Luminatrex is a small-molecule agent indicated for chronic inflammatory disorders that selectively inhibits intracellular signaling involved in cytokine production.",
    "Glycemorin is a synthetic therapeutic used for metabolic dysfunction that is thought to improve insulin sensitivity and reduce hepatic glucose output.",
]
emb = l2v.encode(texts, show_progress_bar=True)
print("Embedding shape:", emb.shape)  # (num_texts, hidden_dim)

# Example: cosine similarity between two drug descriptions
sim = F.cosine_similarity(
    torch.tensor(emb[0]).unsqueeze(0),
    torch.tensor(emb[1]).unsqueeze(0),
).item()
print("Cosine similarity:", round(sim, 4))
```

### Model variants

- `cczzzyyy/DrugSpace-mntp-8B`: MNTP-trained base model built on `Llama-3.1-8B-Instruct`.
- `cczzzyyy/DrugSpace-full-lora`: contrastive-learning LoRA model on top of `cczzzyyy/DrugSpace-mntp-8B`; recommended for general embedding generation.
- `cczzzyyy/DrugSpace-full-lora-eval`: evaluation-oriented checkpoint trained only on pre-2020 drugs.

### Input format recommendations

- Provide complete English drug descriptions.
- Keep wording clinically meaningful and specific; avoid placeholder or template-only text when possible.
- For retrieval/similarity tasks, use consistent writing style across query and candidate descriptions to reduce stylistic noise.

### Output interpretation

- `l2v.encode(...)` returns one embedding vector per input text, typically with shape `(N, D)`.
- Higher cosine similarity generally indicates stronger semantic/functional relatedness between two drug descriptions.
- For retrieval, rank candidate drugs by cosine similarity to the query embedding and report metrics such as Recall@K or MRR.

### Released `.npy` drug embeddings

We also provide precomputed drug embeddings in `.npy` format as part of the release.

```python
import numpy as np

emb = np.load("path/to/drug_embeddings.npy")
print(emb.shape)  # (num_drugs, hidden_dim)
```

### Generate embeddings from raw text

Use the same `encode` API to generate embeddings for new descriptions:

```python
new_emb = l2v.encode(
    [
        "Drug A description ...",
        "Drug B description ...",
    ],
    show_progress_bar=True,
)
```

## Evaluation

### 1) Data processing

For evaluation, we use two DrugBank XML snapshots separated by a user-defined **cutoff date**:

- An earlier snapshot, used to define the **reference drug pool**
- A later snapshot, used to identify **anchor drugs** introduced after the cutoff date

In our experiments, the cutoff date is **2020-01-04**. You can modify this date to construct alternative evaluation splits.

Parse each XML file with the following commands:

```bash
# Earlier DrugBank snapshot for the reference pool (used in our paper: v5.1.5)
python data_process/parse_drugbank.py \
  --input "data/full database v5_1_5.xml" \
  --output data/full_data_ref.csv
```

```bash
# Later DrugBank snapshot for anchor drugs (used in our paper: v5.1.14)
python data_process/parse_drugbank.py \
  --input "data/full database v5_1_14.xml" \
  --output data/full_data_anc.csv
```

### 2) Drug similarity analysis

This analysis evaluates whether the embedding space captures pharmacologically meaningful similarity by comparing each anchor drug against the reference pool based on target-related information.

Create the evaluation dataset:

```bash
python data_process/construct_target_eval_dataset.py \
  --xml_path "data/full database v5_1_14.xml"
```

Expected output:

- The resulting dataframe should contain **1436** rows.
- Among them, **482** triplets have non-empty text in all required description fields and are used for evaluation.

Then run the model on the evaluation dataset:

```bash
python evaluation/target_similarity_eval.py
```

### 3) ATC-based retrieval

This task evaluates whether DrugSpace embeddings can retrieve drugs that share similar therapeutic categories (ATC codes), using nearest-neighbor retrieval in the embedding space.

```bash
python evaluation/atc_retrieval_eval.py
```

### 4) Input perturbation

This task measures robustness by applying controlled perturbations to drug descriptions (e.g., wording or surface-form changes) and checking how stable the embedding-based similarity remains.

```bash
python evaluation/robustness_eval.py
```
