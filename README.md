# DrugSpace

Drug embeddings from biomedical literature and drug descriptions. The public release uses openFDA labels and includes 1,867 drug embeddings with 4,096 dimensions.

- [Models](https://huggingface.co/z-cao/DrugSpace-8B)
- [Data, embeddings and reproduction resources](https://doi.org/10.5281/zenodo.23022876)

![DrugSpace workflow: literature MNTP followed by openFDA or DrugBank contrastive alignment](images/fig1.png)

## Install

Use Python 3.11 and a CUDA-capable PyTorch environment:

```bash
pip install -r requirements.txt
```

Run commands from the repository root. On a cluster, run data processing and model workloads inside the appropriate scheduler allocation.

Optional EmbeddingGemma support requires a separate environment with `transformers==4.57.1` and `sentence-transformers==5.1.1`.

## Use DrugSpace

```python
from embed import Encoder

encoder = Encoder("drugspace")
vectors = encoder.encode([
    "A selective serotonin reuptake inhibitor used to treat depression.",
    "A beta-adrenergic receptor antagonist used to treat hypertension.",
])
print(vectors.shape)  # (2, 4096)
print(vectors[0] @ vectors[1])  # cosine similarity
```

For a CSV containing a `description` column:

```bash
python embed.py --input data/drugs.csv --output outputs/embeddings
```

The output contains normalized `embeddings.npy` and `metadata.csv` in matching row order. See [Tutorial.ipynb](Tutorial.ipynb) for nearest-neighbor retrieval from the released embeddings.

## Reproduce the DrugBank model

Users with DrugBank access can generate the fixed triplets from their licensed XML and train Stage II locally. For the evaluation model, use DrugBank v5.1.5 (2020-01-03) and the pre-2020 MNTP base:

```bash
python data_process/parse_drugbank.py \
  --input "data/drugbank_v5.1.5.xml" \
  --output data/reference.csv --triplets data/triplets.jsonl

pip install flash-attn==2.7.4.post1 --no-build-isolation

torchrun --standalone --nproc_per_node=2 train.py \
  --triplets data/triplets.jsonl --output outputs/drugbank-eval
```

The commands generate a fixed triplet file and save the trained adapter to `outputs/drugbank-eval`.

## Evaluation

Parse both licensed DrugBank snapshots before running evaluation:

```bash
python data_process/parse_drugbank.py \
  --input "data/drugbank_v5.1.5.xml" --output data/reference.csv
python data_process/parse_drugbank.py \
  --input "data/drugbank_v5.1.14.xml" --output data/later.csv
```

**Target similarity.** Extract `target_based_evaluation_triplets.csv` from Zenodo's `evaluation_resources.zip`, then build the evaluation input:

```bash
python data_process/construct_target_eval_dataset.py \
  --reference data/reference.csv --later data/later.csv \
  --triplet-ids data/target_based_evaluation_triplets.csv --output data/target_triplets.csv
python evaluation/target_similarity_eval.py \
  --input data/target_triplets.csv --output results/target
```

Reports accuracy and similarity margins with 95% confidence intervals.

**ATC retrieval.** Reports Hits@1/5/10, Precision@5/10 and MRR:

```bash
python evaluation/atc_retrieval_eval.py \
  --reference data/reference.csv --later data/later.csv --output results/atc
```

**Robustness.** The sentence splitter is included in `requirements.txt`:

```bash
python evaluation/robustness_eval.py \
  --reference data/reference.csv --later data/later.csv --output results/robustness
```

Measures retrieval and embedding stability under sentence shuffling and span deletion.

Append `--model bert` to select a baseline, or `--model drugbank --adapter outputs/drugbank-eval` to evaluate your trained DrugBank model. Use `--help` for other options. Each task saves `scores.csv` and `metrics.json` under its output directory.

## Interactive visualization

Explore the 1,867 released openFDA drug embeddings in [MedViz](https://medviz.org/), colored by GSRS substance class.

![MedViz visualization of DrugSpace openFDA drug embeddings colored by GSRS substance class](images/fig2.png)
