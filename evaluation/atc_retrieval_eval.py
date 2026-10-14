"""Multi-label ATC level-1 retrieval: any shared category counts as relevant."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_process.parse_drugbank import cohorts
from embed import Encoder, model_args


def prepare(reference_csv, later_csv):
    reference, query = cohorts(reference_csv, later_csv)
    for frame in (reference, query):
        frame["categories"] = frame.atc_codes.map(lambda value: {code[0] for code in json.loads(value) if code})
    counts = Counter(code for categories in reference.categories for code in categories)
    valid = {code for code, count in counts.items() if count >= 10}
    for frame in (reference, query):
        frame["categories"] = frame.categories.map(lambda codes: codes & valid)
    reference = reference[reference.categories.map(bool)].reset_index(drop=True)
    query = query[query.categories.map(bool)].reset_index(drop=True)
    if len(reference) < 10 or query.empty:
        raise ValueError("Need at least ten labeled references and one labeled query")
    return reference, query


def retrieval_metrics(scores, relevant):
    ranked = relevant[np.argsort(-scores)]
    positions = np.flatnonzero(ranked)
    if not len(positions):
        raise ValueError("Query has no relevant reference drug")
    return {**{f"Hits@{k}": float(ranked[:k].any()) for k in (1, 5, 10)},
            **{f"P@{k}": float(ranked[:k].mean()) for k in (5, 10)},
            "MRR": float(1 / (positions[0] + 1))}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--later", required=True)
    parser.add_argument("--output", required=True, type=Path)
    model_args(parser)
    args = parser.parse_args()
    reference, query = prepare(args.reference, args.later)
    encoder = Encoder(args.model, args.model_path, args.adapter, args.batch_size)
    refs = encoder.encode(reference.description)
    queries = encoder.encode(query.description)
    rows = []
    for (_, row), embedding in zip(query.iterrows(), queries):
        relevant = np.array([bool(row.categories & codes) for codes in reference.categories])
        rows.append({"id": row.id, **retrieval_metrics(refs @ embedding, relevant)})
    per_query = pd.DataFrame(rows)
    result = {"n_reference": len(reference), "n_query": len(query),
              **per_query.drop(columns="id").mean().to_dict()}
    args.output.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(args.output / "scores.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
