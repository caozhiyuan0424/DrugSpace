"""Target-triplet similarity with an anchor-clustered 95% bootstrap interval."""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from embed import Encoder, model_args, skip_untrained_drugbank


def summarize(data, replicates=10000):
    summaries = []
    for _, group in data.groupby("drug1", sort=True):
        a, b = group.sim_rel.to_numpy(dtype=np.float64), group.sim_unrel.to_numpy(dtype=np.float64)
        summaries.append([len(a), (a > b).sum(), (a - b).sum(), min(a.min(), b.min()), max(a.max(), b.max())])
    summaries = np.asarray(summaries, dtype=np.float64)

    def estimate(values):
        count = values[:, 0].sum()
        span = values[:, 4].max() - values[:, 3].min()
        if span == 0:
            raise ValueError("Cannot normalize identical similarities")
        raw = values[:, 2].sum() / count
        return values[:, 1].sum() / count, raw, raw / span

    point = estimate(summaries)
    rng = random.Random(42)
    bootstrap = np.array([estimate(summaries[rng.choices(range(len(summaries)), k=len(summaries))])
                          for _ in range(replicates)])
    ci = np.quantile(bootstrap, [0.025, 0.975], axis=0)
    return {"n_triplets": len(data), "n_anchors": len(summaries), "bootstrap_replicates": replicates,
            "bootstrap_seed": 42,
            **{key: {"mean": float(point[i]), "ci95": ci[:, i].tolist()}
               for i, key in enumerate(("accuracy", "margin_raw", "margin_normalized"))}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True, type=Path)
    model_args(parser)
    args = parser.parse_args()
    if skip_untrained_drugbank(args):
        return
    data = pd.read_csv(args.input, dtype=str, keep_default_na=False)
    columns = [f"{key}_description_masked" for key in ("drug1", "drug_related", "drug_unrelated")]
    if data.empty or not data[columns].apply(lambda col: col.str.strip().ne("").all()).all():
        raise ValueError("All triplet descriptions must be nonempty")
    texts = list(dict.fromkeys(data[columns].to_numpy().ravel()))
    encoder = Encoder(args.model, args.model_path, args.adapter, args.batch_size)
    embeddings = dict(zip(texts, encoder.encode(texts)))
    a, b, c = [np.stack([embeddings[text] for text in data[column]]) for column in columns]
    data["sim_rel"] = np.einsum("ij,ij->i", a, b)
    data["sim_unrel"] = np.einsum("ij,ij->i", a, c)
    result = summarize(data)
    args.output.mkdir(parents=True, exist_ok=True)
    data.drop(columns=columns).to_csv(args.output / "scores.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
