"""Sentence shuffle and span-deletion robustness over five fixed seeds."""
import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import t

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_process.parse_drugbank import cohorts
from embed import Encoder, model_args, skip_untrained_drugbank

SEEDS = (42, 43, 44, 45, 46)
STRENGTHS = (10, 15, 25)


def rng_for(drug_id, condition, seed):
    digest = hashlib.sha256(f"{seed}|{condition}|{drug_id}".encode()).digest()
    return random.Random(int.from_bytes(digest[:8], "big"))


def shuffle_text(text, doc, rng):
    spans = list(doc.sents)
    sentences = [text[span.start_char:span.end_char] for span in spans]
    if len(sentences) < 2 or len(set(sentences)) < 2:
        return None
    separators = [text[a.end_char:b.start_char] for a, b in zip(spans, spans[1:])]
    prefix, suffix = text[:spans[0].start_char], text[spans[-1].end_char:]
    while True:
        order = list(range(len(sentences)))
        rng.shuffle(order)
        if [sentences[i] for i in order] == sentences:
            continue
        changed = prefix + "".join(sentences[i] + (separators[slot] if slot < len(separators) else "")
                                   for slot, i in enumerate(order)) + suffix
        if changed != text:
            return changed


def deletion_order(n_words, n_delete, rng):
    chosen, order = set(), []
    while len(order) < n_delete:
        length = min(rng.randint(1, 3), n_delete - len(order))
        for attempt in range(51):
            if attempt == 50:
                starts = [i for i in range(n_words - length + 1)
                          if all(i + j not in chosen for j in range(length))]
                if not starts:
                    length = 1
                    starts = [i for i in range(n_words) if i not in chosen]
                start = rng.choice(starts)
                break
            start = rng.randrange(n_words - length + 1)
            if all(start + j not in chosen for j in range(length)):
                break
        for i in range(start, start + length):
            chosen.add(i)
            order.append(i)
    return order


def variants(query):
    import spacy

    nlp = spacy.load("en_core_sci_sm")
    rows = []
    for row, doc in zip(query.itertuples(index=False), nlp.pipe(query.description, batch_size=64)):
        words = row.description.split()
        for seed in SEEDS:
            shuffled = shuffle_text(row.description, doc, rng_for(row.id, "shuffle", seed))
            if shuffled is not None:
                rows.append((row.id, "shuffle", seed, shuffled))
            if len(words) < 10:
                continue
            order = deletion_order(len(words), len(words) * max(STRENGTHS) // 100, rng_for(row.id, "span", seed))
            for strength in STRENGTHS:
                removed = set(order[:len(words) * strength // 100])
                changed = " ".join(word for i, word in enumerate(words) if i not in removed)
                rows.append((row.id, f"span_{strength}", seed, changed))
    return pd.DataFrame(rows, columns=["id", "condition", "seed", "text"])


def top10(scores):
    # Break exact ties by reference CSV order, retaining every drug candidate.
    return set(np.argsort(-scores, kind="stable")[:10])


def summarize(scores):
    # Average seeds within each drug before estimating uncertainty across drugs.
    per_drug = scores.groupby(["condition", "id"])[["overlap@10", "self_similarity"]].mean()
    rows = []
    for condition, group in per_drug.groupby(level="condition"):
        if len(group) < 2:
            raise ValueError(f"Need two eligible drugs for a confidence interval: {condition}")
        for metric in group.columns:
            values = group[metric].to_numpy()
            mean = values.mean()
            halfwidth = t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
            rows.append({"condition": condition, "metric": metric, "n_drugs": len(values),
                         "mean": float(mean), "ci95_low": float(mean - halfwidth),
                         "ci95_high": float(mean + halfwidth)})
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--later", required=True)
    parser.add_argument("--output", required=True, type=Path)
    model_args(parser)
    args = parser.parse_args()
    if skip_untrained_drugbank(args):
        return
    reference, query = cohorts(args.reference, args.later)
    reference = reference[reference.description.str.len() >= 10]
    query = query[query.description.str.len() >= 10]
    if len(reference) < 10 or query.empty:
        raise ValueError("Need ten reference descriptions and at least one query")
    changed = variants(query)
    if changed.empty:
        raise ValueError("No eligible perturbations")
    encoder = Encoder(args.model, args.model_path, args.adapter, args.batch_size)
    refs = encoder.encode(reference.description)
    originals = dict(zip(query.id, encoder.encode(query.description)))
    neighbors = {drug_id: top10(refs @ vector) for drug_id, vector in originals.items()}
    rows = []
    for (condition, seed), group in changed.groupby(["condition", "seed"], sort=False):
        embeddings = encoder.encode(group.text)
        for drug_id, vector in zip(group.id, embeddings):
            before, after = neighbors[drug_id], top10(refs @ vector)
            rows.append({"id": drug_id, "condition": condition, "seed": int(seed),
                         "overlap@10": len(before & after) / len(before | after),
                         "self_similarity": float(originals[drug_id] @ vector)})
        print(f"Completed {condition}, seed {seed}: {len(group)} drugs", flush=True)
    scores = pd.DataFrame(rows)
    result = {"n_reference": len(reference), "n_query": len(query), "seeds": SEEDS,
              "metrics": summarize(scores)}
    args.output.mkdir(parents=True, exist_ok=True)
    scores.to_csv(args.output / "scores.csv", index=False)
    (args.output / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
