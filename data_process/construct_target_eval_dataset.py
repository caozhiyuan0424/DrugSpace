"""Join the released target-triplet IDs to locally licensed DrugBank descriptions."""
import argparse
import json
import re
from pathlib import Path

import pandas as pd

from parse_drugbank import cohorts


def mask_genes(text, genes):
    if not genes:
        return text
    pattern = r"\b(?:" + "|".join(re.escape(gene) for gene in sorted(set(genes), key=lambda x: (-len(x), x))) + r")\b"
    return re.sub(pattern, "[MASK]", text, flags=re.I)


def construct(reference_csv, later_csv, triplet_ids):
    reference, queries = cohorts(reference_csv, later_csv)
    later = pd.read_csv(later_csv, dtype=str, keep_default_na=False).set_index("id")
    reference = reference.set_index("id")
    query_ids = set(queries.id)
    ids = pd.read_csv(triplet_ids, dtype=str, keep_default_na=False)
    if ids.empty or ids.duplicated().any():
        raise ValueError("Triplet IDs must be nonempty and unique")
    rows = []
    for item in ids.to_dict("records"):
        anchor, positive, negative = (item[key] for key in ("anchor_drug_id", "related_drug_id", "unrelated_drug_id"))
        if anchor not in query_ids or positive not in reference.index or negative not in reference.index:
            raise ValueError(f"Triplet does not match the temporal split: {item}")
        row = {"drug1": anchor, "drug_related": positive, "drug_unrelated": negative,
               "shared_target": item["shared_target_id"]}
        for label, drug_id in (("drug1", anchor), ("drug_related", positive), ("drug_unrelated", negative)):
            source = later.loc[drug_id] if label == "drug1" else reference.loc[drug_id]
            text = source["description_raw" if label == "drug1" else "description"]
            if not text.strip():
                raise ValueError(f"Missing description for {drug_id}")
            row[f"{label}_description_masked"] = mask_genes(text, json.loads(later.loc[drug_id, "target_genes"]))
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", required=True)
    parser.add_argument("--later", required=True)
    parser.add_argument("--triplet-ids", required=True, help="target_based_evaluation_triplets.csv from Zenodo")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = construct(args.reference, args.later, args.triplet_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(args.output, index=False)
    print(f"Saved {len(data)} target triplets to {args.output}")


if __name__ == "__main__":
    main()
