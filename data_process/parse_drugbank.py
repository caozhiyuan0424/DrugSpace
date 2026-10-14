"""Parse a licensed DrugBank XML and optionally generate fixed training triplets."""
import argparse
import json
import random
import re
from pathlib import Path
from xml.etree import ElementTree as ET

import pandas as pd

FIELDS = ("description", "indication", "pharmacodynamics", "mechanism-of-action", "toxicity", "metabolism")
PRONOUNS = ("this drug", "the compound", "the agent", "the medication", "the substance",
            "this medication", "this compound", "this agent", "this substance",
            "the pharmaceutical", "this pharmaceutical")
CUTOFF = "2020-01-03"


def clean(text):
    return re.sub(r"\[[A-Za-z0-9, ]+\]", "", text)


def parse_drugbank(path):
    events = ET.iterparse(path, events=("start", "end"))
    _, root = next(events)
    ns = {"d": root.tag.split("}")[0][1:]}
    rows, depth = [], 1
    for event, drug in events:
        if event == "start":
            depth += 1
            continue
        if depth == 2 and drug.tag == f"{{{ns['d']}}}drug":
            def value(path):
                return (drug.findtext(path, default="", namespaces=ns) or "").strip()

            ids = drug.findall("d:drugbank-id", ns)
            primary = next((x.text for x in ids if x.get("primary") == "true"), ids[0].text if ids else "")
            if not primary:
                raise ValueError("Drug record without an ID")
            codes = [x.get("code") for x in drug.findall("d:atc-codes/d:atc-code", ns) if x.get("code")]
            targets, genes = [], []
            for target in drug.findall("d:targets/d:target", ns):
                target_id = target.findtext("d:id", namespaces=ns)
                if target_id:
                    targets.append(target_id)
                if target.findtext("d:organism", namespaces=ns) == "Humans":
                    genes.extend(x.text for x in target.findall("d:polypeptide/d:gene-name", ns) if x.text)
            smiles = next((p.findtext("d:value", default="", namespaces=ns) or ""
                           for p in drug.findall("d:calculated-properties/d:property", ns)
                           if p.findtext("d:kind", namespaces=ns) == "SMILES"), "")
            rows.append({"id": primary, "name": clean(value("d:name")),
                         "date_created": drug.get("created", ""), "date_updated": drug.get("updated", ""),
                         **{field: clean(value(f"d:{field}")) for field in FIELDS},
                         "description_raw": value("d:description"), "smiles": smiles,
                         "atc_codes": json.dumps(codes), "target_ids": json.dumps(targets),
                         "target_genes": json.dumps(genes)})
            root.clear()
        depth -= 1
    result = pd.DataFrame(rows)
    if result.empty or result.id.duplicated().any():
        raise ValueError("XML must contain drugs with unique IDs")
    return result


def cohorts(reference_csv, later_csv):
    reference = pd.read_csv(reference_csv, dtype=str, keep_default_na=False)
    later = pd.read_csv(later_csv, dtype=str, keep_default_na=False)
    if reference.id.duplicated().any() or later.id.duplicated().any():
        raise ValueError("Duplicate DrugBank IDs")
    created = pd.to_datetime(later.date_created, errors="coerce")
    query = later[~later.id.isin(reference.id) & created.gt(pd.Timestamp(CUTOFF))].copy()
    return reference, query


def fixed_triplets(data):
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    rng = random.Random(42)
    texts, patterns, fingerprints = {}, {}, {}
    for row in data.to_dict("records"):
        views, seen = [], set()
        for field in FIELDS:
            text = " ".join(row[field].split())
            if len(text.split()) >= 20 and text.lower() not in seen:
                views.append((field, text))
                seen.add(text.lower())
        if len(views) >= 2:
            texts[row["id"]] = views
            name = " ".join(row["name"].split())
            patterns[row["id"]] = re.compile(r"(?<!\w)" + re.escape(name) + r"(?!\w)", re.I) if name else None
        mol = Chem.MolFromSmiles(row["smiles"]) if row["smiles"] else None
        if mol is not None:
            fingerprints[row["id"]] = AllChem.GetMorganFingerprintAsBitVect(mol, 2)
    eligible = list(texts)
    if len(eligible) < 2:
        raise ValueError("At least two drugs with two eligible text fields are needed")
    fp_ids = list(fingerprints)
    fps = list(fingerprints.values())
    hard = {}
    for drug_id in eligible:
        if drug_id in fingerprints:
            scores = DataStructs.BulkTanimotoSimilarity(fingerprints[drug_id], fps)
            hard[drug_id] = [other for other, score in zip(fp_ids, scores)
                             if other != drug_id and other in texts and 0.6 <= score <= 0.95]

    def mask(text, drug_id, keep, use_token):
        pattern = patterns[drug_id]
        if pattern is None or not text or rng.random() < keep:
            return text
        return pattern.sub("[MASK]" if use_token else rng.choice(PRONOUNS), text)

    # Advance through the first sampling round, then retain the fixed second round.
    for round_name, keep in (("epoch1", 0.3), ("epoch2", 0.2)):
        triplets = []
        for drug_id in eligible:
            anchor = rng.choice(texts[drug_id])
            positive = rng.choice([view for view in texts[drug_id] if view != anchor])
            positive_masked = mask(positive[1], drug_id, keep, rng.random() < 0.5)
            candidates = hard.get(drug_id) or [other for other in eligible if other != drug_id]
            negative_id = rng.choice(candidates)
            negative = rng.choice(texts[negative_id])
            negative_masked = mask(negative[1], negative_id, keep, rng.random() < 0.5)
            triplets.append({"query": anchor[1], "positive": positive_masked, "negative": negative_masked,
                             "meta": {"anchor_col": anchor[0], "positive_col": positive[0],
                                      "negative_col": negative[0], "anchor_id": drug_id,
                                      "negative_id": negative_id, "epoch": round_name}})
    return triplets


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", type=Path, help="Parsed CSV")
    parser.add_argument("--triplets", type=Path, help="Fixed training JSONL")
    args = parser.parse_args()
    if not args.output and not args.triplets:
        parser.error("Provide --output and/or --triplets")
    data = parse_drugbank(args.input)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        data.to_csv(args.output, index=False)
        print(f"Saved {len(data)} drugs to {args.output}", flush=True)
    if args.triplets:
        triplets = fixed_triplets(data)
        args.triplets.parent.mkdir(parents=True, exist_ok=True)
        with args.triplets.open("w") as stream:
            for row in triplets:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Saved {len(triplets)} fixed triplets to {args.triplets}")


if __name__ == "__main__":
    main()
