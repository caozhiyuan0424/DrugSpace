#!/usr/bin/env python3
"""
Construct target-based evaluation dataset.

For each drug d1 from the anchor XML file:
  - Find d2: drugs from the reference set that share at least one target with d1
  - Find d3: drugs from the reference set that share NO targets with d1 or d2
  - Create triplets: (d1, d2, d3) for each shared target

Output includes:
  - anchor drug metadata from the anchor XML
  - reference drug metadata (including description) from the reference CSV
  - masked descriptions using target gene names extracted from XML
"""

import argparse
import os
import random
import re
from collections import defaultdict

import pandas as pd
import xmltodict
from tqdm import tqdm


def get_drugbank_id(drug):
    """Extract primary DrugBank ID from drug entry."""
    drugbank_id = drug.get("drugbank-id")
    if isinstance(drugbank_id, list):
        for item in drugbank_id:
            if isinstance(item, dict) and item.get("@primary") == "true":
                return item.get("#text")
        if drugbank_id:
            first = drugbank_id[0]
            return first.get("#text") if isinstance(first, dict) else first
    elif isinstance(drugbank_id, dict):
        return drugbank_id.get("#text")
    return None


def extract_target_ids(drug):
    """Extract all target IDs from a drug entry."""
    targets = drug.get("targets")
    if not targets:
        return []

    target_entry = targets.get("target")
    if not target_entry:
        return []

    if isinstance(target_entry, dict):
        target_list = [target_entry]
    else:
        target_list = target_entry

    target_ids = []
    for t in target_list:
        target_id = t.get("id")
        if target_id:
            target_ids.append(target_id)

    return list(set(target_ids))


def extract_target_gene_names(drug):
    """Extract all human target gene names from a drug entry."""
    targets = drug.get("targets")
    if not targets:
        return []

    target_entry = targets.get("target")
    if not target_entry:
        return []

    if isinstance(target_entry, dict):
        target_list = [target_entry]
    else:
        target_list = target_entry

    gene_names = []
    for t in target_list:
        if t.get("organism") != "Humans":
            continue

        poly = t.get("polypeptide")
        if isinstance(poly, dict):
            gene_name = poly.get("gene-name")
            if gene_name:
                gene_names.append(gene_name)
        elif isinstance(poly, list):
            for p in poly:
                gene_name = p.get("gene-name")
                if gene_name:
                    gene_names.append(gene_name)

    return list(set(gene_names))


def mask_targets_in_text(text, target_gene_names):
    """Mask target gene names in text by replacing them with [MASK]."""
    if not text or not target_gene_names:
        return text if text else ""

    masked_text = text
    for gene_name in sorted(set(target_gene_names), key=len, reverse=True):
        if not gene_name:
            continue
        pattern = r"\b" + re.escape(gene_name) + r"\b"
        masked_text = re.sub(pattern, "[MASK]", masked_text, flags=re.IGNORECASE)

    return masked_text


def parse_drugs_from_xml(xml_path):
    """Parse all drugs from XML and extract target + metadata information."""
    print(f"Reading XML file: {xml_path}")
    with open(xml_path, "r") as file:
        xml_data = file.read()

    print("Parsing XML...")
    data_dict = xmltodict.parse(xml_data)
    drugs = data_dict["drugbank"]["drug"]
    print(f"Total drugs in XML: {len(drugs)}")

    drug_info = {}
    for drug in tqdm(drugs, desc="Processing drugs"):
        drugbank_id = get_drugbank_id(drug)
        if not drugbank_id:
            continue

        created_date = drug.get("@created", "")
        target_ids = extract_target_ids(drug)
        target_gene_names = extract_target_gene_names(drug)

        if target_ids:
            description = drug.get("description", "") or ""
            drug_info[drugbank_id] = {
                "created_date": created_date,
                "target_ids": target_ids,
                "target_gene_names": target_gene_names,
                "name": drug.get("name", "") or "",
                "description": description,
                "description_masked": mask_targets_in_text(description, target_gene_names),
            }

    print(f"Drugs with targets in XML: {len(drug_info)}")
    return drug_info


def load_reference_info(drugs_before_cutoff_path, drug_info):
    """
    Load reference drug IDs, infer cutoff date from the reference CSV,
    and build a metadata dictionary for reference drugs.
    """
    print(f"Loading reference set from: {drugs_before_cutoff_path}")
    df_ref = pd.read_csv(drugs_before_cutoff_path)

    required_cols = {"id", "date_created"}
    missing_cols = required_cols - set(df_ref.columns)
    if missing_cols:
        raise ValueError(
            f"Missing required columns in {drugs_before_cutoff_path}: {sorted(missing_cols)}. "
            f"Found columns: {list(df_ref.columns)}"
        )

    if "name" not in df_ref.columns:
        df_ref["name"] = ""
    if "description" not in df_ref.columns:
        df_ref["description"] = ""

    df_ref["id"] = df_ref["id"].astype(str)
    ref_ids = set(df_ref["id"])

    date_series = pd.to_datetime(df_ref["date_created"], errors="coerce")
    if date_series.isna().all():
        raise ValueError(
            f"Column 'date_created' in {drugs_before_cutoff_path} could not be parsed as dates."
        )

    cutoff_ts = date_series.max()
    cutoff_date = cutoff_ts.strftime("%Y-%m-%d")

    reference_meta = {}
    for _, row in df_ref.iterrows():
        drug_id = str(row["id"])
        raw_description = row["description"] if pd.notna(row["description"]) else ""

        # mask the reference CSV description using gene names extracted from XML
        target_gene_names = drug_info.get(drug_id, {}).get("target_gene_names", [])
        masked_description = mask_targets_in_text(raw_description, target_gene_names)

        reference_meta[drug_id] = {
            "name": row["name"] if pd.notna(row["name"]) else "",
            "description": raw_description,
            "description_masked": masked_description,
            "date_created": row["date_created"] if pd.notna(row["date_created"]) else "",
        }

    print(f"Reference drugs loaded: {len(ref_ids)}")
    print(f"Inferred cutoff date from reference CSV: {cutoff_date}")
    return ref_ids, cutoff_ts, reference_meta


def construct_triplets(drug_info, drugs_before_cutoff_path, seed=42):
    """Construct evaluation triplets."""
    random.seed(seed)

    ref_ids, cutoff_ts, reference_meta = load_reference_info(
        drugs_before_cutoff_path, drug_info
    )

    reference_drugs = {}
    for drug_id in ref_ids:
        if drug_id in drug_info:
            reference_drugs[drug_id] = {
                "target_ids": drug_info[drug_id]["target_ids"],
                "target_gene_names": drug_info[drug_id]["target_gene_names"],
                "name": reference_meta.get(drug_id, {}).get("name", ""),
                "description": reference_meta.get(drug_id, {}).get("description", ""),
                "description_masked": reference_meta.get(drug_id, {}).get("description_masked", ""),
                "created_date": reference_meta.get(drug_id, {}).get("date_created", ""),
            }

    anchor_drugs = {}
    for drug_id, info in drug_info.items():
        created_date = info.get("created_date", "")
        created_ts = pd.to_datetime(created_date, errors="coerce")

        if (
            drug_id not in ref_ids
            and pd.notna(created_ts)
            and created_ts > cutoff_ts
        ):
            anchor_drugs[drug_id] = info

    print(f"Reference drugs found in XML: {len(reference_drugs)}")
    print(
        f"Anchor drugs (created_date > {cutoff_ts.strftime('%Y-%m-%d')} "
        f"and not in reference pool): {len(anchor_drugs)}"
    )

    if len(reference_drugs) == 0:
        raise ValueError(
            "No reference drugs from --drugs_before_cutoff_path were found in the XML file. "
            "Please make sure the reference CSV matches the DrugBank IDs in the XML."
        )

    target_to_drugs = defaultdict(list)
    for drug_id, info in reference_drugs.items():
        for target_id in info["target_ids"]:
            target_to_drugs[target_id].append(drug_id)

    print(f"Unique targets in reference drugs: {len(target_to_drugs)}")

    triplets = []

    for d1_id in tqdm(anchor_drugs.keys(), desc="Constructing triplets"):
        d1_info = anchor_drugs[d1_id]
        d1_targets = set(d1_info["target_ids"])

        if not d1_targets:
            continue

        for target_id in d1_targets:
            d2_candidates = target_to_drugs.get(target_id, [])
            if not d2_candidates:
                continue

            d2_id = random.choice(d2_candidates)
            d2_info = reference_drugs[d2_id]
            d2_targets = set(d2_info["target_ids"])

            all_excluded_targets = d1_targets | d2_targets

            d3_candidates = []
            for d3_id, d3_info in reference_drugs.items():
                if d3_id == d2_id:
                    continue
                d3_targets = set(d3_info["target_ids"])
                if not (d3_targets & all_excluded_targets):
                    d3_candidates.append(d3_id)

            if not d3_candidates:
                continue

            d3_id = random.choice(d3_candidates)
            d3_info = reference_drugs[d3_id]

            triplets.append(
                {
                    "drug1": d1_id,
                    "drug1_name": d1_info["name"],
                    "drug1_description": d1_info["description"],
                    "drug1_description_masked": d1_info["description_masked"],
                    "drug1_targets": ",".join(sorted(d1_targets)),
                    "drug_related": d2_id,
                    "drug_related_name": d2_info["name"],
                    "drug_related_description": d2_info["description"],
                    "drug_related_description_masked": d2_info["description_masked"],
                    "drug_related_targets": ",".join(sorted(d2_targets)),
                    "drug_unrelated": d3_id,
                    "drug_unrelated_name": d3_info["name"],
                    "drug_unrelated_description": d3_info["description"],
                    "drug_unrelated_description_masked": d3_info["description_masked"],
                    "drug_unrelated_targets": ",".join(sorted(d3_info["target_ids"])),
                    "shared_target": target_id,
                }
            )

    print(f"\nTotal triplets constructed: {len(triplets)}")
    return triplets


def parse_args():
    parser = argparse.ArgumentParser(
        description="Construct target-based drug similarity evaluation dataset."
    )
    parser.add_argument(
        "--xml_path",
        type=str,
        required=True,
        help="Path to the anchor-set DrugBank XML file.",
    )
    parser.add_argument(
        "--drugs_before_cutoff_path",
        type=str,
        default="data/full_data_ref.csv",
        help="Optional path to the reference CSV file. Default: data/full_data_ref.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="data/drug_similarity_eval.csv",
        help="Optional output CSV path. Default: data/drug_similarity_eval.csv",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed. Default: 42",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    drug_info = parse_drugs_from_xml(args.xml_path)
    triplets = construct_triplets(
        drug_info,
        args.drugs_before_cutoff_path,
        seed=args.seed,
    )

    output_dir = os.path.dirname(args.output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    df = pd.DataFrame(triplets)
    # df = df.dropna(subset=[
    #     "drug1_description_masked",
    #     "drug_related_description_masked",
    #     "drug_unrelated_description_masked",
    # ]).reset_index()
    df.to_csv(args.output_path, index=False)

    print(f"\nDataset saved to: {args.output_path}")
    print(f"Dataset shape: {df.shape}")
    print("\nFirst few rows:")
    print(df.head())


if __name__ == "__main__":
    main()