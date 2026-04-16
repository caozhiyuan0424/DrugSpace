#!/usr/bin/env python3
import re
import argparse
import xmltodict
import pandas as pd


def extract_atc_info(atc_codes):
    """
    Extract the ATC code and ATC name from the DrugBank XML atc-codes field.

    Args:
        atc_codes: Value of drug["atc-codes"]

    Returns:
        tuple: (atc_code, atc_name), or (None, None) if unavailable
    """
    if not atc_codes:
        return None, None

    codes = atc_codes.get("atc-code")
    if codes is None:
        return None, None
    if isinstance(codes, dict):
        codes = [codes]

    atc = codes[0]
    full_code = atc.get("@code")
    if not full_code:
        return None, None

    k = 1
    atc_code = full_code[:k]

    atc_name = None
    levels = atc.get("level", [])
    if isinstance(levels, dict):
        levels = [levels]

    for lvl in levels:
        if lvl.get("@code") == atc_code:
            atc_name = lvl.get("#text")
            break

    return atc_code, atc_name


def remove_references(text):
    if text is None:
        return None
    pattern = r"\[[A-Za-z0-9, ]+\]"
    return re.sub(pattern, "", text)


def parse_drugbank(xml_path):
    with open(xml_path, "r", encoding="utf-8") as file:
        xml_data = file.read()

    data_dict = xmltodict.parse(xml_data)
    drugs = data_dict["drugbank"]["drug"]

    rows = []

    for d in drugs:
        raw_ids = d.get("drugbank-id")
        if isinstance(raw_ids, list):
            ids = raw_ids
        elif raw_ids is None:
            ids = []
        else:
            ids = [raw_ids]

        primary_id = None
        for item in ids:
            if isinstance(item, dict) and item.get("@primary") == "true":
                primary_id = item.get("#text")
                break

        if primary_id is None and ids:
            first = ids[0]
            primary_id = first["#text"] if isinstance(first, dict) else first

        if primary_id is None:
            continue

        smiles = None
        calc = d.get("calculated-properties") or {}
        props = calc.get("property") or []
        if isinstance(props, dict):
            props = [props]

        for prop in props:
            if prop.get("kind") == "SMILES":
                smiles = prop.get("value")
                break

        atc_code, atc_name = extract_atc_info(d.get("atc-codes"))

        direct_parent = None
        classification = d.get("classification") or {}
        if classification:
            direct_parent = classification.get("direct-parent")

        rows.append(
            {
                "name": d.get("name"),
                "id": primary_id,
                "date_created": d.get("@created"),
                "date_updated": d.get("@updated"),
                "description": d.get("description"),
                "indication": d.get("indication"),
                "pharmacodynamics": d.get("pharmacodynamics"),
                "mechanism-of-action": d.get("mechanism-of-action"),
                "toxicity": d.get("toxicity"),
                "metabolism": d.get("metabolism"),
                "smiles": smiles,
                "atc_code": atc_code,
                "atc_name": atc_name,
                "direct_parent": direct_parent,
            }
        )

    df = pd.DataFrame(rows)

    text_cols = [
        "name",
        "description",
        "indication",
        "pharmacodynamics",
        "mechanism-of-action",
        "toxicity",
        "metabolism",
    ]

    for col in text_cols:
        if col in df.columns:
            df[col] = df[col].apply(remove_references)

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Parse DrugBank XML and extract selected fields into a CSV file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to input DrugBank XML file",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output CSV file",
    )

    args = parser.parse_args()

    df = parse_drugbank(args.input)

    print(f"Extracted {len(df)} records in total")
    print(f"Records with SMILES: {df['smiles'].notna().sum()}")
    print(f"Records with ATC code: {df['atc_code'].notna().sum()}")
    print(f"Records with direct_parent: {df['direct_parent'].notna().sum()}")
    print(df.columns.tolist())

    df.to_csv(args.output, index=False)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()