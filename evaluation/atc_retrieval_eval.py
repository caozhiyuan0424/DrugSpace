#!/usr/bin/env python3

import argparse
from abc import ABC, abstractmethod
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from llm2vec import LLM2Vec
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModel, AutoTokenizer


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# 1. ATC preprocessing
# ============================================================
def to_atc_level1(x: str) -> str:
    if pd.isna(x):
        return x
    x = str(x).strip()
    return x[0] if len(x) > 0 else x


def prepare_anchor_reference_atc(
    reference_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    atc_col: str = "atc_code",
    min_count: int = 10,
    force_level1: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, int], Dict[int, str]]:
    reference = reference_df.copy()
    anchor = anchor_df.copy()

    if force_level1:
        reference[atc_col] = reference[atc_col].map(to_atc_level1)
        anchor[atc_col] = anchor[atc_col].map(to_atc_level1)

    reference = reference.dropna(subset=[atc_col]).reset_index(drop=True)
    counts = Counter(reference[atc_col].tolist())
    valid_codes = {c for c, cnt in counts.items() if cnt >= min_count}
    reference = reference[reference[atc_col].isin(valid_codes)].reset_index(drop=True)

    unique_codes = sorted(reference[atc_col].unique())
    code2id = {c: i for i, c in enumerate(unique_codes)}
    id2code = {i: c for c, i in code2id.items()}
    reference["label"] = reference[atc_col].map(code2id)

    anchor = anchor.dropna(subset=[atc_col]).reset_index(drop=True)
    anchor = anchor[anchor[atc_col].isin(valid_codes)].reset_index(drop=True)
    anchor["label"] = anchor[atc_col].map(code2id)

    return reference, anchor, code2id, id2code


# ============================================================
# 2. Encoder interface
# ============================================================
class TextEncoder(ABC):
    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError


# ============================================================
# 3. Encoders
# ============================================================
class SimpleTextDataset(Dataset):
    def __init__(self, texts: List[str]):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]


def _mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).type_as(last_hidden)
    summed = (last_hidden * mask).sum(dim=1)
    denom = mask.sum(dim=1).clamp(min=1e-6)
    return summed / denom


class HFEncoder(TextEncoder):
    def __init__(
        self,
        model_name: str,
        pooling: str = "cls",
        default_max_length: int = 128,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
    ):
        assert pooling in ("cls", "mean")
        self.model_name = model_name
        self.pooling = pooling
        self.default_max_length = default_max_length
        self.device = device or torch.device("cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
        if self.tokenizer.pad_token is None:
            if self.tokenizer.eos_token is not None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            else:
                self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        self.tokenizer.padding_side = "right"

        self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype)
        self.model.resize_token_embeddings(len(self.tokenizer))
        self.model.to(self.device)
        self.model.eval()

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        dataset = SimpleTextDataset(texts)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        max_len = max_length or self.default_max_length
        all_embs = []

        with torch.no_grad():
            for batch_texts in loader:
                enc = self.tokenizer(
                    list(batch_texts),
                    padding=True,
                    truncation=True,
                    max_length=max_len,
                    return_tensors="pt",
                )
                enc = {k: v.to(self.device) for k, v in enc.items()}
                outputs = self.model(**enc)
                last_hidden = outputs.last_hidden_state

                if self.pooling == "cls":
                    emb = last_hidden[:, 0, :]
                else:
                    emb = _mean_pool(last_hidden, enc["attention_mask"])

                all_embs.append(emb.float().cpu())

        return torch.cat(all_embs, dim=0).numpy()


class DrugSpaceEncoder(TextEncoder):
    def __init__(
        self,
        base_model_name: str,
        peft_model_name: str,
        device: Optional[torch.device] = None,
        torch_dtype: Optional[torch.dtype] = None,
        default_max_length: int = 512,
    ):
        self.base_model_name = base_model_name
        self.peft_model_name = peft_model_name
        self.device = device or torch.device("cpu")
        self.default_max_length = default_max_length

        self.model = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_name,
            peft_model_name_or_path=peft_model_name,
            enable_bidirectional=True,
            device_map="cuda" if torch.cuda.is_available() else "cpu",
            torch_dtype=torch_dtype,
        )

    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        embs = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
        )
        if isinstance(embs, torch.Tensor):
            embs = embs.float().cpu().numpy()
        return embs


# ============================================================
# 4. Retrieval metrics
# ============================================================
def compute_anchor_reference_similarity(
    anchor_embs: np.ndarray,
    reference_embs: np.ndarray,
    chunk_size: int = 512,
    device: torch.device = DEVICE,
) -> np.ndarray:
    anchor_t = torch.tensor(anchor_embs, dtype=torch.float32, device=device)
    reference_t = torch.tensor(reference_embs, dtype=torch.float32, device=device)

    anchor_t = torch.nn.functional.normalize(anchor_t, p=2, dim=1)
    reference_t = torch.nn.functional.normalize(reference_t, p=2, dim=1)

    n_anchor = anchor_t.shape[0]
    sims_all = []

    with torch.no_grad():
        for start in range(0, n_anchor, chunk_size):
            anchor_chunk = anchor_t[start : start + chunk_size]
            sim_chunk = anchor_chunk @ reference_t.T
            sims_all.append(sim_chunk.cpu())

    return torch.cat(sims_all, dim=0).numpy()


def anchor_reference_retrieval_metrics(
    sim_anchor_reference: np.ndarray,
    anchor_labels: np.ndarray,
    reference_labels: np.ndarray,
    ks: Tuple[int, ...] = (1, 5, 10),
) -> Dict[str, float]:
    n_anchor, n_reference = sim_anchor_reference.shape
    assert n_anchor == len(anchor_labels)
    assert n_reference == len(reference_labels)

    hits_at = {k: [] for k in ks}
    p_at = {k: [] for k in ks}
    reciprocal_ranks = []

    for i in range(n_anchor):
        sims = sim_anchor_reference[i]
        ranked = np.argsort(-sims)
        rel = (reference_labels[ranked] == anchor_labels[i]).astype(np.int32)

        for k in ks:
            topk = rel[:k]
            hits_at[k].append(1.0 if topk.any() else 0.0)
            p_at[k].append(float(topk.mean()))

        pos = np.where(rel == 1)[0]
        reciprocal_ranks.append(0.0 if len(pos) == 0 else 1.0 / float(pos[0] + 1))

    out: Dict[str, float] = {}
    for k in ks:
        out[f"Hits@{k}"] = float(np.mean(hits_at[k]))
        out[f"P@{k}"] = float(np.mean(p_at[k]))
    out["MRR"] = float(np.mean(reciprocal_ranks))
    return out


def run_atc_retrieval_anchor_reference(
    reference_df: pd.DataFrame,
    anchor_df: pd.DataFrame,
    encoder: TextEncoder,
    text_field: str = "description",
    atc_col: str = "atc_code",
    min_count: int = 10,
    force_level1: bool = True,
    batch_size_reference: int = 16,
    batch_size_anchor: int = 16,
    max_length: int = 128,
    ks: Tuple[int, ...] = (1, 5, 10),
    sim_chunk_size: int = 512,
) -> Dict[str, float]:
    reference, anchor, _, _ = prepare_anchor_reference_atc(
        reference_df=reference_df,
        anchor_df=anchor_df,
        atc_col=atc_col,
        min_count=min_count,
        force_level1=force_level1,
    )

    reference_texts = reference[text_field].fillna("").tolist()
    anchor_texts = anchor[text_field].fillna("").tolist()

    reference_labels = reference["label"].values
    anchor_labels = anchor["label"].values

    reference_embs = encoder.encode(
        reference_texts,
        batch_size=batch_size_reference,
        max_length=max_length,
    )
    anchor_embs = encoder.encode(
        anchor_texts,
        batch_size=batch_size_anchor,
        max_length=max_length,
    )

    sim_anchor_reference = compute_anchor_reference_similarity(
        anchor_embs=anchor_embs,
        reference_embs=reference_embs,
        chunk_size=sim_chunk_size,
        device=getattr(encoder, "device", DEVICE),
    )

    return anchor_reference_retrieval_metrics(
        sim_anchor_reference=sim_anchor_reference,
        anchor_labels=anchor_labels,
        reference_labels=reference_labels,
        ks=ks,
    )


# ============================================================
# 5. Data loading
# ============================================================
def build_anchor_reference_sets(
    ref_path: str,
    anc_path: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    df_ref = pd.read_csv(ref_path)
    df_anc = pd.read_csv(anc_path)

    df_ref["id"] = df_ref["id"].astype(str)
    df_anc["id"] = df_anc["id"].astype(str)

    df_ref["date_created"] = pd.to_datetime(df_ref["date_created"], errors="coerce")
    df_anc["date_created"] = pd.to_datetime(df_anc["date_created"], errors="coerce")

    cutoff_date = df_ref["date_created"].max()
    ref_ids = set(df_ref["id"])

    anchor_df = df_anc[
        (~df_anc["id"].isin(ref_ids))
        & (df_anc["date_created"].notna())
        & (df_anc["date_created"] > cutoff_date)
    ].copy()

    reference_df = df_ref.copy()
    return reference_df, anchor_df


MODELS = [
    {
        "name": "BERT-base",
        "type": "hf",
        "model_name": "bert-base-uncased",
        "pooling": "cls",
        "max_length": 128,
        "batch_reference": 16,
        "batch_anchor": 16,
        "torch_dtype": None,
    },
    {
        "name": "BioBERT",
        "type": "hf",
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "pooling": "cls",
        "max_length": 128,
        "batch_reference": 16,
        "batch_anchor": 16,
        "torch_dtype": None,
    },
    {
        "name": "PubMedBERT",
        "type": "hf",
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "pooling": "cls",
        "max_length": 128,
        "batch_reference": 16,
        "batch_anchor": 16,
        "torch_dtype": None,
    },
    {
        "name": "SapBERT",
        "type": "hf",
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "pooling": "cls",
        "max_length": 128,
        "batch_reference": 16,
        "batch_anchor": 16,
        "torch_dtype": None,
    },
    {
        "name": "Llama-3.1-8B",
        "type": "hf",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "pooling": "mean",
        "max_length": 256,
        "batch_reference": 4,
        "batch_anchor": 4,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
    },
    {
        "name": "DrugSpace",
        "type": "drugspace",
        "base_model_name": "cczzzyyy/DrugSpace-mntp-8B",
        "peft_model_name": "cczzzyyy/DrugSpace-full-lora-eval",
        "max_length": 512,
        "batch_reference": 4,
        "batch_anchor": 4,
        "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else None,
    },
]


def build_encoder(cfg: Dict) -> TextEncoder:
    if cfg["type"] == "hf":
        return HFEncoder(
            model_name=cfg["model_name"],
            pooling=cfg["pooling"],
            default_max_length=cfg["max_length"],
            device=DEVICE,
            torch_dtype=cfg["torch_dtype"],
        )
    if cfg["type"] == "drugspace":
        return DrugSpaceEncoder(
            base_model_name=cfg["base_model_name"],
            peft_model_name=cfg["peft_model_name"],
            device=DEVICE,
            torch_dtype=cfg["torch_dtype"],
            default_max_length=cfg["max_length"],
        )
    raise ValueError(f"Unknown model type: {cfg['type']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_ref",
        type=str,
        default="data/full_data_ref.csv",
    )
    parser.add_argument(
        "--input_anc",
        type=str,
        default="data/full_data_anc.csv",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/atc_retrieval",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "results.csv"

    reference_df, anchor_df = build_anchor_reference_sets(
        ref_path=args.input_ref,
        anc_path=args.input_anc,
    )
    # anchor_df = pd.read_csv("data/drugs_after_2024.csv")

    all_results = []
    for cfg in MODELS:
        encoder = build_encoder(cfg)
        metrics = run_atc_retrieval_anchor_reference(
            reference_df=reference_df,
            anchor_df=anchor_df,
            encoder=encoder,
            text_field="description",
            atc_col="atc_code",
            min_count=10,
            force_level1=True,
            batch_size_reference=cfg["batch_reference"],
            batch_size_anchor=cfg["batch_anchor"],
            max_length=cfg["max_length"],
            ks=(1, 5, 10),
            sim_chunk_size=512,
        )
        all_results.append({"Model": cfg["name"], **metrics})

    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()
