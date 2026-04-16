#!/usr/bin/env python3
"""
Evaluate robustness under description perturbations for all models:
BERT-base, BioBERT, PubMedBERT, SapBERT, Llama-3.1-8B, Drugspace-mntp, DrugSpace-mntp-contrastive

Changes vs original:
- Use argparse for input/output paths and eval params
- Rename train/test -> ref/anc
- Build anc_after_cutoff from input_ref + input_anc, following atc_retrieval_eval.py logic
"""

import argparse
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from typing import List, Dict, Optional
from abc import ABC, abstractmethod
import random
import re
from llm2vec import LLM2Vec


# ============================================================
# Abstract encoder interface
# ============================================================
class TextEncoder(ABC):
    """Abstract base class for any text encoder."""

    @abstractmethod
    def encode(
        self,
        texts: List[str],
        batch_size: int = 32,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        raise NotImplementedError


# ============================================================
# Perturbation functions
# ============================================================
def split_into_sentences(text: str) -> List[str]:
    sentences = re.split(r"[.!?]+\s+", text.strip())
    return [s.strip() for s in sentences if s.strip()]


def sentence_dropout(text: str, dropout_rate: float = 0.3, seed: Optional[int] = None) -> str:
    if seed is not None:
        random.seed(seed)

    sentences = split_into_sentences(text)
    if len(sentences) == 0:
        return text

    num_to_keep = max(1, int(len(sentences) * (1 - dropout_rate)))
    kept = random.sample(sentences, num_to_keep)
    kept = [s for s in sentences if s in kept]  # preserve original order
    return ". ".join(kept) + "." if kept else text


def sentence_shuffle(text: str, seed: Optional[int] = None) -> str:
    if seed is not None:
        random.seed(seed)

    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return text

    shuffled = sentences.copy()
    random.shuffle(shuffled)
    return ". ".join(shuffled) + "." if shuffled else text


def span_token_drop(
    text: str,
    drop_rate: float = 0.15,
    max_span_length: int = 3,
    seed: Optional[int] = None,
) -> str:
    if seed is not None:
        random.seed(seed)

    tokens = text.split()
    if len(tokens) == 0:
        return text

    num_to_drop = max(0, int(len(tokens) * drop_rate))
    if num_to_drop == 0:
        return text

    dropped_indices = set()
    remaining = num_to_drop

    while remaining > 0 and len(dropped_indices) < len(tokens):
        span_length = min(random.randint(1, max_span_length), remaining)
        start_idx = random.randint(0, len(tokens) - 1)

        for offset in range(len(tokens)):
            idx = (start_idx + offset) % len(tokens)
            if idx not in dropped_indices:
                for i in range(idx, min(idx + span_length, len(tokens))):
                    if i not in dropped_indices and remaining > 0:
                        dropped_indices.add(i)
                        remaining -= 1
                break

        if remaining == 0:
            break

    kept_tokens = [tokens[i] for i in range(len(tokens)) if i not in dropped_indices]
    return " ".join(kept_tokens) if kept_tokens else text


# ============================================================
# Self-similarity computation
# ============================================================
def cosine_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-8)
    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-8)
    return float(np.dot(emb1_norm, emb2_norm))


def compute_self_similarity(original_embs: np.ndarray, perturbed_embs: np.ndarray) -> Dict[str, float]:
    N = original_embs.shape[0]
    assert perturbed_embs.shape[0] == N

    sims = []
    for i in range(N):
        sims.append(cosine_similarity(original_embs[i], perturbed_embs[i]))
    sims = np.array(sims)

    return {
        "mean": float(np.mean(sims)),
        "std": float(np.std(sims)),
        "min": float(np.min(sims)),
        "max": float(np.max(sims)),
        "median": float(np.median(sims)),
    }


# ============================================================
# Data loading: build ref/anc like atc_retrieval_eval.py
# ============================================================
def build_anchor_reference_sets(ref_path: str, anc_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Follow atc_retrieval_eval.py logic:
    - read ref + anc
    - id as str
    - date_created to datetime
    - cutoff_date = ref.date_created.max()
    - anc_after_cutoff = anc with:
        id not in ref
        date_created notna
        date_created > cutoff_date
    """
    df_ref = pd.read_csv(ref_path)
    df_anc = pd.read_csv(anc_path)

    if "id" not in df_ref.columns or "id" not in df_anc.columns:
        raise ValueError("Both ref and anc CSV must contain column: 'id'")
    if "date_created" not in df_ref.columns or "date_created" not in df_anc.columns:
        raise ValueError("Both ref and anc CSV must contain column: 'date_created'")

    df_ref["id"] = df_ref["id"].astype(str)
    df_anc["id"] = df_anc["id"].astype(str)

    df_ref["date_created"] = pd.to_datetime(df_ref["date_created"], errors="coerce")
    df_anc["date_created"] = pd.to_datetime(df_anc["date_created"], errors="coerce")

    cutoff_date = df_ref["date_created"].max()
    ref_ids = set(df_ref["id"])

    anc_after_cutoff = df_anc[
        (~df_anc["id"].isin(ref_ids))
        & (df_anc["date_created"].notna())
        & (df_anc["date_created"] > cutoff_date)
    ].copy()

    reference_df = df_ref.copy()
    return reference_df, anc_after_cutoff


def filter_text_and_dedup(
    df: pd.DataFrame,
    text_field: str = "description",
    min_text_length: int = 10,
) -> pd.DataFrame:
    """
    Keep behavior consistent with your original robustness script:
    - dropna(text_field)
    - keep len(text) >= min_text_length
    - drop duplicate ids (keep first)
    """
    if text_field not in df.columns:
        raise ValueError(f"Missing text field column: '{text_field}'")

    out = df.dropna(subset=[text_field]).copy()
    out = out[out[text_field].astype(str).str.len() >= min_text_length].copy()
    if "id" in out.columns:
        out = out.drop_duplicates(subset=["id"], keep="first").reset_index(drop=True)
    else:
        out = out.reset_index(drop=True)
    return out


# ============================================================
# Encoder classes
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
    """Generic HuggingFace encoder (BERT, BioBERT, PubMedBERT, SapBERT, Llama)."""

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

        # keep your original behavior (only resize when torch_dtype is None)
        if torch_dtype is None:
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


class LLM2VecEncoder(TextEncoder):
    """Wrapper for LLM2Vec models."""

    def __init__(
        self,
        l2v_model,
        device: Optional[str] = None,
        default_max_length: int = 512,
        normalize: bool = True,
    ):
        self.l2v = l2v_model
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.default_max_length = default_max_length
        self.normalize = normalize

    def encode(
        self,
        texts: List[str],
        batch_size: int = 16,
        max_length: Optional[int] = None,
    ) -> np.ndarray:
        _ = max_length or self.default_max_length  # keep signature parity; LLM2Vec.encode may ignore it
        all_embs = []

        self.l2v.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i : i + batch_size]
                emb = self.l2v.encode(batch_texts, show_progress_bar=False)

                if isinstance(emb, np.ndarray):
                    emb = torch.from_numpy(emb)

                if self.normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

                all_embs.append(emb.cpu())

        return torch.cat(all_embs, dim=0).numpy()


# ============================================================
# Neighborhood stability computation
# ============================================================
def compute_neighborhood_stability_cross(
    anc_original_embs: np.ndarray,
    anc_perturbed_embs: np.ndarray,
    ref_embs: np.ndarray,
    ks: List[int] = [5, 10, 20, 50],
) -> Dict[str, float]:
    N_anc = anc_original_embs.shape[0]
    assert anc_perturbed_embs.shape[0] == N_anc

    anc_orig_norm = anc_original_embs / (np.linalg.norm(anc_original_embs, axis=1, keepdims=True) + 1e-8)
    anc_pert_norm = anc_perturbed_embs / (np.linalg.norm(anc_perturbed_embs, axis=1, keepdims=True) + 1e-8)
    ref_norm = ref_embs / (np.linalg.norm(ref_embs, axis=1, keepdims=True) + 1e-8)

    max_k = max(ks)
    all_overlaps = {k: [] for k in ks}

    for i in range(N_anc):
        orig_sims = anc_orig_norm[i] @ ref_norm.T
        pert_sims = anc_pert_norm[i] @ ref_norm.T

        orig_top = np.argsort(-orig_sims)[:max_k]
        pert_top = np.argsort(-pert_sims)[:max_k]

        for k in ks:
            orig_topk = set(orig_top[:k])
            pert_topk = set(pert_top[:k])
            inter = len(orig_topk & pert_topk)
            uni = len(orig_topk | pert_topk)
            all_overlaps[k].append(inter / uni if uni > 0 else 0.0)

    results = {}
    for k in ks:
        overlaps = np.array(all_overlaps[k])
        results[f"overlap@{k}_mean"] = float(np.mean(overlaps))
        results[f"overlap@{k}_std"] = float(np.std(overlaps))
        results[f"overlap@{k}_min"] = float(np.min(overlaps))
        results[f"overlap@{k}_max"] = float(np.max(overlaps))
        results[f"overlap@{k}_median"] = float(np.median(overlaps))
    return results


# ============================================================
# Main evaluation function
# ============================================================
def evaluate_model_robustness(
    anc_df: pd.DataFrame,
    ref_df: pd.DataFrame,
    encoder: TextEncoder,
    model_name: str,
    text_field: str = "description",
    batch_size: int = 32,
    max_length: Optional[int] = None,
    k_neighbors: List[int] = [5, 10, 20, 50],
    dropout_rate: float = 0.3,
    span_drop_rate: float = 0.15,
    seed: int = 42,
) -> Dict[str, Dict]:
    print(f"\n{'='*80}")
    print(f"Evaluating: {model_name}")
    print(f"{'='*80}")

    anc_texts = anc_df[text_field].fillna("").tolist()
    ref_texts = ref_df[text_field].fillna("").tolist()

    print(f"Anchor drugs (after cutoff): {len(anc_texts)}")
    print(f"Reference drugs (candidate set): {len(ref_texts)}")

    print("Encoding original anchor descriptions...")
    anc_original_embs = encoder.encode(anc_texts, batch_size=batch_size, max_length=max_length)
    print(f"  Shape: {anc_original_embs.shape}")

    print("Encoding reference descriptions...")
    ref_embs = encoder.encode(ref_texts, batch_size=batch_size, max_length=max_length)
    print(f"  Shape: {ref_embs.shape}")

    results = {}
    perturbation_types = ["sentence_dropout", "sentence_shuffle", "span_drop"]

    for pert_type in perturbation_types:
        print(f"\n--- {pert_type} ---")

        perturbed_texts = []
        for i, text in enumerate(anc_texts):
            if pert_type == "sentence_dropout":
                pert_text = sentence_dropout(text, dropout_rate=dropout_rate, seed=seed + i)
            elif pert_type == "sentence_shuffle":
                pert_text = sentence_shuffle(text, seed=seed + i)
            elif pert_type == "span_drop":
                pert_text = span_token_drop(text, drop_rate=span_drop_rate, seed=seed + i)
            else:
                raise ValueError(f"Unknown perturbation type: {pert_type}")
            perturbed_texts.append(pert_text)

        print("Encoding perturbed anchor descriptions...")
        anc_perturbed_embs = encoder.encode(perturbed_texts, batch_size=batch_size, max_length=max_length)
        print(f"  Shape: {anc_perturbed_embs.shape}")

        print("Computing self-similarity...")
        self_sim = compute_self_similarity(anc_original_embs, anc_perturbed_embs)

        print("Computing neighborhood stability...")
        neighbor_stab = compute_neighborhood_stability_cross(
            anc_original_embs, anc_perturbed_embs, ref_embs, ks=k_neighbors
        )

        results[pert_type] = {
            "self_similarity": self_sim,
            "neighborhood_stability": neighbor_stab,
        }

        print(f"  Self-similarity (mean): {self_sim['mean']:.4f} ± {self_sim['std']:.4f}")
        for k in k_neighbors:
            print(
                f"  Neighborhood overlap@{k} (mean): "
                f"{neighbor_stab[f'overlap@{k}_mean']:.4f} ± {neighbor_stab[f'overlap@{k}_std']:.4f}"
            )

    return results


# ============================================================
# Model configurations (unchanged)
# ============================================================
MODEL_CONFIGS = [
    {
        "name": "BERT-base",
        "type": "hf",
        "model_name": "bert-base-uncased",
        "pooling": "cls",
        "max_length": 128,
        "batch_size": 16,
        "torch_dtype": None,
    },
    {
        "name": "BioBERT",
        "type": "hf",
        "model_name": "dmis-lab/biobert-base-cased-v1.2",
        "pooling": "cls",
        "max_length": 128,
        "batch_size": 16,
        "torch_dtype": None,
    },
    {
        "name": "PubMedBERT",
        "type": "hf",
        "model_name": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
        "pooling": "cls",
        "max_length": 128,
        "batch_size": 16,
        "torch_dtype": None,
    },
    {
        "name": "SapBERT",
        "type": "hf",
        "model_name": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
        "pooling": "cls",
        "max_length": 128,
        "batch_size": 16,
        "torch_dtype": None,
    },
    {
        "name": "Llama-3.1-8B-Instruct",
        "type": "hf",
        "model_name": "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "pooling": "mean",
        "max_length": 256,
        "batch_size": 8,
        "torch_dtype": torch.bfloat16,
    },
    {
        "name": "DrugSpace",
        "type": "llm2vec",
        # "base_model_path": "/gpfs/radev/project/xu_hua/zc347/Drug_Embedder/llm2vec/output/mntp-pubtator/Meta-Llama-3.1-8B-Instruct-Pubtator7m-ver2-entitymask-surface-emlm0_2_full_model",
        # "peft_model_path": "/gpfs/radev/project/xu_hua/zc347/Drug_Embedder/llm2vec/output/mntp-pubtator-sft/Meta-Llama-3.1-8B-Instruct-Pubtator7m-sft-ver3-2020/DrugFinetuneData_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-5_s-42_w-300_lr-0.0001_lora_r-16/checkpoint-1330",
        "base_model_path": "cczzzyyy/DrugSpace-mntp-8B",
        "peft_model_path": "cczzzyyy/DrugSpace-full-lora-eval",
        "max_length": 512,
        "batch_size": 8,
    },
]


# ============================================================
# Argparse
# ============================================================
def _parse_int_list(csv: str) -> List[int]:
    parts = [p.strip() for p in csv.split(",") if p.strip()]
    return [int(p) for p in parts]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Robustness evaluation under description perturbations.")
    parser.add_argument(
        "--input_ref",
        type=str,
        default="data/full_data_ref.csv",
        help="Reference CSV (candidate set). Must include columns: id, date_created, description.",
    )
    parser.add_argument(
        "--input_anc",
        type=str,
        default="data/full_data_anc.csv",
        help="Anchor CSV (raw). Anchor-after-cutoff will be derived from this using ref cutoff. Must include columns: id, date_created, description.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/robustness_evaluation/results.csv",
        help="Output CSV path.",
    )

    parser.add_argument("--text_field", type=str, default="description")
    parser.add_argument("--min_text_length", type=int, default=10)

    parser.add_argument("--k_neighbors", type=_parse_int_list, default="5,10,20,50")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--span_drop_rate", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


# ============================================================
# Main execution
# ============================================================
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Build ref/anc sets like atc_retrieval_eval.py
    print(f"\nLoading ref from: {args.input_ref}")
    print(f"Loading anc (raw) from: {args.input_anc}")
    ref_df_raw, anc_df_raw_after_cutoff = build_anchor_reference_sets(args.input_ref, args.input_anc)

    print(f"Reference raw: {len(ref_df_raw)} rows")
    print(f"Anchor after cutoff raw: {len(anc_df_raw_after_cutoff)} rows")

    # Filter text + dedup (keeps original robustness behavior)
    ref_df = filter_text_and_dedup(ref_df_raw, text_field=args.text_field, min_text_length=args.min_text_length)
    anc_df = filter_text_and_dedup(
        anc_df_raw_after_cutoff, text_field=args.text_field, min_text_length=args.min_text_length
    )

    print(f"Reference filtered: {len(ref_df)} rows")
    print(f"Anchor after cutoff filtered: {len(anc_df)} rows")

    # Store all results
    all_results = []

    # Evaluate each model
    for config in MODEL_CONFIGS:
        model_name = config["name"]
        print(f"\n\n{'#'*80}")
        print(f"# Processing: {model_name}")
        print(f"{'#'*80}")

        try:
            if config["type"] == "hf":
                encoder = HFEncoder(
                    model_name=config["model_name"],
                    pooling=config["pooling"],
                    default_max_length=config["max_length"],
                    device=device,
                    torch_dtype=config.get("torch_dtype"),
                )
                batch_size = config["batch_size"]
                max_length = config["max_length"]

            elif config["type"] == "llm2vec":
                l2v_model = LLM2Vec.from_pretrained(
                    base_model_name_or_path=config["base_model_path"],
                    peft_model_name_or_path=config.get("peft_model_path"),
                    enable_bidirectional=True,
                    device_map="cuda" if torch.cuda.is_available() else "cpu",
                    torch_dtype=torch.bfloat16,
                )
                l2v_model.eval()

                encoder = LLM2VecEncoder(
                    l2v_model=l2v_model,
                    device=str(device),
                    default_max_length=config["max_length"],
                    normalize=True,
                )
                batch_size = config["batch_size"]
                max_length = config["max_length"]

            else:
                raise ValueError(f"Unknown model type: {config['type']}")

            results = evaluate_model_robustness(
                anc_df=anc_df,
                ref_df=ref_df,
                encoder=encoder,
                model_name=model_name,
                text_field=args.text_field,
                batch_size=batch_size,
                max_length=max_length,
                k_neighbors=args.k_neighbors,
                dropout_rate=args.dropout_rate,
                span_drop_rate=args.span_drop_rate,
                seed=args.seed,
            )

            for pert_type, metrics in results.items():
                row = {
                    "Model": model_name,
                    "Perturbation": pert_type,
                    "Self-Similarity_Mean": metrics["self_similarity"]["mean"],
                    "Self-Similarity_Std": metrics["self_similarity"]["std"],
                    "Self-Similarity_Min": metrics["self_similarity"]["min"],
                    "Self-Similarity_Max": metrics["self_similarity"]["max"],
                    "Self-Similarity_Median": metrics["self_similarity"]["median"],
                }
                for k in args.k_neighbors:
                    row[f"Neighborhood_Overlap@{k}_Mean"] = metrics["neighborhood_stability"][f"overlap@{k}_mean"]
                    row[f"Neighborhood_Overlap@{k}_Std"] = metrics["neighborhood_stability"][f"overlap@{k}_std"]
                    row[f"Neighborhood_Overlap@{k}_Min"] = metrics["neighborhood_stability"][f"overlap@{k}_min"]
                    row[f"Neighborhood_Overlap@{k}_Max"] = metrics["neighborhood_stability"][f"overlap@{k}_max"]
                    row[f"Neighborhood_Overlap@{k}_Median"] = metrics["neighborhood_stability"][f"overlap@{k}_median"]
                all_results.append(row)

            # cleanup
            del encoder
            if config["type"] == "llm2vec":
                del l2v_model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        except Exception as e:
            print(f"\nERROR evaluating {model_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    
    results_df = pd.DataFrame(all_results)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    print(f"\n\n{'='*80}")
    print(f"Results saved to: {args.output}")
    print(f"{'='*80}\n")

    # Summary table (same style as original)
    print("\nSummary Table:")
    print("=" * 120)

    for k in args.k_neighbors:
        print(f"\nNeighborhood Overlap@{k}:")
        pivot_df = results_df.pivot_table(
            index="Model",
            columns="Perturbation",
            values=[f"Neighborhood_Overlap@{k}_Mean"],
            aggfunc="first",
        )
        print(pivot_df)

    print("\nSelf-Similarity:")
    pivot_df = results_df.pivot_table(
        index="Model",
        columns="Perturbation",
        values=["Self-Similarity_Mean"],
        aggfunc="first",
    )
    print(pivot_df)
    print("=" * 120)


if __name__ == "__main__":
    main()