"""Shared 512-token, mean-pooling encoder for embedding and evaluation."""
import argparse
from pathlib import Path

import numpy as np

MAX_LENGTH = 512
SEPARATOR = "!@#$%^&*()"
MODELS = {
    "bert": "bert-base-uncased",
    "biobert": "dmis-lab/biobert-base-cased-v1.2",
    "pubmedbert": "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "sapbert": "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "biomistral": "BioMistral/BioMistral-7B",
    "embeddinggemma": "google/embeddinggemma-300m",
    "drugspace": "z-cao/DrugSpace-8B",
    "drugspace-eval": "z-cao/DrugSpace-eval-8B",
    "drugbank": "z-cao/DrugSpace-mntp-eval-8B",
}


def model_args(parser, default="drugspace-eval"):
    parser.add_argument("--model", choices=MODELS, default=default)
    parser.add_argument("--model-path", help="Override the base model with a local path or Hub ID")
    parser.add_argument("--adapter", help="Local trained LoRA directory (required for drugbank)")
    parser.add_argument("--batch-size", type=int, default=4)


def skip_untrained_drugbank(args):
    if args.model == "drugbank" and not args.adapter:
        print("SKIPPED DrugSpace (DrugBank): no --adapter supplied. "
              "Use --model drugspace-eval or a baseline, or provide your trained adapter.")
        return True
    return False


def normalize(values):
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Embeddings must be a finite matrix")
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    if (norms == 0).any():
        raise ValueError("Encountered a zero embedding")
    return values / norms


class Encoder:
    def __init__(self, model="drugspace", model_path=None, adapter=None, batch_size=4):
        import torch

        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        if model == "drugbank" and not adapter:
            raise ValueError("drugbank requires --adapter pointing to your trained LoRA")
        if adapter and model not in {"drugspace", "drugspace-eval", "drugbank"}:
            raise ValueError("Adapters are supported only for DrugSpace models")
        self.batch_size = batch_size
        self._cache = {}  # Reuse identical text vectors across batches and encode calls.
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        path = model_path or MODELS[model]
        self.kind = "gemma" if model == "embeddinggemma" else "hf"
        if model in {"biomistral", "drugspace", "drugspace-eval", "drugbank"}:
            from llm2vec import LLM2Vec

            self.kind = "llm2vec"
            is_drugspace = model != "biomistral"
            self.model = LLM2Vec.from_pretrained(
                path, peft_model_name_or_path=adapter,
                enable_bidirectional=is_drugspace, pooling_mode="mean",
                max_length=MAX_LENGTH, doc_max_length=MAX_LENGTH,
                skip_instruction=is_drugspace, device_map=self.device,
                torch_dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            self.model.eval()
        elif self.kind == "gemma":
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(path, device=self.device)
            self.model.max_seq_length = MAX_LENGTH
            if any(layer.self_attn.is_causal for layer in self.model[0].auto_model.layers):
                raise RuntimeError("EmbeddingGemma needs bidirectional attention; install transformers==4.57.1")
        else:
            from transformers import AutoModel, AutoTokenizer

            self.tokenizer = AutoTokenizer.from_pretrained(path)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.padding_side = "right"
            dtype = torch.bfloat16 if model == "llama" and self.device == "cuda" else torch.float32
            self.model = AutoModel.from_pretrained(path, torch_dtype=dtype).to(self.device).eval()

    def encode(self, texts):
        import torch

        texts = list(texts)
        if not texts or any(not isinstance(text, str) for text in texts):
            raise ValueError("Provide a nonempty sequence of strings")
        missing = list(dict.fromkeys(text for text in texts if text not in self._cache))
        with torch.inference_mode():
            for start in range(0, len(missing), self.batch_size):
                batch = missing[start:start + self.batch_size]
                if self.kind == "llm2vec":
                    tokens = self.model.tokenize([
                        self.model.prepare_for_tokenization(SEPARATOR + text) for text in batch
                    ])
                    tokens = {key: value.to(self.device) for key, value in tokens.items()}
                    # Empty descriptions pool their special token, never neighboring padding.
                    tokens["embed_mask"] = torch.where(tokens["embed_mask"].any(1, keepdim=True),
                                                       tokens["embed_mask"], tokens["attention_mask"])
                    values = self.model(tokens)
                elif self.kind == "gemma":
                    values = self.model.encode(batch, batch_size=len(batch), show_progress_bar=False,
                                               convert_to_numpy=True, normalize_embeddings=True)
                else:
                    tokens = self.tokenizer(batch, padding=True, truncation=True,
                                            max_length=MAX_LENGTH, return_tensors="pt").to(self.device)
                    hidden = self.model(**tokens).last_hidden_state
                    mask = tokens["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                    values = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1)
                if isinstance(values, torch.Tensor):
                    values = values.float().cpu().numpy()
                self._cache.update(zip(batch, normalize(values)))
        return np.stack([self._cache[text] for text in texts])


def main():
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="CSV with one text per row")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--text-column", default="description")
    model_args(parser, default="drugspace")
    args = parser.parse_args()
    data = pd.read_csv(args.input, keep_default_na=False)
    if data.empty or not data[args.text_column].str.strip().ne("").all():
        raise ValueError("Every input row must contain text")
    encoder = Encoder(args.model, args.model_path, args.adapter, args.batch_size)
    values = encoder.encode(data[args.text_column])
    args.output.mkdir(parents=True, exist_ok=True)
    np.save(args.output / "embeddings.npy", values)
    data.to_csv(args.output / "metadata.csv", index=False)
    print(f"Saved {values.shape} embeddings; metadata.csv has the same row order")


if __name__ == "__main__":
    main()
