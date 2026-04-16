#!/usr/bin/env python3
"""
Evaluate target-based drug similarity with multiple baseline models.

Baseline models:
1. BERT-base
2. BioBERT
3. PubMedBERT
4. SapBERT
5. Llama-3.1-8B (pure, no MNTP, no SFT)
6. Llama-3.1-8B + MNTP + SFT (full model)
"""

import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from tqdm import tqdm
import torch
from llm2vec import LLM2Vec
import os
from sentence_transformers import SentenceTransformer

# Import the evaluation function
# from evaluate_target_similarity import evaluate_target_similarity

def load_bert_base(device="cuda"):
    """Load BERT-base model."""
    print("Loading BERT-base...")
    model = SentenceTransformer('bert-base-uncased', device=device)
    return model

def load_biobert(device="cuda"):
    """Load BioBERT model."""
    print("Loading BioBERT...")
    model = SentenceTransformer('dmis-lab/biobert-base-cased-v1.2', device=device)
    return model

def load_pubmedbert(device="cuda"):
    """Load PubMedBERT model."""
    print("Loading PubMedBERT...")
    model = SentenceTransformer('microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext', device=device)
    return model

def load_sapbert(device="cuda"):
    """Load SapBERT model."""
    print("Loading SapBERT...")
    model = SentenceTransformer('cambridgeltl/SapBERT-from-PubMedBERT-fulltext', device=device)
    return model

def load_llama_pure(device="cuda"):
    """Load pure Llama-3.1-8B (no MNTP, no SFT)."""
    print("Loading Llama-3.1-8B (pure)...")
    # Load base Llama model using LLM2Vec
    # Use HuggingFace model identifier for pure Llama
    base_model_path = "meta-llama/Meta-Llama-3.1-8B-Instruct"
    try:
        l2v = LLM2Vec.from_pretrained(
            base_model_name_or_path=base_model_path,
            enable_bidirectional=False,  # Pure Llama doesn't have bidirectional attention
            device_map=device if torch.cuda.is_available() else "cpu",
            torch_dtype=torch.bfloat16,
        )
        l2v.eval()
        return l2v
    except Exception as e:
        print(f"Warning: Could not load pure Llama model: {e}")
        print("Trying alternative: loading with bidirectional=True...")
        try:
            l2v = LLM2Vec.from_pretrained(
                base_model_name_or_path=base_model_path,
                enable_bidirectional=True,
                device_map=device if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
            )
            l2v.eval()
            return l2v
        except Exception as e2:
            print(f"Error loading pure Llama: {e2}")
            return None

def load_llama_mntp_sft(device="cuda"):
    """Load Llama-3.1-8B + MNTP + SFT (full model)."""
    print("Loading Llama-3.1-8B + MNTP + SFT...")
    # base_model_path = "/gpfs/radev/project/xu_hua/zc347/Drug_Embedder/llm2vec/output/mntp-pubtator/Meta-Llama-3.1-8B-Instruct-Pubtator7m-ver2-entitymask-surface-emlm0_2_full_model"
    # peft_model_path = "/gpfs/radev/project/xu_hua/zc347/Drug_Embedder/llm2vec/output/mntp-pubtator-sft/Meta-Llama-3.1-8B-Instruct-Pubtator7m-sft-ver3-2020/DrugFinetuneData_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-3_s-42_w-300_lr-0.0001_lora_r-16/checkpoint-480"
    # peft_model_path = "/gpfs/radev/project/xu_hua/zc347/Drug_Embedder/llm2vec/output/mntp-pubtator-sft/Meta-Llama-3.1-8B-Instruct-Pubtator7m-sft-ver3-2020/DrugFinetuneData_train_m-Meta-Llama-3.1-8B-Instruct_p-mean_b-64_l-512_bidirectional-True_e-5_s-42_w-300_lr-0.0001_lora_r-16/checkpoint-1330"
    base_model_path = "cczzzyyy/DrugSpace-mntp-8B"
    peft_model_path = "cczzzyyy/DrugSpace-full-lora-eval"
    l2v = LLM2Vec.from_pretrained(
        base_model_name_or_path=base_model_path,
        peft_model_name_or_path=peft_model_path,
        enable_bidirectional=True,
        device_map=device if torch.cuda.is_available() else "cpu",
        torch_dtype=torch.bfloat16,
    )
    l2v.eval()
    return l2v

def create_embedder_function(model, model_type):
    """
    Create a unified embedding function for different model types.
    
    Args:
        model: The loaded model
        model_type: Type of model ('sentence_transformer', 'llm2vec', or 'bioconceptvec')
    
    Returns:
        Function that takes list of texts and returns embeddings
    """
    if model_type == 'sentence_transformer':
        def embed_batch(texts):
            return model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        return embed_batch
    elif model_type == 'llm2vec':
        def embed_batch(texts):
            return model.encode(texts, show_progress_bar=False)
        return embed_batch
    elif model_type == 'bioconceptvec':
        import re
        def embed_batch(texts):
            embeddings = []
            for text in texts:
                tokens = re.findall(r'\b[\w-]+\b', text.lower())
                vecs = [model[t] for t in tokens if t in model]
                embeddings.append(np.mean(vecs, axis=0) if vecs else np.zeros(model.vector_size))
            return np.array(embeddings)
        return embed_batch
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def evaluate_target_similarity(
    dataset_path,
    embed_batch,
    output_dir=None,
    batch_size=32
):
    """
    Evaluate target-based drug similarity on triplet dataset.
    
    Args:
        dataset_path: Path to CSV file with triplets
        embed_batch: Function that takes list of texts and returns embeddings
        output_dir: Directory to save results
        batch_size: Batch size for embedding generation
    
    Returns:
        dict with evaluation metrics
    """
    # 1. Load dataset
    print("Loading dataset...")
    df = pd.read_csv(dataset_path)
    print(f"Total triplets: {len(df)}")
    
    # Filter out rows with empty descriptions
    initial_count = len(df)
    df = df[
        (df['drug1_description_masked'].notna()) & 
        (df['drug1_description_masked'] != '') &
        (df['drug_related_description_masked'].notna()) & 
        (df['drug_related_description_masked'] != '') &
        (df['drug_unrelated_description_masked'].notna()) & 
        (df['drug_unrelated_description_masked'] != '')
    ].copy()
    print(f"Triplets with valid descriptions: {len(df)} (dropped {initial_count - len(df)})")
    
    if len(df) == 0:
        raise ValueError("No valid triplets found!")
    
    # 2. Collect all unique descriptions
    print("Collecting descriptions...")
    all_descriptions = []
    desc_to_idx = {}
    
    for idx, row in df.iterrows():
        d1_desc = str(row['drug1_description_masked']).strip()
        d2_desc = str(row['drug_related_description_masked']).strip()
        d3_desc = str(row['drug_unrelated_description_masked']).strip()
        
        for desc, key in [(d1_desc, f"d1_{idx}"), (d2_desc, f"d2_{idx}"), (d3_desc, f"d3_{idx}")]:
            if desc and desc not in desc_to_idx:
                desc_to_idx[desc] = len(all_descriptions)
                all_descriptions.append(desc)
    
    print(f"Unique descriptions: {len(all_descriptions)}")
    
    # 3. Generate embeddings in batches
    print("Generating embeddings...")
    all_embeddings = []
    for i in tqdm(range(0, len(all_descriptions), batch_size), desc="Embedding batches"):
        batch = all_descriptions[i:i+batch_size]
        batch_embeddings = embed_batch(batch)
        if isinstance(batch_embeddings, torch.Tensor):
            batch_embeddings = batch_embeddings.cpu().numpy()
        all_embeddings.append(batch_embeddings)
    
    # Stack all embeddings
    embeddings_matrix = np.vstack(all_embeddings)
    print(f"Embeddings shape: {embeddings_matrix.shape}")
    
    # Create description -> embedding mapping
    desc_embeddings = {desc: embeddings_matrix[desc_to_idx[desc]] for desc in desc_to_idx}
    
    # 4. Calculate similarities for each triplet
    print("Calculating similarities...")
    sim_rel_list = []
    sim_unrel_list = []
    triplet_results = []
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing triplets"):
        d1_desc = str(row['drug1_description_masked']).strip()
        d2_desc = str(row['drug_related_description_masked']).strip()
        d3_desc = str(row['drug_unrelated_description_masked']).strip()
        
        # Get embeddings
        emb_d1 = desc_embeddings[d1_desc]
        emb_d2 = desc_embeddings[d2_desc]
        emb_d3 = desc_embeddings[d3_desc]
        
        # Normalize embeddings for cosine similarity
        emb_d1_norm = emb_d1 / (np.linalg.norm(emb_d1) + 1e-12)
        emb_d2_norm = emb_d2 / (np.linalg.norm(emb_d2) + 1e-12)
        emb_d3_norm = emb_d3 / (np.linalg.norm(emb_d3) + 1e-12)
        
        # Calculate cosine similarities
        sim_rel = np.dot(emb_d1_norm, emb_d2_norm)
        sim_unrel = np.dot(emb_d1_norm, emb_d3_norm)
        
        sim_rel_list.append(sim_rel)
        sim_unrel_list.append(sim_unrel)
        
        triplet_results.append({
            'drug1': row['drug1'],
            'drug1_name': row['drug1_name'],
            'drug_related': row['drug_related'],
            'drug_related_name': row['drug_related_name'],
            'drug_unrelated': row['drug_unrelated'],
            'drug_unrelated_name': row['drug_unrelated_name'],
            'shared_target': row['shared_target'],
            'sim_rel': sim_rel,
            'sim_unrel': sim_unrel,
            'diff_raw': sim_rel - sim_unrel
        })
    
    sim_rel_array = np.array(sim_rel_list)
    sim_unrel_array = np.array(sim_unrel_list)
    
    # 5. Standardization (bioconceptvec method)
    print("Standardizing similarities...")
    # Combine all similarities
    all_sims = np.concatenate([sim_rel_array, sim_unrel_array])
    
    # Z-score standardization
    zscaler = StandardScaler()
    all_sims_z = zscaler.fit_transform(all_sims.reshape(-1, 1)).flatten()
    
    # Min-Max normalization to [0, 1]
    mmscaler = MinMaxScaler()
    all_sims_normalized = mmscaler.fit_transform(all_sims_z.reshape(-1, 1)).flatten()
    
    # Split back
    sim_rel_normalized = all_sims_normalized[:len(sim_rel_array)]
    sim_unrel_normalized = all_sims_normalized[len(sim_rel_array):]
    
    # Calculate normalized differences
    diff_normalized = sim_rel_normalized - sim_unrel_normalized
    
    # Add normalized values to results
    for i, result in enumerate(triplet_results):
        result['sim_rel_normalized'] = sim_rel_normalized[i]
        result['sim_unrel_normalized'] = sim_unrel_normalized[i]
        result['diff_normalized'] = diff_normalized[i]
    
    # 6. Calculate metrics
    print("Calculating metrics...")
    
    # Mean difference (normalized)
    mean_diff_normalized = np.mean(diff_normalized)
    std_diff_normalized = np.std(diff_normalized)
    
    # Mean difference (raw)
    mean_diff_raw = np.mean(sim_rel_array - sim_unrel_array)
    std_diff_raw = np.std(sim_rel_array - sim_unrel_array)
    
    # Accuracy: sim_rel > sim_unrel
    accuracy = np.mean(sim_rel_array > sim_unrel_array)
    
    # Accuracy (normalized)
    accuracy_normalized = np.mean(sim_rel_normalized > sim_unrel_normalized)
    
    # Statistics (convert numpy types to Python native types for JSON serialization)
    metrics = {
        'mean_diff_normalized': float(mean_diff_normalized),
        'std_diff_normalized': float(std_diff_normalized),
        'mean_diff_raw': float(mean_diff_raw),
        'std_diff_raw': float(std_diff_raw),
        'accuracy': float(accuracy),
        'accuracy_normalized': float(accuracy_normalized),
        'num_triplets': int(len(df)),
        'mean_sim_rel': float(np.mean(sim_rel_array)),
        'mean_sim_unrel': float(np.mean(sim_unrel_array)),
        'std_sim_rel': float(np.std(sim_rel_array)),
        'std_sim_unrel': float(np.std(sim_unrel_array)),
    }
    
    # 7. Save results
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Save metrics
        metrics_path = os.path.join(output_dir, 'target_eval_metrics.json')
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        print(f"\nMetrics saved to: {metrics_path}")
        
        # Save detailed results
        results_df = pd.DataFrame(triplet_results)
        results_path = os.path.join(output_dir, 'target_eval_detailed_results.csv')
        results_df.to_csv(results_path, index=False)
        print(f"Detailed results saved to: {results_path}")
        
        # Print summary
        print("\n" + "="*60)
        print("EVALUATION RESULTS")
        print("="*60)
        print(f"Number of triplets: {metrics['num_triplets']}")
        print(f"\nNormalized metrics (bioconceptvec method):")
        print(f"  Mean difference: {mean_diff_normalized:.4f} ± {std_diff_normalized:.4f}")
        print(f"  Accuracy: {accuracy_normalized:.4f} ({accuracy_normalized*100:.2f}%)")
        print(f"\nRaw metrics:")
        print(f"  Mean difference: {mean_diff_raw:.4f} ± {std_diff_raw:.4f}")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\nSimilarity statistics:")
        print(f"  Mean sim(d1, d2): {metrics['mean_sim_rel']:.4f} ± {metrics['std_sim_rel']:.4f}")
        print(f"  Mean sim(d1, d3): {metrics['mean_sim_unrel']:.4f} ± {metrics['std_sim_unrel']:.4f}")
        print("="*60)
    
    return metrics, triplet_results



def evaluate_all_models(dataset_path, output_dir, device="cuda"):
    """Evaluate all baseline models."""
    
    # Define all models to evaluate
    models_to_evaluate = {
        'bert-base': {
            'load_func': load_bert_base,
            'model_type': 'sentence_transformer',
            'display_name': 'BERT-base'
        },
        'biobert': {
            'load_func': load_biobert,
            'model_type': 'sentence_transformer',
            'display_name': 'BioBERT'
        },
        'pubmedbert': {
            'load_func': load_pubmedbert,
            'model_type': 'sentence_transformer',
            'display_name': 'PubMedBERT'
        },
        'sapbert': {
            'load_func': load_sapbert,
            'model_type': 'sentence_transformer',
            'display_name': 'SapBERT'
        },
        'llama-pure': {
            'load_func': load_llama_pure,
            'model_type': 'llm2vec',
            'display_name': 'Llama-3.1-8B-Instruct'
        },
        'drugspace': {
            'load_func': load_llama_mntp_sft,
            'model_type': 'llm2vec',
            'display_name': 'DrugSpace'
        }
    }
    
    all_results = {}
    
    for model_key, model_config in models_to_evaluate.items():
        print("\n" + "="*60)
        print(f"Evaluating: {model_config['display_name']}")
        print("="*60)
        
        try:
            # Load model
            model = model_config['load_func'](device)
            if model is None:
                print(f"Skipping {model_key} (failed to load)")
                continue
            
            # Create embedder function
            embed_batch = create_embedder_function(model, model_config['model_type'])
            
            # Create model-specific output directory
            model_output_dir = os.path.join(output_dir, model_key)
            
            # Run evaluation
            metrics, results = evaluate_target_similarity(
                dataset_path=dataset_path,
                embed_batch=embed_batch,
                output_dir=model_output_dir,
                batch_size=32
            )
            
            # Store results
            all_results[model_key] = {
                'display_name': model_config['display_name'],
                'metrics': metrics
            }
            
            # Clean up GPU memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            
        except Exception as e:
            print(f"Error evaluating {model_key}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Save comparison results
    comparison_path = os.path.join(output_dir, 'model_comparison.json')
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison results saved to: {comparison_path}")
    
    # Create comparison table
    create_comparison_table(all_results, output_dir)
    
    return all_results

def create_comparison_table(all_results, output_dir):
    """Create a comparison table of all models."""
    rows = []
    for model_key, result in all_results.items():
        metrics = result['metrics']
        rows.append({
            'Model': result['display_name'],
            'Mean Diff (Normalized)': f"{metrics['mean_diff_normalized']:.4f} ± {metrics['std_diff_normalized']:.4f}",
            'Accuracy': f"{metrics['accuracy_normalized']:.4f} ({metrics['accuracy_normalized']*100:.2f}%)",
            'Mean Diff (Raw)': f"{metrics['mean_diff_raw']:.4f} ± {metrics['std_diff_raw']:.4f}",
            'Num Triplets': metrics['num_triplets']
        })
    
    comparison_df = pd.DataFrame(rows)
    comparison_path = os.path.join(output_dir, 'model_comparison_table.csv')
    comparison_df.to_csv(comparison_path, index=False)
    print(f"Comparison table saved to: {comparison_path}")
    
    # Print table
    print("\n" + "="*80)
    print("MODEL COMPARISON")
    print("="*80)
    print(comparison_df.to_string(index=False))
    print("="*80)

def main():
    """Main evaluation function."""
    dataset_path = 'data/drug_similarity_eval.csv'
    # dataset_path = 'data/drug_similarity_eval_24.csv'
    output_dir = 'results/drug_similarity'
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Evaluate all models
    results = evaluate_all_models(dataset_path, output_dir, device)
    
    print("\n All evaluations completed!")
    return results

if __name__ == '__main__':
    main()

