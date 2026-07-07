import os
import json
import argparse
import string
import pickle
import numpy as np
from collections import defaultdict
from scipy.stats import pearsonr
from sklearn.metrics import cohen_kappa_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LBM predictions grouped by question type.")
    parser.add_argument(
        "--predictions", 
        type=str, 
        required=True, 
        help="Path to the predictions.jsonl file"
    )
    # ADDED: Separate arguments for Matrix and MC mappings
    parser.add_argument(
        "--matrix_mapping", 
        type=str, 
        required=True, 
        help="Path to datasets/results/evals/matrixQ_likert.pkl"
    )
    parser.add_argument(
        "--mc_mapping", 
        type=str, 
        required=True, 
        help="Path to datasets/results/evals/mcQ_ordinal.pkl"
    )
    parser.add_argument(
        "--history", 
        type=str, 
        default=None, 
        help="Optional: Path to Wave 1-3 history to compute Set 1 (Human vs Human baseline)"
    )
    parser.add_argument(
        "--metrics_path", 
        type=str, 
        default=None, 
        help="Optional: Path to save the evaluation metrics as a JSON file"
    )
    return parser.parse_args()

def normalize_answer(text):
    """
    Lowercases, strips leading/trailing whitespace, and removes punctuation.
    """
    if text is None:
        return ""
    text = str(text).lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return " ".join(text.split())

def determine_question_type(row):
    """
    Determines the question type based on the JSON or infers from ID.
    """
    if "question_type" in row:
        q_type = str(row["question_type"]).upper()
        if "MATRIX" in q_type:
            return "Matrix"
        if "MC" in q_type:
            return "MC_Ordinal"
        return q_type
    
    q_id = str(row.get("question_id", ""))
    if "_" in q_id and q_id.split("_")[-1].isdigit():
        return "Matrix"
    
    return "MC_Ordinal"

def main():
    args = parse_args()
    
    # 1. Validate all file paths
    if not os.path.exists(args.predictions):
        raise FileNotFoundError(f"Could not find prediction file: {args.predictions}")
    if not os.path.exists(args.matrix_mapping):
        raise FileNotFoundError(f"Could not find Matrix mapping file: {args.matrix_mapping}")
    if not os.path.exists(args.mc_mapping):
        raise FileNotFoundError(f"Could not find MC mapping file: {args.mc_mapping}")
        
    print(f"Loading predictions from {args.predictions}...")
    
    # 2. Load BOTH mapping dictionaries
    with open(args.matrix_mapping, 'rb') as f:
        matrix_dict = pickle.load(f)
        
    with open(args.mc_mapping, 'rb') as f:
        mc_dict = pickle.load(f)
        
    # Merge them into a single dictionary for seamless lookups
    mapping_dict = {**matrix_dict, **mc_dict}
    print(f"Successfully loaded {len(matrix_dict)} Matrix mappings and {len(mc_dict)} MC mappings.")
        
    # Optional: Load human history for Set 1 (Human vs Human ceiling)
    historical_answers = {}
    if args.history and os.path.exists(args.history):
        with open(args.history, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    historical_answers[(data["pid"], data["question_id"])] = data.get("selected_answer")

    # Data structures to hold exact match stats and numerical arrays for advanced stats
    metrics = defaultdict(lambda: {"total": 0, "exact_matches": 0, "normalized_matches": 0})
    
    # Store numerical values for correlation/deviation math
    num_data = {
        "Matrix": {"target": [], "pred": [], "history": []},
        "MC_Ordinal": {"target": [], "pred": [], "history": []}
    }
    
    unmappable_count = 0

    with open(args.predictions, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
                
            row = json.loads(line)
            pid = row.get("pid")
            q_id = str(row.get("question_id", ""))
            target_str = str(row.get("target_answer", ""))
            pred_str = str(row.get("predicted_answer", ""))
            
            q_type = determine_question_type(row)
            
            # --- 1. Basic Accuracy (Exact & Normalized) ---
            is_exact = (target_str == pred_str)
            is_norm = (normalize_answer(target_str) == normalize_answer(pred_str))
            
            for group in ["Overall", q_type]:
                metrics[group]["total"] += 1
                if is_exact: metrics[group]["exact_matches"] += 1
                if is_norm: metrics[group]["normalized_matches"] += 1

            # --- 2. Advanced Metrics Data Gathering ---
            if q_type in num_data:
                # Base question ID for mapping (e.g., QID287_0 -> QID287 if the mapping is stored by base QID)
                base_qid = q_id.split('_')[0] if '_' in q_id else q_id
                
                # Fetch the specific mapping for this question
                q_mapping = mapping_dict.get(q_id, mapping_dict.get(base_qid, {}))
                
                if q_mapping:
                    norm_target = normalize_answer(target_str)
                    norm_pred = normalize_answer(pred_str)
                    
                    # Convert to integers using the mapping
                    t_val = q_mapping.get(norm_target)
                    p_val = q_mapping.get(norm_pred)
                    
                    if t_val is not None and p_val is not None:
                        num_data[q_type]["target"].append(t_val)
                        num_data[q_type]["pred"].append(p_val)
                        
                        # Set 1 (Human vs Human) if history is provided
                        if args.history:
                            h_str = historical_answers.get((pid, q_id))
                            h_val = q_mapping.get(normalize_answer(h_str)) if h_str else None
                            if h_val is not None:
                                num_data[q_type]["history"].append(h_val)
                    else:
                        unmappable_count += 1
                else:
                    unmappable_count += 1

    # --- PRINT DASHBOARD ---
    print("\n" + "="*60)
    print(" LBM EVALUATION RESULTS BY QUESTION TYPE")
    print("="*60)
    
    if unmappable_count > 0:
        print(f"⚠️ Warning: {unmappable_count} predictions could not be mapped to numerical values (Check PKL keys or Hallucinations).")
    
    final_report = {}
    
    for group in ["Overall", "Matrix", "MC_Ordinal"]:
        if metrics[group]["total"] == 0:
            continue
            
        data = metrics[group]
        total = data["total"]
        exact_acc = (data["exact_matches"] / total) * 100
        norm_acc = (data["normalized_matches"] / total) * 100
        
        report_dict = {
            "total_questions": total,
            "exact_match_accuracy": round(exact_acc, 2),
            "normalized_accuracy": round(norm_acc, 2)
        }
        
        print(f"\n--- {group.upper()} ---")
        print(f"Total Evaluated  : {total}")
        print(f"Exact Match Acc  : {exact_acc:.2f}%")
        
        # Calculate Advanced Metrics if numerical data exists
        if group in num_data and len(num_data[group]["target"]) > 1:
            targets = np.array(num_data[group]["target"])
            preds = np.array(num_data[group]["pred"])
            
            # Mean Absolute Deviation (MAD)
            mad = np.mean(np.abs(targets - preds))
            report_dict["mean_absolute_deviation"] = round(mad, 4)
            print(f"Mean Abs Dev (MAD): {mad:.4f}")
            
            # Pearson Correlation (For Matrix)
            if group == "Matrix":
                # Ensure variance isn't 0 to avoid Pearson warnings
                if np.std(targets) > 0 and np.std(preds) > 0:
                    r, _ = pearsonr(targets, preds)
                    report_dict["pearson_r"] = round(r, 4)
                    print(f"Set 2 Pearson (r) : {r:.4f} (Model vs W4 Human)")
                    
                    # Set 1 Human Ceiling
                    if len(num_data[group]["history"]) == len(targets):
                        hist = np.array(num_data[group]["history"])
                        if np.std(hist) > 0:
                            r_human, _ = pearsonr(targets, hist)
                            print(f"Set 1 Pearson (r) : {r_human:.4f} (W1-3 Human vs W4 Human Ceiling)")

            # Cohen's Weighted Kappa (For Ordinal MC & Matrix)
            # Ensure more than 1 unique class is present to avoid ill-defined kappa
            if len(set(targets).union(set(preds))) > 1:
                kappa = cohen_kappa_score(targets, preds, weights='quadratic')
                report_dict["cohens_weighted_kappa"] = round(kappa, 4)
                print(f"Set 2 W. Kappa    : {kappa:.4f} (Model vs W4 Human)")
                
                # Set 1 Human Ceiling for Kappa
                if len(num_data[group]["history"]) == len(targets):
                    hist = np.array(num_data[group]["history"])
                    kappa_human = cohen_kappa_score(targets, hist, weights='quadratic')
                    print(f"Set 1 W. Kappa    : {kappa_human:.4f} (W1-3 Human vs W4 Human Ceiling)")
                
        final_report[group] = report_dict

    print("\n" + "="*60)

    if args.metrics_path:
        os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
        with open(args.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=4)
        print(f"Metrics saved to {args.metrics_path}")

if __name__ == "__main__":
    main()