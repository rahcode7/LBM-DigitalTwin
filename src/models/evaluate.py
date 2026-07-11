import os
import json
import argparse
import pickle
import random
import numpy as np
from collections import defaultdict
from sklearn.metrics import cohen_kappa_score

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate LBM predictions matched strictly against historical human support.")
    parser.add_argument("--predictions", type=str, required=True, help="Path to predictions.jsonl")
    parser.add_argument("--matrix_mapping", type=str, required=True, help="Path to matrixQ_likert.pkl")
    parser.add_argument("--mc_mapping", type=str, required=True, help="Path to mcQ_ordinal.pkl")
    parser.add_argument("--history", type=str, required=True, help="Path to Wave 1-3 history")
    parser.add_argument("--metrics_path", type=str, default=None, help="Optional: Path to save JSON report")
    return parser.parse_args()

def main():
    args = parse_args()
    random.seed(42)
    
    if not os.path.exists(args.predictions): raise FileNotFoundError("Predictions file not found.")
    if not os.path.exists(args.matrix_mapping): raise FileNotFoundError("Matrix mapping not found.")
    if not os.path.exists(args.mc_mapping): raise FileNotFoundError("MC mapping not found.")
    if not os.path.exists(args.history): raise FileNotFoundError("History file is required for subset intersection.")
        
    print(f"Loading files and calculating intersection support...")
    
    with open(args.matrix_mapping, 'rb') as f: matrix_dict = pickle.load(f)
    with open(args.mc_mapping, 'rb') as f: mc_dict = pickle.load(f)
    mapping_dict = {**matrix_dict, **mc_dict}
        
    historical_answers = {}
    historical_types = {}
    with open(args.history, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                pid = data.get("pid")
                q_id = data.get("question_id")
                if pid and q_id:
                    historical_answers[(pid, q_id)] = data.get("selected_answer")
                    # Store the exact question type from the history file
                    q_type_raw = str(data.get("question_type", "MC")).upper()
                    historical_types[q_id] = "Matrix" if "MATRIX" in q_type_raw else "MC"

    metrics = defaultdict(lambda: {"total": 0})
    
    num_data = {
        "Overall": {"target": [], "pred": [], "history": [], "random_pred": []},
        "Matrix": {"target": [], "pred": [], "history": [], "random_pred": []},
        "MC_Ordinal": {"target": [], "pred": [], "history": [], "random_pred": []},
        "MC_Binary": {"target": [], "pred": [], "history": [], "random_pred": []}
    }
    
    unmappable_count = 0
    skipped_no_history = 0

    with open(args.predictions, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
                
            row = json.loads(line)
            pid = row.get("pid")
            q_id = str(row.get("question_id", ""))
            
            # --- STRICT INTERSECTION FILTER ---
            if (pid, q_id) not in historical_answers or historical_answers[(pid, q_id)] is None:
                skipped_no_history += 1
                continue
                
            target_str = str(row.get("target_answer", ""))
            pred_str = str(row.get("predicted_answer", ""))
            base_q_type = historical_types.get(q_id, "MC")
            
            base_qid = q_id.split('_')[0] if '_' in q_id else q_id
            q_mapping = mapping_dict.get(q_id, mapping_dict.get(base_qid, {}))

            # Dynamic Binary Fallback using Raw Strings
            if not q_mapping and "options" in row:
                opts = row["options"]
                if len(opts) == 2:
                    q_mapping = {str(opts[0]): 0, str(opts[1]): 1}

            q_type = base_q_type
            if base_q_type == "MC":
                if q_mapping and len(set(q_mapping.values())) <= 2:
                    q_type = "MC_Binary"
                else:
                    q_type = "MC_Ordinal"
            
            for group in ["Overall", q_type]:
                metrics[group]["total"] += 1

            if q_type in num_data:
                if q_mapping:
                    h_str = historical_answers[(pid, q_id)]
                    
                    t_val = q_mapping.get(target_str)
                    p_val = q_mapping.get(pred_str)
                    h_val = q_mapping.get(h_str)
                    
                    if t_val is not None and p_val is not None and h_val is not None:
                        unique_scale_values = list(set(q_mapping.values()))
                        random_val = random.choice(unique_scale_values)
                        
                        # Subgroup arrays
                        num_data[q_type]["target"].append(t_val)
                        num_data[q_type]["pred"].append(p_val)
                        num_data[q_type]["history"].append(h_val)
                        num_data[q_type]["random_pred"].append(random_val)
                        
                        # Master Overall arrays
                        num_data["Overall"]["target"].append(t_val)
                        num_data["Overall"]["pred"].append(p_val)
                        num_data["Overall"]["history"].append(h_val)
                        num_data["Overall"]["random_pred"].append(random_val)
                    else:
                        unmappable_count += 1
                else:
                    unmappable_count += 1

    print("\n" + "="*70)
    print(" LBM EVALUATION RESULTS (STRICT HISTORICAL INTERSECTION SUPPORT)")
    print("="*70)
    print(f"Filtered out {skipped_no_history} evaluation items lacking human baseline history.")
    if unmappable_count > 0:
        print(f"⚠️ Warning: {unmappable_count} rows could not be scaled/mapped to numerical integers.")
    
    final_report = {}
    
    
    agg_kappa = {"model_val": 0, "rand_val": 0, "human_val": 0, "model_n": 0, "rand_n": 0, "human_n": 0}
    
    # Iterate Subgroups FIRST, then Overall LAST to allow aggregation
    for group in ["Matrix", "MC_Ordinal", "MC_Binary", "Overall"]:
        if metrics[group]["total"] == 0: continue
            
        data = metrics[group]
        total = data["total"]
        
        print(f"\n--- {group.upper()} ---")
        print(f"Total Shared Support Evaluated: {total}")
        
        if group in num_data and len(num_data[group]["target"]) > 0:
            targets = np.array(num_data[group]["target"])
            preds = np.array(num_data[group]["pred"])
            hist = np.array(num_data[group]["history"])
            rands = np.array(num_data[group]["random_pred"])

            # --- 1. EXACT MATCH METRICS ---
            hist_exact_acc = np.mean(targets == hist) * 100
            model_exact_acc = np.mean(targets == preds) * 100
            rand_exact_acc = np.mean(targets == rands) * 100
            
            report_dict = {
                "total_questions": total,
                "exact_match_human_ceiling": round(hist_exact_acc, 2),
                "exact_match_accuracy": round(model_exact_acc, 2),
                "exact_match_random": round(rand_exact_acc, 2)
            }
            
            print(f"Set 1 Exact Acc  : {hist_exact_acc:.2f}% (Human Ceiling / Test-Retest)")
            print(f"Set 2 Exact Acc  : {model_exact_acc:.2f}% (Model Prediction)")
            print(f"Set 3 Exact Acc  : {rand_exact_acc:.2f}% (Random Baseline)")
            print("-" * 45)

            # --- 2. COHEN'S KAPPA METRICS ---
            if group != "Overall":
                # Calculate Independent Kappas for Subgroups
                if len(set(targets).union(set(preds))) > 1:
                    weight_type = 'quadratic' if group in ["Matrix", "MC_Ordinal"] else None
                    
                    kappa_model = cohen_kappa_score(targets, preds, weights=weight_type)
                    kappa_rand = cohen_kappa_score(targets, rands, weights=weight_type)
                    kappa_human = cohen_kappa_score(targets, hist, weights=weight_type)
                    
                    report_dict["kappa_human_ceiling"] = round(kappa_human, 4)
                    report_dict["kappa_model"] = round(kappa_model, 4)
                    report_dict["kappa_random"] = round(kappa_rand, 4)
                    
                    print(f"Set 1 Kappa      : {kappa_human:.4f} (Human Ceiling / Test-Retest)")
                    print(f"Set 2 Kappa      : {kappa_model:.4f} (Model Prediction)")
                    print(f"Set 3 Kappa      : {kappa_rand:.4f} (Random Baseline)")
                    
                    # Store values for Overall Weighted Average
                    agg_kappa["model_val"] += (kappa_model * total)
                    agg_kappa["model_n"] += total
                    agg_kappa["rand_val"] += (kappa_rand * total)
                    agg_kappa["rand_n"] += total
                    agg_kappa["human_val"] += (kappa_human * total)
                    agg_kappa["human_n"] += total
                else:
                    print("Kappa            : 0.0000 (Low Variance - Model predicted same class)")
            
            else:
                # Calculate Stratified Weighted Average Kappa for OVERALL block
                if agg_kappa["model_n"] > 0:
                    overall_kappa_human = agg_kappa["human_val"] / agg_kappa["human_n"]
                    overall_kappa_model = agg_kappa["model_val"] / agg_kappa["model_n"]
                    overall_kappa_rand = agg_kappa["rand_val"] / agg_kappa["rand_n"]
                    
                    report_dict["kappa_human_ceiling_w_avg"] = round(overall_kappa_human, 4)
                    report_dict["kappa_model_w_avg"] = round(overall_kappa_model, 4)
                    report_dict["kappa_random_w_avg"] = round(overall_kappa_rand, 4)

                    print(f"Set 1 W.Avg Kappa: {overall_kappa_human:.4f} (Human Ceiling / Test-Retest)")
                    print(f"Set 2 W.Avg Kappa: {overall_kappa_model:.4f} (Model Prediction)")
                    print(f"Set 3 W.Avg Kappa: {overall_kappa_rand:.4f} (Random Baseline)")

        final_report[group] = report_dict

    print("\n" + "="*70)

    if args.metrics_path:
        os.makedirs(os.path.dirname(args.metrics_path), exist_ok=True)
        with open(args.metrics_path, 'w', encoding='utf-8') as f:
            json.dump(final_report, f, indent=4)
        print(f"Metrics saved to {args.metrics_path}")

if __name__ == "__main__":
    main()