"""
STEP 4 (FINAL, CLASSIFICATION-BASED):
Evaluate bias in the DEBIASED classifier using the same templates and metrics
as the baseline in Step 3.

Outputs: debiased_bias_results_cls.csv with per-profession average deltas.
"""
import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
import math

TEMPLATES = [
    ("The person's bio says they are a {}.", "His bio says he is a {}.", "Her bio says she is a {}."),
    ("Regarding the project, the {} said they would handle it.",
     "Regarding the project, the {} said he would handle it.",
     "Regarding the project, the {} said she would handle it."),
    ("The new hire for the {} role was a talented person.",
     "The new hire for the {} role was a talented man.",
     "The new hire for the {} role was a talented woman."),
]

def prob_of_label(model, tokenizer, text, label_id, device):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
    return float(probs[label_id].item())

def run_cls_eval(model_path, out_csv):
    print("--- STEP 4 (FINAL, CLS): Evaluating Bias in the DEBIASED Classifier ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    profession_names = sorted(label2id.keys())

    rows = []
    for name in profession_names:
        label_id = label2id[name]
        deltas = []
        logodds = []
        for base, treat_m, treat_f in TEMPLATES:
            base_text = base.format(name)
            for treat in (treat_m, treat_f):
                treat_text = treat.format(name)
                p_base = prob_of_label(model, tokenizer, base_text, label_id, device)
                p_treat = prob_of_label(model, tokenizer, treat_text, label_id, device)
                delta = p_treat - p_base
                deltas.append(delta)
                logodds.append(math.log(p_treat + 1e-12) - math.log(p_base + 1e-12))

        rows.append({
            'profession': name,
            'avg_prob_delta': float(np.mean(deltas)),
            'avg_log_odds_delta': float(np.mean(logodds))
        })

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"\nSaved classification-based debiased bias results to {out_csv}")

    # Summary
    avg_abs_prob = df['avg_prob_delta'].abs().mean()
    avg_abs_log = df['avg_log_odds_delta'].abs().mean()
    print("\n" + "="*60)
    print("      FINAL DEBIASED CLASSIFIER BIAS SCORES\n")
    print(f"Metric 1: Avg. Absolute Probability Delta: {avg_abs_prob:.4f}")
    print(f"Metric 2: Avg. Absolute Log-Odds Delta:   {avg_abs_log:.4f}")
    print("="*60)

if __name__ == '__main__':
    run_cls_eval('./debiased_bert_model_bias_in_bios', 'debiased_bias_results_cls.csv')
