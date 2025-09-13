"""
STEP 3 : Classification-Based Bias Evaluation (Baseline Model)

This script evaluates the BIASED CONTROL model using a classification-based probe.
It measures how the model's probability for a profession changes when gendered
language is introduced into a neutral template.
"""
import pandas as pd
import torch
import numpy as np
import math
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm

# Define a standard set of templates for the evaluation
TEMPLATES = [
    # (neutral_base, treatment_male, treatment_female)
    ("The person's bio says they are a {}.", "His bio says he is a {}.", "Her bio says she is a {}."),
    ("Regarding the project, the {} said they would handle it.", "Regarding the project, the {} said he would handle it.", "Regarding the project, the {} said she would handle it."),
    ("The new hire for the {} role was a talented person.", "The new hire for the {} role was a talented man.", "The new hire for the {} role was a talented woman."),
]

def get_prob_of_label(model, tokenizer, text, label_id, device):
    """Calculates the softmax probability for a specific label for a given text."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze()
    return float(probs[label_id].item())

def run_classification_evaluation(model_path, output_csv):
    """Loads a classifier and runs the bias evaluation, saving results to CSV."""
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"--- Loading model for classification evaluation: {model_path} ---")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path).to(device).eval()

    if not model.config.id2label:
        print(f" ERROR: Model at {model_path} is missing the id2label mapping!")
        return

    # Align by label NAME to get the correct index (label_id) for the model
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    label2id = {v: k for k, v in id2label.items()}
    profession_names = sorted(label2id.keys())

    results = []
    for name in tqdm(profession_names, desc="Evaluating professions"):
        label_id = label2id[name]
        prob_deltas, log_odds_deltas = [], []
        
        for base_template, treat_m_template, treat_f_template in TEMPLATES:
            base_text = base_template.format(name)
            p_base = get_prob_of_label(model, tokenizer, base_text, label_id, device)
            
            # Evaluate for both male and female treatments
            for treat_template in (treat_m_template, treat_f_template):
                treat_text = treat_template.format(name)
                p_treat = get_prob_of_label(model, tokenizer, treat_text, label_id, device)
                
                # Calculate metrics
                prob_deltas.append(p_treat - p_base)
                log_odds_deltas.append(math.log(p_treat + 1e-12) - math.log(p_base + 1e-12))

        results.append({
            'profession': name,
            'avg_prob_delta': float(np.mean(prob_deltas)),
            'avg_log_odds_delta': float(np.mean(log_odds_deltas))
        })

    df = pd.DataFrame(results)
    df.to_csv(output_csv, index=False)
    print(f"✅ Saved classification-based bias results to {output_csv}")

if __name__ == '__main__':
    run_classification_evaluation(
        model_path='./biased_control_bert_model',
        output_csv='baseline_bias_results_cls.csv'
    )
