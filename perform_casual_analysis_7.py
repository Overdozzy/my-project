"""
STEP 7 (FINAL DISSERTATION ANALYSIS): Causal Mediation Analysis

This script estimates the *causal effect* of gendered language on model predictions.

It goes beyond correlation to measure how much biased words (e.g., "his", "her",
"man", "woman") *cause* the model to change its output probability.

THE EXPERIMENT:
1.  Define pairs of templates: a neutral "base" sentence and a gendered "treatment".
    Example:
      Base: "The person's bio says they are a doctor."
      Treatment: "His bio says he is a doctor."
2.  For each profession label:
      a) Total Effect = model probability with biased sentence.
      b) Direct Effect = model probability with neutral sentence.
      c) Natural Indirect Effect (NIE) = difference (a - b).
         → This captures the *causal impact* of the biased word.
3.  Compare the average NIE for:
      - Biased Control Model (trained on original data).
      - Debiased Model (trained on counterfactual data).
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os

# ==============================================================================
# 1. CORE CAUSAL MEDIATION ANALYSIS FUNCTION
# ==============================================================================
def perform_causal_mediation_analysis(model, tokenizer, base_template, treatment_template, profession_label_id, device):
    """
    Performs a single Causal Mediation Analysis test for one profession.
    """
    model.to(device)
    model.eval()

    with torch.no_grad():
        # --- a) Total Effect: probability under biased treatment sentence ---
        treatment_inputs = tokenizer(treatment_template, return_tensors="pt", truncation=True, max_length=128).to(device)
        treatment_outputs = model(**treatment_inputs)
        treatment_probs = torch.softmax(treatment_outputs.logits, dim=-1).squeeze()
        total_effect_prob = treatment_probs[profession_label_id].item()

        # --- b) Direct Effect: probability under neutral base sentence ---
        base_inputs = tokenizer(base_template, return_tensors="pt", truncation=True, max_length=128).to(device)
        base_outputs = model(**base_inputs)
        base_probs = torch.softmax(base_outputs.logits, dim=-1).squeeze()
        direct_effect_prob = base_probs[profession_label_id].item()

        # --- c) Natural Indirect Effect (NIE) ---
        nie = total_effect_prob - direct_effect_prob

    return {
        'total_effect_prob': total_effect_prob,
        'direct_effect_prob': direct_effect_prob,
        'nie': nie
    }

# ==============================================================================
# 2. MAIN SCRIPT EXECUTION
# ==============================================================================
def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Paths to saved fine-tuned models ---
    BIASED_CONTROL_MODEL_PATH = "./biased_control_bert_model"
    DEBIASED_MODEL_PATH = "./debiased_bert_model_bias_in_bios"
    
    if not os.path.exists(BIASED_CONTROL_MODEL_PATH) or not os.path.exists(DEBIASED_MODEL_PATH):
        print(" ERROR: One or both of the required models were not found.")
        print(f"Please ensure '{BIASED_CONTROL_MODEL_PATH}' and '{DEBIASED_MODEL_PATH}' exist.")
        return

    print("--- Loading tokenizer and models for Causal Mediation Analysis ---")
    tokenizer = AutoTokenizer.from_pretrained(DEBIASED_MODEL_PATH)
    biased_model = AutoModelForSequenceClassification.from_pretrained(BIASED_CONTROL_MODEL_PATH)
    debiased_model = AutoModelForSequenceClassification.from_pretrained(DEBIASED_MODEL_PATH)

    # --- Step 1: Extract profession mapping directly from model config ---
    # The config's id2label map is the most reliable source of truth.
    # e.g., config.id2label = { 0: "nurse", 1: "engineer", ... }
    model_id2label = {int(k): v for k, v in debiased_model.config.id2label.items()}
    testable_professions = {name: idx for idx, name in model_id2label.items()}
    
    if not testable_professions:
        print("\n ERROR: No profession labels found in the model's config.json.")
        return
        
    print(f"\nFound {len(testable_professions)} testable professions. Proceeding with analysis.\n")

    # --- Step 2: Define Template Pairs ---
    TEST_CASES = [
        # Pronoun-based stereotypes
        {"base": "The person's bio says they are a {}.", "treatment": "His bio says he is a {}."},
        {"base": "The person's bio says they are a {}.", "treatment": "Her bio says she is a {}."},
        {"base": "Regarding the project, the {} said they would handle it.", "treatment": "Regarding the project, the {} said he would handle it."},
        {"base": "Regarding the project, the {} said they would handle it.", "treatment": "Regarding the project, the {} said she would handle it."},
        # Noun-based stereotypes
        {"base": "The new hire for the {} role was a talented person.", "treatment": "The new hire for the {} role was a talented man."},
        {"base": "The new hire for the {} role was a talented woman.", "treatment": "The new hire for the {} role was a talented person."}, # anti-stereotype flip
    ]
    
    results = []
    
    # --- Step 3: Run Analysis ---
    print("--- Running Causal Mediation Analysis ---")
    for profession_name, profession_id in tqdm(testable_professions.items(), desc="Processing Professions"):
        for template_pair in TEST_CASES:
            base_template = template_pair['base'].format(profession_name)
            treatment_template = template_pair['treatment'].format(profession_name)

            # Biased Control Model
            biased_result = perform_causal_mediation_analysis(
                biased_model, tokenizer, base_template, treatment_template, profession_id, device
            )
            results.append({
                'model': 'Biased Control',
                'profession': profession_name,
                'nie': biased_result['nie']
            })

            # Debiased Model
            debiased_result = perform_causal_mediation_analysis(
                debiased_model, tokenizer, base_template, treatment_template, profession_id, device
            )
            results.append({
                'model': 'Debiased',
                'profession': profession_name,
                'nie': debiased_result['nie']
            })

    # --- Step 4: Summarize Results ---
    df = pd.DataFrame(results)
    df['abs_nie'] = df['nie'].abs()
    
    summary = df.groupby('model')['abs_nie'].mean().reset_index()
    summary = summary.rename(columns={'abs_nie': 'Average Causal Effect (Abs. NIE)'})

    print("\n" + "="*70)
    print("        FINAL CAUSAL MEDIATION ANALYSIS RESULTS")
    print("="*70)
    print("The 'Average Causal Effect' measures how much a biased word *causes*")
    print("the model's final prediction to change. A lower score is better.")
    print("-"*70)
    
    print(summary.to_string(index=False))

    print("\n" + "-"*70)
    print("                          SUMMARY")
    print("-"*70)
    
    try:
        biased_nie = summary[summary['model'] == 'Biased Control']['Average Causal Effect (Abs. NIE)'].iloc[0]
        debiased_nie = summary[summary['model'] == 'Debiased']['Average Causal Effect (Abs. NIE)'].iloc[0]
        
        if biased_nie > 0:
            reduction = (biased_nie - debiased_nie) / biased_nie * 100
            print(f"The debiasing technique reduced the average causal effect of")
            print(f"   biased language on model predictions by {reduction:.2f}%.")
        else:
            print("Could not calculate percentage reduction (baseline effect was zero).")
    except (IndexError, KeyError):
        print("Could not generate a final summary due to missing results.")
        
    print("="*70)

if __name__ == '__main__':
    main()

