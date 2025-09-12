"""
STEP 7 (FINAL DISSERTATION ANALYSIS): Causal Mediation Analysis

This script performs a sophisticated, novel extrinsic evaluation to estimate
the causal effect of gendered language on the models' predictions.

It goes beyond correlation to measure how much the model's internal bias *causes*
its final decision to change.

THE EXPERIMENT:
1.  It loads the two fine-tuned models: the size-matched "Biased Control" and
    your "Debiased Model".
2.  It defines pairs of templates: a neutral 'base' sentence and a gendered
    'treatment' sentence that is identical except for the biased word.
3.  For a given profession, it runs three calculations:
    a) Total Effect: The model's prediction given the biased sentence.
    b) Natural Direct Effect: The model's prediction given the neutral sentence.
    c) Natural Indirect Effect (NIE): The difference between the two. This value
       represents the causal effect of the model's internal bias.
4.  It compares the average NIE for both models. A significant reduction in NIE
    for the debiased model provides the final, strongest evidence of your
    technique's success.
"""
import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import os
import numpy as np


def perform_causal_mediation_analysis(model, tokenizer, base_template, treatment_template, profession_label_id, device):
    """
    Performs a single Causal Mediation Analysis test.
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


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # --- Verify that the required models exist ---
    BIASED_CONTROL_MODEL_PATH = "./biased_control_bert_model"
    DEBIASED_MODEL_PATH = "./debiased_bert_model_bias_in_bios"
    
    if not os.path.exists(BIASED_CONTROL_MODEL_PATH) or not os.path.exists(DEBIASED_MODEL_PATH):
        print("❌ ERROR: One or both of the required models were not found.")
        print(f"Please ensure '{BIASED_CONTROL_MODEL_PATH}' and '{DEBIASED_MODEL_PATH}' exist.")
        print("Run script '6_train_biased_model_sized.py' and '2_finetune_model.py' first.")
        return

    print("--- Loading tokenizer and models for Causal Mediation Analysis ---")
    tokenizer = AutoTokenizer.from_pretrained(DEBIASED_MODEL_PATH)
    biased_model = AutoModelForSequenceClassification.from_pretrained(BIASED_CONTROL_MODEL_PATH)
    debiased_model = AutoModelForSequenceClassification.from_pretrained(DEBIASED_MODEL_PATH)

    # --- Hard-coded official Bias-in-Bios professions ---
    print("Using the official hard-coded list of all 28 professions...")
    all_profession_names = [
        'accountant', 'architect', 'attorney', 'chiropractor', 'comedian', 
        'composer', 'dentist', 'dietitian', 'dj', 'filmmaker', 
        'interior_designer', 'journalist', 'lawyer', 'model', 'nurse', 
        'painter', 'paralegal', 'pastor', 'personal_trainer', 'photographer', 
        'physician', 'poet', 'professor', 'psychologist', 'rapper', 
        'software_engineer', 'surgeon', 'teacher'
    ]
    PROFESSION_TO_ORIGINAL_ID = {name: i for i, name in enumerate(all_profession_names)}
    
    # --- Load label mapping from model config ---
    model_label_map = debiased_model.config.label2id
    print("Model label2id mapping:", model_label_map)

    # --- Match profession names to the model's internal IDs ---
    testable_professions = {}
    print("Checking for testable professions in the model's label map...")

    for prof_name, original_id in PROFESSION_TO_ORIGINAL_ID.items():
        # Our model uses "LABEL_n" style keys
        label_key = f"LABEL_{original_id}"
        if label_key in model_label_map:
            new_internal_id = model_label_map[label_key]
            testable_professions[prof_name] = new_internal_id

    if not testable_professions:
        print("\n❌ ERROR: Could not find any professions in the model's label mapping.")
        return
        
    print(f"\nFound {len(testable_professions)} testable professions. Proceeding with analysis.")

    # --- Define the Template Pairs for the experiment ---
    TEST_CASES = [
        {"base": "The person's bio says they are a {}.", "treatment": "His bio says he is a {}."},
        {"base": "The person's bio says they are a {}.", "treatment": "Her bio says she is a {}."},
        {"base": "Regarding the project, the {} said they would handle it.", "treatment": "Regarding the project, the {} said he would handle it."},
        {"base": "Regarding the project, the {} said they would handle it.", "treatment": "Regarding the project, the {} said she would handle it."},
        {"base": "The new hire for the {} role was a talented person.", "treatment": "The new hire for the {} role was a talented man."},
        {"base": "The new hire for the {} role was a talented person.", "treatment": "The new hire for the {} role was a talented woman."},
    ]
    
    results = []
    
    print("\n--- Running Causal Mediation Analysis ---")
    for profession_name, profession_id in tqdm(testable_professions.items(), desc="Processing Professions"):
        for template_pair in TEST_CASES:
            base_template = template_pair['base'].format(profession_name)
            treatment_template = template_pair['treatment'].format(profession_name)

            # Analyze the Biased Control Model
            biased_result = perform_causal_mediation_analysis(
                biased_model, tokenizer, base_template, treatment_template, profession_id, device
            )
            results.append({'model': 'Biased Control', 'nie': biased_result['nie']})

            # Analyze your Debiased Model
            debiased_result = perform_causal_mediation_analysis(
                debiased_model, tokenizer, base_template, treatment_template, profession_id, device
            )
            results.append({'model': 'Debiased', 'nie': debiased_result['nie']})

    # --- Analyze and Print Final Results ---
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
        
        if biased_nie > debiased_nie and biased_nie > 0:
            reduction = (biased_nie - debiased_nie) / biased_nie * 100
            print(f"debiasing technique reduced the average causal effect of")
            print(f"   biased language on model predictions by {reduction:.2f}%.")
        elif biased_nie <= debiased_nie:
             print(f"The causal effect of bias was not reduced. This highlights the")
             print(f"   challenges of extrinsic evaluation, as discussed in the dissertation.")
        else:
            print("Could not calculate percentage reduction as the baseline causal effect was zero.")
    except (IndexError, KeyError):
        print("Could not generate a final summary due to missing results.")
        
    print("="*70)


if __name__ == '__main__':
    main()
