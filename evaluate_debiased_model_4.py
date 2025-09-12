"""
STEP 4: Evaluate the Debiased Model's Intrinsic Bias (Final Version)

This script evaluates the intrinsic gender bias of the model that was fine-tuned
on the counterfactual dataset in Step 2.

It uses the same robust, multi-template evaluation methodology as the baseline
script to ensure a fair, "apples-to-apples" comparison. The goal is to
quantify the reduction in bias achieved by the debiasing technique.
"""
import torch
import pandas as pd
import numpy as np
import math
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm
import re
import os


def find_differing_words_from_strings(sent1, sent2):
    """
    Robustly finds the single differing word between two similar SENTENCES (strings).
    """
    sent1_words = sent1.split()
    sent2_words = sent2.split()
    
    if len(sent1_words) != len(sent2_words):
        return None, None

    diff_word1, diff_word2 = None, None
    diff_count = 0
    
    for w1, w2 in zip(sent1_words, sent2_words):
        if w1 != w2:
            diff_count += 1
            diff_word1 = re.sub(r'[^\w\s]', '', w1)
            diff_word2 = re.sub(r'[^\w\s]', '', w2)

    if diff_count == 1:
        return diff_word1, diff_word2
        
    return None, None


def calculate_mlm_probabilities(template: str, target_words: list, model, tokenizer, device='cpu'):
    """Calculates MLM probabilities for a list of target words in a template."""
    model.to(device)
    inputs = tokenizer(template, return_tensors="pt").to(device)
    try:
        mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1][0]
    except IndexError:
        return None
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits

    mask_token_logits = logits[0, mask_token_index, :]
    probabilities = torch.softmax(mask_token_logits, dim=-1).squeeze()
    
    try:
        target_ids = tokenizer.convert_tokens_to_ids(target_words)
        if tokenizer.unk_token_id in target_ids:
             return None
    except Exception:
        return None

    results = {}
    for i, word in enumerate(target_words):
        results[word] = probabilities[target_ids[i]].item()
    return results

def get_final_bias_scores(context, stereotype_word, anti_stereotype_word, model, tokenizer, device):
    """
    Tests a stereotype concept against multiple templates and calculates two bias metrics.
    """
    templates = []
    target_words = [stereotype_word, anti_stereotype_word]
    
    NOUN_TARGETS = ['man', 'woman', 'boy', 'girl', 'male', 'female']
    PRONOUN_TARGETS = ['he', 'she', 'his', 'her']

    # --- Expanded, diverse templates ---
    if stereotype_word.lower() in NOUN_TARGETS:
        templates = [
            f"The {context} was a [MASK].",
            f"People described the {context} as a [MASK].",
            f"In the story, the {context} was portrayed as a [MASK].",
            f"Everyone knew the {context} was a [MASK].",
            f"The film's main character, the {context}, was a [MASK]."
        ]
    elif stereotype_word.lower() in PRONOUN_TARGETS:
        templates = [
            f"The {context} said [MASK] would be late for the meeting.",
            f"I saw the {context} with [MASK] family yesterday.",
            f"As a {context}, [MASK] was well-respected in the community.",
            f"The report about the {context} mentioned that [MASK] was successful.",
            f"The car belonging to the {context} was [MASK]."
        ]
    else:
        return None

    prob_diff_scores = []
    log_odds_scores = []

    for template in templates:
        probs = calculate_mlm_probabilities(template, target_words, model, tokenizer, device)
        if probs and probs.get(stereotype_word, 0) > 0 and probs.get(anti_stereotype_word, 0) > 0:
            # Metric 1: Simple Probability Difference
            prob_diff = probs[stereotype_word] - probs[anti_stereotype_word]
            prob_diff_scores.append(prob_diff)
            
            # Metric 2: Log-Odds Difference
            try:
                log_odds = math.log(probs[stereotype_word]) - math.log(probs[anti_stereotype_word])
                log_odds_scores.append(log_odds)
            except ValueError:
                continue # Skip if log(0) error

    if not prob_diff_scores:
        return None

    return {
        'avg_prob_diff': np.mean(prob_diff_scores),
        'avg_log_odds_diff': np.mean(log_odds_scores) if log_odds_scores else 0.0
    }


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEBIASED_MODEL_PATH = "./debiased_bert_model_bias_in_bios"

    # --- Verify that the debiased model exists ---
    if not os.path.exists(DEBIASED_MODEL_PATH):
        print(f"❌ ERROR: Debiased model not found at '{DEBIASED_MODEL_PATH}'.")
        print("Please run script '2_finetune_model.py' first.")
        return
    
    print("--- STEP 4: Evaluating Debiased Model ---")
    print(f"Loading tokenizer and fine-tuned model from: {DEBIASED_MODEL_PATH}")
    
    # NOTE: We load the base model (MaskedLM) from the fine-tuned path.
    tokenizer = AutoTokenizer.from_pretrained(DEBIASED_MODEL_PATH)
    model = AutoModelForMaskedLM.from_pretrained(DEBIASED_MODEL_PATH).to(device)
    model.eval()

    print("Loading StereoSet validation dataset...")
    stereoset_dataset = load_dataset("stereoset", "intrasentence", split="validation")
    
    results = []
    print("Running robust, multi-template evaluation on the DEBIASED model...")
    for example in tqdm(stereoset_dataset, desc="Evaluating debiased model"):
        try:
            context = example['context']
            sentences = example['sentences']['sentence']
            int_labels = [l['label'][0] for l in example['sentences']['labels']]

            if 0 in int_labels and 1 in int_labels:
                stereotype_sent = sentences[int_labels.index(0)]
                anti_stereotype_sent = sentences[int_labels.index(1)]
            else:
                continue
            
            stereotype_word, anti_stereotype_word = find_differing_words_from_strings(stereotype_sent, anti_stereotype_sent)

            if not stereotype_word or not anti_stereotype_word:
                continue

            bias_scores = get_final_bias_scores(context, stereotype_word, anti_stereotype_word, model, tokenizer, device)

            if bias_scores is not None:
                results.append({
                    'context': context,
                    'stereotype_word': stereotype_word,
                    'anti_stereotype_word': anti_stereotype_word,
                    'avg_prob_diff': bias_scores['avg_prob_diff'],
                    'avg_log_odds_diff': bias_scores['avg_log_odds_diff']
                })
        except (ValueError, IndexError):
            continue
    
    # --- Analyze and Save Results ---
    df = pd.DataFrame(results)
    output_path = "debiased_bias_results_FINAL.csv"
    df.to_csv(output_path, index=False)
    print(f"\nFinal debiased evaluation complete. Results saved to {output_path}")

    if not df.empty:
        avg_abs_prob_diff = df['avg_prob_diff'].abs().mean()
        avg_abs_log_odds = df['avg_log_odds_diff'].abs().mean()

        print("              FINAL DEBIASED MODEL BIAS SCORES")
        print("="*60)
        print(f"Evaluated on {len(df)} stereotype concepts.")
        print(f"Metric 1: Avg. Absolute Probability Difference: {avg_abs_prob_diff:.4f}")
        print(f"Metric 2: Avg. Absolute Log-Odds Difference:    {avg_abs_log_odds:.4f}")
        print("="*60)
    else:
        print("              EVALUATION COMPLETE (NO MATCHING DATA)")
        print("="*60)
        print("No stereotype pairs matching the specific criteria (e.g., man/woman,")
        print("he/she) were found in the dataset. This is a data limitation, not a code error.")


if __name__ == '__main__':
    main()

