"""
This script generates a counterfactual dataset to mitigate gender bias.
It loads a subset of the "Bias in Bios" dataset, identifies biographies
containing gendered words, and creates gender-swapped versions. The final
dataset, containing both original and counterfactual examples, is saved
to a CSV file for model fine-tuning.
"""
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
import nltk
import os
from transformers import AutoTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

def setup_nltk():
    """Downloads necessary NLTK data packages if they do not exist."""
    print("Setting up NLTK...")
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
    try:
        nltk.data.find('corpora/wordnet')
    except LookupError:
        nltk.download('wordnet', quiet=True)
    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger', quiet=True)
    print("NLTK setup complete.")

def get_wordnet_pos(word):
    """Map POS tag to the format required by the WordNetLemmatizer."""
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def create_counterfactual(tokens, tokenizer, gender_swap_map, lemmatizer):
    """
    Swaps gendered words in a list of tokens to create a counterfactual sentence string.
    """
    new_tokens = tokens[:]
    swapped = False
    for i in range(len(new_tokens)):
        token = new_tokens[i]
        clean_token = token.replace("##", "")
        if clean_token.isalpha():
            lemma = lemmatizer.lemmatize(clean_token, get_wordnet_pos(clean_token))
            if lemma in gender_swap_map:
                swap_word = gender_swap_map[lemma]
                if token[0].isupper():
                    swap_word = swap_word.capitalize()
                new_tokens[i] = swap_word
                swapped = True
    
    if not swapped:
        return None
    
    return tokenizer.convert_tokens_to_string(new_tokens)

def main():
    setup_nltk()

    # Define gendered word pairs and create a comprehensive swap map
    GENDER_PAIRS = {
        "he": "she", "him": "her", "his": "her", "himself": "herself",
        "man": "woman", "boy": "girl", "male": "female", "father": "mother",
        "son": "daughter", "brother": "sister", "husband": "wife",
        "uncle": "aunt", "mr": "mrs", "sir": "madam", "king": "queen", "prince": "princess"
    }
    full_gender_swap_map = GENDER_PAIRS.copy()
    full_gender_swap_map.update({v: k for k, v in GENDER_PAIRS.items()})
    GENDER_LEMMAS_SET = set(full_gender_swap_map.keys())

    lemmatizer = WordNetLemmatizer()

    # Load tokenizer
    MODEL_NAME = "bert-base-uncased"
    print(f"Loading tokenizer: {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print("Tokenizer loaded.")

    # Load and process the Bias in Bios dataset
    print("Loading 'Bias in Bios' dataset...")
    dataset = load_dataset("LabHC/bias_in_bios", split="train[:15000]")

    print("Filtering for biographies with gendered words...")
    def find_gendered_words(example):
        text = str(example['hard_text']).lower()
        words = nltk.word_tokenize(text)
        lemmas = [lemmatizer.lemmatize(w, get_wordnet_pos(w)) for w in words]
        example['contains_gender'] = any(lemma in GENDER_LEMMAS_SET for lemma in lemmas)
        return example

    dataset = dataset.map(find_gendered_words, num_proc=2)
    filtered_dataset = dataset.filter(lambda example: example['contains_gender'])
    print(f"Found {len(filtered_dataset)} gendered bios to process.")

    # Create counterfactual pairs
    print("Creating counterfactual data augmentation (CDA) pairs...")
    cda_pairs = []
    for example in tqdm(filtered_dataset, desc="Processing biographies"):
        text = str(example['hard_text'])
        label = str(example['profession'])
        
        cda_pairs.append({'text': text, 'label': label})
        
        tokens = tokenizer.tokenize(text)
        counterfactual_sentence = create_counterfactual(tokens, tokenizer, full_gender_swap_map, lemmatizer)

        if counterfactual_sentence:
            cda_pairs.append({'text': counterfactual_sentence, 'label': label})

    # Save the final dataset
    cda_df = pd.DataFrame(cda_pairs)
    OUTPUT_FILE = "cda_debiasing_dataset_BIAS_IN_BIOS.csv"
    cda_df.to_csv(OUTPUT_FILE, index=False)
    
    print("\nCDA dataset creation complete.")
    print(f"Dataset created with {len(cda_df)} examples.")
    print(f"Saved to: {OUTPUT_FILE}")

if __name__ == '__main__':
    main()

