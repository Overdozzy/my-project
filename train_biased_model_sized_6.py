"""
STEP 6 (): Train a Size-Matched Biased Control Model

This script implements the crucial step of training the "Biased Control" model
on a dataset that is perfectly size-matched to the counterfactual dataset.

This ensures a true "apples-to-apples" comparison in the final evaluation,
isolating the debiasing technique as the only variable.

THE PROCESS:
1.  It first loads your counterfactual dataset (`cda_debiasing_dataset...csv`)
    to determine its exact size (N).
2.  It then loads the original, biased `Bias in Bios` dataset, but takes only
    the first N examples.
3.  It fine-tunes a fresh BERT model on this size-matched, biased dataset.
4.  The resulting model is the perfect control for the final evaluations.
"""
import pandas as pd
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from datasets import load_dataset, Features, ClassLabel, Value, Dataset, load_dataset_builder
import os

def main():
    # --- 1. Determine the size of the counterfactual dataset ---
    CDA_DATA_FILE = "cda_debiasing_dataset_BIAS_IN_BIOS.csv"
    if not os.path.exists(CDA_DATA_FILE):
        print(f" ERROR: Counterfactual dataset not found at '{CDA_DATA_FILE}'.")
        print("Please run script '1_create_cda_dataset.py' first.")
        return
        
    cda_df = pd.read_csv(CDA_DATA_FILE)
    target_dataset_size = len(cda_df)
    print(f"Found counterfactual dataset with {target_dataset_size} examples.")
    print("The Biased Control model will be trained on this many examples.")

    # --- 2. Setup Tokenizer ---
    # We can use a fresh tokenizer or one from the debiased model path.
    # Using a fresh one is cleanest for this control model.
    MODEL_NAME = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # --- 3. Load and slice the original Bias in Bios dataset ---
    print(f"\nLoading the first {target_dataset_size} examples from the original 'Bias in Bios' dataset...")
    original_dataset = load_dataset("LabHC/bias_in_bios", split=f"train[:{target_dataset_size}]")

    df = original_dataset.to_pandas()
    df['text'] = df['hard_text'].astype(str)
    df['label'] = df['profession'].astype(int)
    
    unique_labels = sorted(df['label'].unique())
    num_labels = len(unique_labels)
    
    # Get the official label names for the full dataset to create a consistent config
    # The datasets library is having trouble inferring the ClassLabel type.
    # To fix this, we will hard-code the known list of 28 professions.
    print("Using a hard-coded list of professions to ensure robustness...")
    all_profession_names = [
        'accountant', 'architect', 'attorney', 'chiropractor', 'comedian', 
        'composer', 'dentist', 'dietitian', 'dj', 'filmmaker', 
        'interior_designer', 'journalist', 'model', 'nurse', 'painter', 
        'paralegal', 'pastor', 'personal_trainer', 'photographer', 'physician', 
        'poet', 'professor', 'psychologist', 'rapper', 'software_engineer', 
        'surgeon', 'teacher', 'yoga_instructor'
    ]
    
    # The model needs to know the full set of possible labels, even if not all are in this subset
    full_num_labels = len(all_profession_names)
    
    # Create the label mapping for the model's configuration
    id2label = {i: name for i, name in enumerate(all_profession_names)}
    label2id = {name: i for i, name in enumerate(all_profession_names)}
    
    print(f"Found {num_labels} unique professions in this subset of {target_dataset_size} examples.")
    print(f"Model will be configured for all {full_num_labels} possible professions.")

    # --- 4. Prepare dataset for Hugging Face Trainer ---
    features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=full_num_labels, names=all_profession_names)
    })
    
    prepared_df = df[['text', 'label']]
    biased_hf_dataset = Dataset.from_pandas(prepared_df, features=features)
    
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True)
    
    tokenized_dataset = biased_hf_dataset.map(tokenize_function, batched=True)

    # --- 5. Train the Biased Control Model ---
    model_to_finetune = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, 
        num_labels=full_num_labels,
        id2label=id2label,
        label2id=label2id
    )

    BIASED_MODEL_PATH = "./biased_control_bert_model"
    
    training_args = TrainingArguments(
        output_dir="./results_biased_control_sized",
        num_train_epochs=2,
        per_device_train_batch_size=8,
        logging_steps=100,
        save_strategy="epoch",
        report_to="none"
    )
    trainer = Trainer(model=model_to_finetune, args=training_args, train_dataset=tokenized_dataset)

    print("\n Starting fine-tuning on SIZE-MATCHED BIASED dataset...")
    trainer.train()
    print(" Fine-tuning of biased control model complete.")
    
    trainer.save_model(BIASED_MODEL_PATH)
    tokenizer.save_pretrained(BIASED_MODEL_PATH)
    print(f"Size-matched biased control model saved to {BIASED_MODEL_PATH}")

if __name__ == '__main__':
    main()

