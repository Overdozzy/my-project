"""
STEP 2: Fine-Tuning (The Debiasing Process)

This script loads the counterfactual dataset created in Step 1 and uses it
to fine-tune the BERT model. The task is profession classification.

By training on a dataset where gender is not predictive of the profession,
the model learns to reduce its reliance on gendered words for this task,
thus mitigating its bias.
"""
import pandas as pd
import torch
from datasets import load_dataset, Features, ClassLabel, Value
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer

def main():
    print("--- STEP 2: Fine-Tuning the Model on the CDA Dataset ---")
    
    # --- SETUP ---
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # --- LOAD THE PREPARED CDA DATASET ---
    cda_data_file = "cda_debiasing_dataset_BIAS_IN_BIOS.csv"
    print(f"Loading CDA debiasing dataset from '{cda_data_file}'...")

    df = pd.read_csv(cda_data_file)
    df['label'] = df['label'].astype(int)
    unique_labels = sorted(df['label'].unique())
    num_labels = len(unique_labels)
    print(f"Found {num_labels} unique profession labels.")

    cleaned_file = "cda_bias_in_bios_cleaned.csv"
    df.to_csv(cleaned_file, index=False)

    # Define features for the HuggingFace dataset loader
    cda_features = Features({
        'text': Value('string'),
        'label': ClassLabel(num_classes=num_labels, names=[str(x) for x in unique_labels])
    })

    cda_dataset = load_dataset('csv', data_files=cleaned_file, features=cda_features, split='train')

    # --- TOKENIZE DATASET ---
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=256)

    tokenized_dataset = cda_dataset.map(tokenize_function, batched=True)

    # --- MODEL SETUP ---
       # --- MODEL SETUP ---
    # Define the official list of all possible professions to ensure consistency
    all_profession_names = [
        'accountant', 'architect', 'attorney', 'chiropractor', 'comedian', 
        'composer', 'dentist', 'dietitian', 'dj', 'filmmaker', 
        'interior_designer', 'journalist', 'model', 'nurse', 'painter', 
        'paralegal', 'pastor', 'personal_trainer', 'photographer', 'physician', 
        'poet', 'professor', 'psychologist', 'rapper', 'software_engineer', 
        'surgeon', 'teacher', 'yoga_instructor'
    ]
    
    # Create the label mapping for the model's configuration
    id2label = {i: name for i, name in enumerate(all_profession_names)}
    label2id = {name: i for i, name in enumerate(all_profession_names)}
    
    # Load the model with the correct label mappings
    model_to_finetune = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(all_profession_names),
        id2label=id2label,
        label2id=label2id
    )
    model_to_finetune.to(device)
    # --- TRAINING ARGUMENTS ---
    training_args = TrainingArguments(
        output_dir="./results_bias_in_bios",
        num_train_epochs=2, # Keep epochs low to avoid catastrophic forgetting
        per_device_train_batch_size=8,
        learning_rate=2e-5, # A smaller learning rate is crucial for fine-tuning
        logging_dir='./logs_bias_in_bios',
        logging_steps=100,
        save_strategy="epoch",
        report_to="none"
    )

    trainer = Trainer(
        model=model_to_finetune,
        args=training_args,
        train_dataset=tokenized_dataset,
    )

    print("\n🚀 Starting fine-tuning...")
    trainer.train()
    print("✅ Fine-tuning complete.")

    # --- SAVE THE DEBIASED MODEL ---
    debiased_model_path = "./debiased_bert_model_bias_in_bios"
    trainer.save_model(debiased_model_path)
    tokenizer.save_pretrained(debiased_model_path)
    print(f"\nDebiased model saved to '{debiased_model_path}'")

if __name__ == '__main__':
    main()
