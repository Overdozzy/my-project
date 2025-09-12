Mitigating Gender Bias in Pre-trained Language Models: A Counterfactual Approach
Author: Chidozie Marvelous Okpala
Enrolment Number: 30115693
Course: MSc Artificial Intelligence


1. Project Overview

This project examines and addresses gender bias in the bert-base-uncased pre-trained language model. The main approach is a Counterfactual Data Augmentation (CDA) method utilized on the "Bias in Bios" dataset. This procedure generates a dataset free from bias that is subsequently utilized to refine the model.

The efficiency of this debiasing method is assessed via a thorough, multi-dimensional evaluation:

Intrinsic Bias Examination: A rigorous, multi-template assessment employing Masked Language Modeling (MLM) to gauge the model's fundamental word connections. The outcomes are confirmed using a paired t-test to verify statistical significance.
Extrinsic Bias Assessment: A new Causal Mediation Analysis is conducted to evaluate the causal impact of gendered terms on the model's performance in downstream tasks (profession classification).
Interactive Demonstration: A Streamlit app offers a real-time, visual contrast of the attention mechanisms in the biased and debiased models
----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------






2. Directory Structure
The project is organized into a series of numbered Python scripts, saved model artifacts, and documentation.
MSc_Project_Code/
├── biased_control_bert_model/      # The fine-tuned, size-matched biased control model.
├── debiased_bert_model_bias_in_bios/ # The fine-tuned, debiased model.
|
├── 1_create_cda_dataset.py           # Script to generate the counterfactual dataset.
├── 2_finetune_model.py               # Script to train the debiased model.
├── 3_evaluate_baseline_final.py      # Script to evaluate the original BERT model's bias.
├── 4_evaluate_debiased_final.py      # Script to evaluate the debiased model's bias.
├── 5_perform_statistical_test_final.py # Script for t-tests and generating the results chart.
├── 6_train_biased_model_sized.py     # Script to train the size-matched control model.
├── 7_perform_causal_analysis.py      # Script for the final causal mediation analysis.
|
├── app.py                            # The Streamlit interactive demonstration application.
|
├── requirements.txt                  # A list of all required Python libraries.
└── README.md                         # This documentation file.


3. Setup Instructions
To run this project, it is recommended to use a Python virtual environment.
1. Create a Virtual Environment (Optional but Recommended):
python -m venv venv
# On Windows
.\venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate










----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------



2. Install Dependencies:

Install all required Python packages using the provided requirements.txt file.

pip install -r requirements.txt



----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------




3 How to Run the Experiments


There are two primary ways to run this project: a full reproduction that includes re-training the models, or a quick evaluation that uses the provided, pre-trained models.

Option A: Full Reproduction (including model training)
This option will regenerate all artifacts from scratch.

Important Note on Hardware:
The model training scripts (2_finetune_model.py and 6_train_biased_model_sized.py) are computationally intensive. It is highly recommended to run them in an environment with a CUDA-enabled GPU (such as Google Colab or Kaggle). Running on a CPU-only machine is possible but will be significantly slower (potentially taking several hours per script).

Execution Order:

Phase 1: Data Creation & Model Training
(Run these first, preferably on a GPU)

python 1_create_cda_dataset.py

python 2_finetune_model.py

python 6_train_biased_model_sized.py

Phase 2: Analysis & Evaluation (These can be run locally on a CPU) 4. python 3_evaluate_baseline_final.py 5. python 4_evaluate_debiased_final.py 6. python 5_perform_statistical_test_final.py 7. python 7_perform_causal_analysis.py
Option B: Quick Evaluation (Using Provided Models)

This is the recommended method for quickly verifying the project's findings without re-running the lengthy training process. I have included the finetuned models unzipped, the two model folders (biased_control_bert_model and debiased_bert_model_bias_in_bios) into the main project directory.

You can skip the training steps and run the analysis scripts directly.

Execution Order:

python 3_evaluate_baseline_final.py

python 4_evaluate_debiased_final.py

python 5_perform_statistical_test_final.py

python 7_perform_causal_analysis.py







----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

4. How to Run the ipynb Demonstration

The ipynb file is also with me. It was quite difficult to get the same result on my local machine as on Kaggle, so I included the file to represent the work shown in my dissertation.

