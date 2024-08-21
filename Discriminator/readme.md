# Active Learning and Fine-tuning BERT and Roberta for Text Classification

## Overview

Active learning.ipynb script performs active learning using BERT and RoBERTa models to classify text files as either AI-generated or human-written. It starts with an initial training phase using manually labeled data and then iteratively fine-tunes both models using high-confidence predictions from the active learning loop.
gen_multi.py trains a RoBERTa model for text classification using distributed data parallel (DDP) training across multiple GPUs. After training, the model is used to classify text files as belonging to one of the predefined classes (e.g., AI-generated or human-written). The script handles everything from data preprocessing to model training, evaluation, and prediction on new data.


"'" 
## **Usage Active learning.ipynb**

### File Paths

- **`input_directory`**: Path to the directory containing text files for prediction.
- **`output_directory`**: Path to the directory where results will be saved.
- **`results_file`**: Path to save the CSV file with prediction results.

### Execution

1. **Initial Training**: The script first performs initial training on a small manually labeled dataset.
2. **Active Learning Loop**: The script then enters an active learning loop, where it uses the trained models to make predictions on new text files, fine-tunes the models with high-confidence predictions, and iterates this process.

### Usage 

```python
input_directory = "/path/to/text/files/"
output_directory = "/path/to/save/results/"
results_file = "/path/to/save/results.csv"

# Run the initial training and active learning loop
initial_training()
active_learning_loop_bert_roberta(input_directory, output_directory, results_file)

print("Results saved.")
"'"

"'"
## **Usage gen_multi.py**

### Overview

This script is used to generate multiple text samples in a single run using a pre-trained model.

### File Paths

- **`input_prompts`**: Path to a text file containing prompts, each on a new line.

### Execution

1. **Text Generation**: The script processes multiple input prompts from the file and generates corresponding text outputs.

### Usage

```python
input_prompts = "/path/to/prompts.txt"
"'"

# Run the text generation script
python gen_multi.py --input_prompts=input_prompts

