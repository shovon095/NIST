import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from datasets import Dataset
import evaluate
from transformers import pipeline
import os
import csv
import argparse
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

def main(data_path, model_save_path, input_directory, results_file):
    # Initialize the process group
    dist.init_process_group(backend='nccl')
    local_rank = int(os.getenv('LOCAL_RANK', '0'))
    torch.cuda.set_device(local_rank)

    # Load dataset
    df = pd.read_csv(data_path)

    # Prepare dataset
    df.columns = ['text', 'label']  # Ensure the columns are named 'text' and 'label'
    df['label'] = df['label'].astype(int)  # Ensure labels are integers

    # Split dataset into training and validation sets (80/20 split)
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Load tokenizer and model
    model_name = "roberta-base"
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    # Move model to GPU and wrap it in DistributedDataParallel
    model.to(local_rank)
    model = DDP(model, device_ids=[local_rank])

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_data = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
    val_data = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'label': val_labels}))

    # Determine the number of CPU cores for parallel processing
    num_proc = os.cpu_count()
    print(f"Number of CPU cores available: {num_proc}")

    train_data = train_data.map(tokenize_function, batched=True, num_proc=num_proc)
    val_data = val_data.map(tokenize_function, batched=True, num_proc=num_proc)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # number of training epochs
        per_device_train_batch_size=16,  # batch size for training
        per_device_eval_batch_size=32,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",     # Use "evaluation_strategy" instead of "eval_strategy"
        save_strategy="epoch",           # Save checkpoint per epoch
        load_best_model_at_end=True,     # Load the best model at the end of training
        fp16=True,                       # Use mixed precision
        dataloader_num_workers=4,        # Number of workers for data loading
        pin_memory=True,                 # Pin memory for faster data transfer to GPU
        gradient_accumulation_steps=2,   # Simulate larger batch size
        local_rank=local_rank,           # Local rank for distributed training
    )

    # Define accuracy metric
    metric = evaluate.load("accuracy")

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        return metric.compute(predictions=preds, references=p.label_ids)

    # Create Trainer instance
    trainer = Trainer(
        model=model,                         # the instantiated 🤗 Transformers model to be trained
        args=training_args,                  # training arguments, defined above
        train_dataset=train_data,            # training dataset
        eval_dataset=val_data,               # evaluation dataset
        compute_metrics=compute_metrics      # the callback that computes metrics of interest
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Validation Accuracy: {eval_result['eval_accuracy']}")

    # Save the model
    if local_rank == 0:  # Save only from the main process
        model.module.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

    # Use the trained model for prediction on new files
    classifier = pipeline('text-classification', model=model_save_path, tokenizer=model_save_path)

    # Example usage with the previous script's files
    results = []

    for filename in os.listdir(input_directory):
        if filename.endswith(".txt"):
            summary_path = os.path.join(input_directory, filename)
            with open(summary_path, 'r', encoding='utf-8') as file:
                summary_content = file.read()
                prediction = classifier(summary_content)[0]
                label = prediction['label']
                score = prediction['score']
                results.append([filename, label, score])

    # Save results to CSV
    if local_rank == 0:  # Save only from the main process
        with open(results_file, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Filename', 'Prediction', 'Confidence'])
            writer.writerows(results)

    print("Results saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a RoBERTa model and use it for predictions on new files.")
    parser.add_argument("--data_path", type=str, required=True, help="Path to the CSV file containing the dataset.")
    parser.add_argument("--model_save_path", type=str, required=True, help="Path to save the trained model.")
    parser.add_argument("--input_directory", type=str, required=True, help="Directory containing the text files for prediction.")
    parser.add_argument("--results_file", type=str, required=True, help="Path to save the prediction results as a CSV file.")

    args = parser.parse_args()

    main(args.data_path, args.model_save_path, args.input_directory, args.results_file)
