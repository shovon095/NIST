# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from datasets import Dataset, load_metric
from transformers import pipeline
import csv
import argparse
import multiprocessing
import os

def main(args):
    # Load dataset
    df = pd.read_csv(args.data_path)

    # Prepare dataset
    df.columns = ['text', 'label']  # Ensure the columns are named 'text' and 'label'
    df['label'] = df['label'].astype(int)  # Ensure labels are integers

    # Split dataset into training and validation sets (80/20 split)
    train_texts, val_texts, train_labels, val_labels = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

    # Load tokenizer and model
    model_name = "distilroberta-base"  # Use DistilRoBERTa for faster training
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaForSequenceClassification.from_pretrained(model_name)

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_data = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
    val_data = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'label': val_labels}))

    # Get the number of available CPU cores
    num_proc = multiprocessing.cpu_count()
    print(f"Using {num_proc} processes for tokenization")

    # Map with batched processing and multiprocessing
    train_data = train_data.map(tokenize_function, batched=True, num_proc=num_proc)
    val_data = val_data.map(tokenize_function, batched=True, num_proc=num_proc)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=args.model_save_path,      # output directory
        num_train_epochs=3,                   # number of training epochs
        per_device_train_batch_size=16,       # batch size for training
        gradient_accumulation_steps=2,        # accumulate gradients over 2 steps
        per_device_eval_batch_size=16,        # batch size for evaluation
        warmup_steps=500,                     # number of warmup steps for learning rate scheduler
        weight_decay=0.01,                    # strength of weight decay
        logging_dir='./logs',                 # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch",          # evaluation strategy
        save_strategy="epoch",                # save strategy
        fp16=True,                            # Enable mixed precision training
        dataloader_num_workers=num_proc,      # Number of workers for data loading
        save_total_limit=2,                   # Limit the total amount of checkpoints
        load_best_model_at_end=True,          # Load the best model at the end of training
        metric_for_best_model="eval_loss",    # Metric to use to compare models
        greater_is_better=False               # Whether the metric_for_best_model should be maximized or not
    )

    # Load metrics
    accuracy_metric = load_metric('accuracy', trust_remote_code=True)
    precision_metric = load_metric('precision', trust_remote_code=True)
    recall_metric = load_metric('recall', trust_remote_code=True)
    f1_metric = load_metric('f1', trust_remote_code=True)

    def compute_metrics(p):
        preds = np.argmax(p.predictions, axis=1)
        accuracy = accuracy_metric.compute(predictions=preds, references=p.label_ids)
        precision = precision_metric.compute(predictions=preds, references=p.label_ids, average='binary')
        recall = recall_metric.compute(predictions=preds, references=p.label_ids, average='binary')
        f1 = f1_metric.compute(predictions=preds, references=p.label_ids, average='binary')
        return {
            'accuracy': accuracy['accuracy'],
            'precision': precision['precision'],
            'recall': recall['recall'],
            'f1': f1['f1']
        }

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

    # Save the model before evaluation to prevent retraining if an error occurs
    model.save_pretrained(args.model_save_path)
    tokenizer.save_pretrained(args.model_save_path)
    print('Model saved.')

    # Evaluate the model
    eval_result = trainer.evaluate()
    print(f"Validation Accuracy: {eval_result['eval_accuracy']}")
    print(f"Validation Precision: {eval_result['eval_precision']}")
    print(f"Validation Recall: {eval_result['eval_recall']}")
    print(f"Validation F1 Score: {eval_result['eval_f1']}")

    # Use the trained model for prediction on new files
    classifier = pipeline('text-classification', model=args.model_save_path, tokenizer=args.model_save_path)

    # Directory containing text files for prediction
    files_directory = args.input_directory

    # Predict and save results
    results = []
    for filename in os.listdir(files_directory):
        if filename.endswith('.txt'):
            file_path = os.path.join(files_directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                prediction = classifier(content)[0]
                label = prediction['label']
                score = prediction['score']
                results.append([filename, label, score])

    # Save results to CSV
    with open(args.results_file, 'w', encoding='utf-8', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Filename', 'Prediction', 'Confidence'])
        writer.writerows(results)

    print('Results saved.')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a RoBERTa model and use it for predictions on new files.')
    parser.add_argument('--data_path', type=str, required=True, help='Path to the CSV file containing the dataset.')
    parser.add_argument('--model_save_path', type=str, required=True, help='Path to save the trained model.')
    parser.add_argument('--input_directory', type=str, required=True, help='Directory containing the text files for prediction.')
    parser.add_argument('--results_file', type=str, required=True, help='Path to save the prediction results as a CSV file.')

    args = parser.parse_args()

    main(args)
