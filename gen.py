import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments
import torch
import numpy as np
from datasets import Dataset, load_metric
from transformers import pipeline
import os
import csv
import argparse

def main(data_path, model_save_path, input_directory, results_file):
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

    # Tokenize datasets
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding="max_length", truncation=True, max_length=512)

    train_data = Dataset.from_pandas(pd.DataFrame({'text': train_texts, 'label': train_labels}))
    val_data = Dataset.from_pandas(pd.DataFrame({'text': val_texts, 'label': val_labels}))

    train_data = train_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir='./results',          # output directory
        num_train_epochs=3,              # number of training epochs
        per_device_train_batch_size=8,   # batch size for training
        per_device_eval_batch_size=16,   # batch size for evaluation
        warmup_steps=500,                # number of warmup steps for learning rate scheduler
        weight_decay=0.01,               # strength of weight decay
        logging_dir='./logs',            # directory for storing logs
        logging_steps=10,
        evaluation_strategy="epoch"      # Use "evaluation_strategy" instead of "eval_strategy"
    )

    # Define accuracy metric
    metric = load_metric("accuracy")

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
    model.save_pretrained(model_save_path)
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
