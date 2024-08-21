This project involves training a RoBERTa model for text classification using distributed data parallel (DDP) training across multiple GPUs. After training, the model is used to classify text files as belonging to one of the predefined classes (e.g., AI-generated or human-written). The script handles everything from data preprocessing to model training, evaluation, and prediction on new data.

Requirements:<br/>

Python 3.x<br/>
PyTorch<br/>
Transformers (Hugging Face)<br/>
Datasets<br/>
scikit-learn<br/>
pandas<br/>
argparse<br/>
Distributed Data Parallel (DDP) setup with torch.distributed<br/>

Installation:<br/>

pip install torch transformers datasets scikit-learn pandas argparse evaluate<br/>

Script Details:<br/>
1. Data Loading and Preparation<br/>
The script begins by loading a CSV dataset containing text and label columns. It ensures the text is properly formatted and labels are integers. The dataset is then split into training and validation sets with an 80/20 split.<br/>

2. Model and Tokenizer<br/>
The script uses the RobertaTokenizer and RobertaForSequenceClassification from the Hugging Face Transformers library. The model is wrapped in PyTorch’s DistributedDataParallel (DDP) for efficient multi-GPU training.<br/>

3. Training Configuration<br/>

Output Directory: Location to save the model and checkpoints.<br/>
Batch Size: Configurable per device for training and evaluation.<br/>
Logging and Saving: Logs and checkpoints are saved after each epoch.<br/>
Mixed Precision: Enabled for faster training with FP16.<br/>
Gradient Accumulation: Simulates larger batch sizes by accumulating gradients.<br/>
DDP Setup: Local rank and other settings are configured for distributed training.<br/>

4. Training and Evaluation<br/>
The model is trained on the prepared dataset. After training, it is evaluated on the validation set, with accuracy as the evaluation metric.<br/>

5. Prediction<br/>
After training, the model is used to classify new text files located in a specified directory. The classification results, including the predicted label and confidence score, are saved to a CSV file.<br/>

Usage<br/>
python script_name.py --data_path /path/to/dataset.csv --model_save_path /path/to/save/model --input_directory /path/to/text/files --results_file /path/to/save/results.csv<br/>

export CUDA_VISIBLE_DEVICES=0,1  # Modify this to match your GPU setup<br/>
python -m torch.distributed.launch --nproc_per_node=2 script_name.py --data_path /path/to/dataset.csv --model_save_path /path/to/save/model --input_directory /path/to/text/files --results_file /path/to/save/results.csv<br/>

Output<br/>
Model: The trained RoBERTa model is saved to the specified directory.<br/>
Results: A CSV file with predictions and confidence scores is generated.<br/>

Usage of Pred.py<br/>
This script loads a trained RoBERTa model to classify text files as either AI-generated or human-written. The results are saved to a CSV file.<br/>

python script_name.py --model_save_path /path/to/model --input_directory /path/to/text/files --results_file /path/to/save/results.csv<br/>
Output<br/>
Results: A CSV file containing the file number and the predicted label (AI or Human).<br/>
