import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from transformers import TrainingArguments, Trainer
from datasets import Dataset
import numpy as np
import evaluate
from tqdm.auto import tqdm

# Charge model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-cased', num_labels=2)

# Load dataset
def load_dataset(file_paths, label):
    dataset = []
    for file_path in file_paths:
        with open(file_path, "r") as file:
            for line in file:
                encoded_line = tokenizer(line, truncation=True, padding='max_length', max_length=512)
                encoded_line['label'] = label
                dataset.append(encoded_line)
    return dataset

# File paths e labels
antisocial_files = [f"/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/Human Language Technologies/python/Personality Project/corpus/antisocial {i}.txt" for i in range(1, 11)]
nonantisocial_files = [f"/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/Human Language Technologies/python/Personality Project/corpus/antisocial {i}.txt" for i in range(11, 21)]

antisocial_dataset = load_dataset(antisocial_files, 0)
nonantisocial_dataset = load_dataset(nonantisocial_files, 1)

# Create a single dataset
dataset = antisocial_dataset + nonantisocial_dataset
hf_dataset = Dataset.from_list(dataset)

# train and eval data
split_dataset = hf_dataset.train_test_split(test_size=0.5)
train_dataset = split_dataset['train']
eval_dataset = split_dataset['test']

# Train arguments
training_args = TrainingArguments(
    output_dir="test_trainer",
    evaluation_strategy="epoch",
    logging_dir="logs",
)

# Accuracy
metric = evaluate.load("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    compute_metrics=compute_metrics,
)

# model training
trainer.train()

# Save it
model.save_pretrained("/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/Human Language Technologies/python/Personality Project/corpus")
tokenizer.save_pretrained("/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/Human Language Technologies/python/Personality Project/corpus")