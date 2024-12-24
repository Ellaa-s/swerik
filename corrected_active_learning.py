import torch
import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from torch.optim import AdamW
from torch.nn.utils import clip_grad_norm_
import random
from safetensors.torch import load_file, save_file

# Load KB-BERT model and tokenizer
tokenizer = BertTokenizer.from_pretrained("KB/bert-base-swedish-cased")
saved_model_path = "output/output/margin_prediction_model/model.safetensors"  # Path to the saved model
model = BertForSequenceClassification.from_pretrained("KB/bert-base-swedish-cased", num_labels=2)
model.load_state_dict(load_file(saved_model_path))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Debugging: Check CUDA usage
if device.type == "cuda":
    print(f"Using CUDA: {torch.cuda.get_device_name(torch.cuda.current_device())}")
else:
    print("Using CPU")

# Load data from CSV
data_path = "data_new_annotation.csv"
data = pd.read_csv(data_path)
unlabeled_data = data["text_line"].tolist()

# Debugging prints
print(f"Unlabeled Data: {len(unlabeled_data)} rows loaded.")

# Define a Dataset class
class TextDataset(Dataset):
    def __init__(self, texts, labels=None, tokenizer=None, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        if self.labels is not None:
            item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item

# Active Learning parameters
batch_size = 8
learning_rate = 1e-5
num_epochs = 5
pool_subset_size = 50  # Number of samples to evaluate from the pool per iteration
acquisition_size = 10  # Number of samples to acquire for labeling per iteration
patience = 2  # Early stopping patience

# Training function with additional metrics
def train_model(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    for batch in dataloader:
        inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
        labels = batch['labels'].to(device)
        optimizer.zero_grad()
        outputs = model(**inputs, labels=labels)
        loss = criterion(outputs.logits, labels)
        loss.backward()
        clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return total_loss / len(dataloader), accuracy, precision, recall, f1

# Evaluation function with additional metrics
def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            labels = batch['labels'].to(device)
            outputs = model(**inputs)
            loss = criterion(outputs.logits, labels)
            total_loss += loss.item()

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average="weighted")
    recall = recall_score(all_labels, all_preds, average="weighted")
    f1 = f1_score(all_labels, all_preds, average="weighted")

    return total_loss / len(dataloader), accuracy, precision, recall, f1

# Uncertainty sampling function
def uncertainty_sampling(model, dataloader):
    model.eval()
    uncertainties = []
    with torch.no_grad():
        for batch in dataloader:
            inputs = {key: val.to(device) for key, val in batch.items() if key != 'labels'}
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=1)
            uncertainty = 1 - torch.max(probs, dim=1).values.cpu().numpy()
            uncertainties.extend(uncertainty)
    return np.array(uncertainties)

# Function to simulate human annotation
def human_annotation(texts):
    print("Please label the following texts as '0' or '1':")
    labels = []
    for text in texts:
        print(f"Text: {text}")
        while True:
            try:
                label = int(input("Label (0 or 1): "))
                if label in [0, 1]:
                    labels.append(label)
                    break
                else:
                    print("Invalid input. Please enter 0 or 1.")
            except ValueError:
                print("Invalid input. Please enter 0 or 1.")
    return labels

# Active Learning loop
labeled_data = []
labeled_labels = []
labeled_dataset = TextDataset(labeled_data, labeled_labels, tokenizer)
optimizer = AdamW(model.parameters(), lr=learning_rate)
criterion = torch.nn.CrossEntropyLoss()

for iteration in range(5):  # Number of Active Learning iterations
    print(f"Iteration {iteration}: Starting Active Learning iteration")
    print(f"Labeled dataset size: {len(labeled_data)}, Unlabeled dataset size: {len(unlabeled_data)}")

    # Train model on labeled data if available
    if len(labeled_data) > 0:
        labeled_dataloader = DataLoader(labeled_dataset, batch_size=batch_size, shuffle=True)
        for epoch in range(num_epochs):
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = train_model(
                model, labeled_dataloader, optimizer, criterion
            )
            print(
                f"Epoch {epoch + 1}/{num_epochs} - "
                f"Loss: {train_loss:.4f}, Accuracy: {train_accuracy:.4f}, "
                f"Precision: {train_precision:.4f}, Recall: {train_recall:.4f}, F1: {train_f1:.4f}"
            )

    # Uncertainty sampling
    if len(unlabeled_data) < pool_subset_size:
        pool_subset_size = len(unlabeled_data)

    subset_indices = random.sample(range(len(unlabeled_data)), pool_subset_size)
    subset_texts = [unlabeled_data[i] for i in subset_indices]
    subset_dataset = TextDataset(subset_texts, None, tokenizer)
    subset_dataloader = DataLoader(subset_dataset, batch_size=batch_size)

    uncertainties = uncertainty_sampling(model, subset_dataloader)
    top_indices = uncertainties.argsort()[-acquisition_size:][::-1]

    # Acquire new labels through human-in-the-loop annotation
    acquired_texts = [subset_texts[i] for i in top_indices]
    acquired_labels = human_annotation(acquired_texts)

    # Update labeled and unlabeled datasets
    labeled_data.extend(acquired_texts)
    labeled_labels.extend(acquired_labels)
    labeled_dataset = TextDataset(labeled_data, labeled_labels, tokenizer)

    # Sort indices in descending order to delete without shifting
    sorted_indices = sorted([subset_indices[i] for i in top_indices], reverse=True)
    for i in sorted_indices:
        del unlabeled_data[i]

    # Validation phase (if validation data exists)
    if len(labeled_data) > 10:  # Ensure enough data for validation
        validation_dataloader = DataLoader(labeled_dataset, batch_size=batch_size)
        val_loss, val_accuracy, val_precision, val_recall, val_f1 = evaluate_model(
            model, validation_dataloader, criterion
        )
        print(
            f"Iteration {iteration} Validation - "
            f"Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}, "
            f"Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}"
        )

# Save the model and tokenizer using save_pretrained()
model_output_dir = "fine_tuned_model"
model.save_pretrained(model_output_dir)
tokenizer.save_pretrained(model_output_dir)
print(f"Model and tokenizer saved to {model_output_dir}")

print("Fine-tuning with active learning completed.")
