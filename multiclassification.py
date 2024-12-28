import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from transformers import EarlyStoppingCallback
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report
from tqdm import tqdm
import argparse

# Ensure Tokenizers Parallelism is set to False to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Set model directory and load the tokenizer
model_dir = 'KB/bert-base-swedish-cased'
tokenizer = AutoTokenizer.from_pretrained(model_dir)

# Load datasets
train_data = pd.read_csv('./data/multiclassification_data_set/train_data_multi.csv')
val_data = pd.read_csv('./data/multiclassification_data_set/val_set_multi.csv')
test_data = pd.read_csv('./data/multiclassification_data_set/test_set_multi.csv')

# Function to encode dataset
def encode(df, tokenizer, max_length=125):  # Reduced max_length to 64 to reduce memory usage
    input_ids = []
    attention_masks = []

    for _, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
            row['text_line'],
            add_special_tokens=True,
            max_length=max_length,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    # Concatenate tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['final_target'].tolist(), dtype=torch.long)

    return input_ids, attention_masks, labels

# Encode train, validation, and test datasets
train_input_ids, train_attention_masks, train_labels = encode(train_data, tokenizer)
val_input_ids, val_attention_masks, val_labels = encode(val_data, tokenizer)
test_input_ids, test_attention_masks, test_labels = encode(test_data, tokenizer)

# Create TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# Create DataLoaders
batch_size = 8  # Reduced batch size to minimize memory usage
num_workers = 2  # Set to 0 if memory is very limited

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

# Model loading and other parameters
n_epochs = 10  # Number of epochs to train the model
learning_rate = 1e-4  # Learning rate for the optimizer

def freeze_bert_layers(model):
    # Freeze all layers except the last layer to reduce resource usage
    for name, param in model.named_parameters():
        if 'encoder.layer.8' in name or 'encoder.layer.9' in name or 'encoder.layer.10' in name or 'encoder.layer.11' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def evaluate(model, loader, device):
    total_loss, accuracy = 0.0, []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=input_mask, labels=labels)
        total_loss += output.loss.item()
        preds_batch = torch.argmax(output.logits, axis=1)
        batch_acc = torch.mean((preds_batch == labels).float())
        accuracy.append(batch_acc.item())

    avg_loss = total_loss / len(loader)
    avg_accuracy = sum(accuracy) / len(accuracy)

    return avg_loss, avg_accuracy

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    # Define label mapping
    id2label = {0: 'other', 1: 'marginal_text', 2: 'merged'}
    label2id = {'other': 0, 'marginal_text': 1, 'merged': 2}

    # Load model
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    # Freeze some layers to reduce memory usage
    freeze_bert_layers(model)

    # Assign class weights to handle class imbalance
    class_weights = torch.tensor([1.0, 2.0, 5.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights , label_smoothing=0.2)

    # Set up optimizer and scheduler
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    num_training_steps = len(train_loader) * n_epochs
    num_warmup_steps = num_training_steps // 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )


    # Training Loop
    best_val_loss = float('inf')
    best_model_state_dict = None
    patience = 5
    count = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        correct_predictions = 0
        total_samples = 0

        for batch in tqdm(train_loader, total=len(train_loader)):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

        # Forward pass
            outputs = model(input_ids, attention_mask=input_mask)
            logits = outputs.logits

        # Compute the loss
            loss = loss_fn(logits, labels)
            train_loss += loss.item()

        # Backward pass and optimization step
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()  # Reset gradients after each step

        # Compute accuracy for the batch
            predictions = torch.argmax(logits, dim=1)
            correct_predictions += (predictions == labels).sum().item()
            total_samples += labels.size(0)

    # Calculate and print average train accuracy and loss for the epoch
        train_accuracy = correct_predictions / total_samples
        avg_train_loss = train_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}")

    def get_predictions(model, loader, device):
        model.eval()
        predictions = []
        true_labels = []

        for batch in tqdm(loader, total=len(loader)):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=input_mask)
                logits = outputs.logits

            preds_batch = torch.argmax(logits, axis=1)
            predictions.extend(preds_batch.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

        return predictions, true_labels

# Evaluate on the test set and calculate metrics
    test_predictions, test_labels = get_predictions(model, test_loader, device)

    # Calculate precision, recall, and F1-score for each class
    precision = precision_score(test_labels, test_predictions, average=None)
    recall = recall_score(test_labels, test_predictions, average=None)
    f1 = f1_score(test_labels, test_predictions, average=None)

    print("Precision per class:", precision)
    print("Recall per class:", recall)
    print("F1-score per class:", f1)

# Print a full classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, test_predictions, target_names=['other', 'marginal_text', 'merged']))

    # Load the best model
    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)

    # Evaluate on the test set
    test_loss, test_accuracy = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

    # Save the final model
    if args.save_folder:
        model.save_pretrained(f'{args.save_folder}/margin_prediction_model')
        tokenizer.save_pretrained(f'{args.save_folder}/margin_prediction_model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for multi-class text classification")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with CUDA.")
    parser.add_argument("--save_folder", type=str, default="./output", help="Directory to save the trained model.")
    args = parser.parse_args()
    main(args)
