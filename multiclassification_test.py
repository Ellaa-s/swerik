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
train_data = pd.read_csv('multiclassification_data_set/train_data_multi.csv')
val_data = pd.read_csv('multiclassification_data_set/val_set_multi.csv')
test_data = pd.read_csv('multiclassification_data_set/test_set_multi.csv')

# Function to encode dataset
def encode(df, tokenizer, max_length=125):
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

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['final_target'].tolist(), dtype=torch.long)

    return input_ids, attention_masks, labels

# Encode datasets
train_input_ids, train_attention_masks, train_labels = encode(train_data, tokenizer)
val_input_ids, val_attention_masks, val_labels = encode(val_data, tokenizer)
test_input_ids, test_attention_masks, test_labels = encode(test_data, tokenizer)

# Create TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

# Create DataLoaders
batch_size = 8
num_workers = 2

train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
val_loader = DataLoader(val_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)
test_loader = DataLoader(test_dataset, shuffle=True, batch_size=batch_size, num_workers=num_workers)

# Model loading and other parameters
n_epochs = 10
learning_rate = 1e-4

def freeze_bert_layers(model):
    for name, param in model.named_parameters():
        if 'encoder.layer.8' in name or 'encoder.layer.9' in name or 'encoder.layer.10' in name or 'encoder.layer.11' in name or 'classifier' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False

def evaluate(model, loader, device):
    total_loss = 0.0
    all_labels = []
    all_preds = []
    model.eval()

    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=input_mask, labels=labels)
        total_loss += output.loss.item()
        preds_batch = torch.argmax(output.logits, axis=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds_batch.cpu().numpy())

    avg_loss = total_loss / len(loader)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')

    return avg_loss, precision, recall, f1

def main(args):
    device = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'

    id2label = {0: 'other', 1: 'marginal_text', 2: 'merged'}
    label2id = {'other': 0, 'marginal_text': 1, 'merged': 2}

    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        num_labels=3,
        id2label=id2label,
        label2id=label2id
    ).to(device)

    freeze_bert_layers(model)

    class_weights = torch.tensor([1.0, 2.0, 5.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.2)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    num_training_steps = len(train_loader) * n_epochs
    num_warmup_steps = num_training_steps // 10

    scheduler = get_linear_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )

    best_val_loss = float('inf')
    best_model_state_dict = None
    patience = 5
    count = 0

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0
        all_labels = []
        all_preds = []

        for batch in tqdm(train_loader, total=len(train_loader)):
            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            outputs = model(input_ids, attention_mask=input_mask)
            logits = outputs.logits

            loss = loss_fn(logits, labels)
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            model.zero_grad()

            predictions = torch.argmax(logits, dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predictions.cpu().numpy())

        avg_train_loss = train_loss / len(train_loader)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        print(f"Epoch {epoch + 1}/{n_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")

        val_loss, val_precision, val_recall, val_f1 = evaluate(model, val_loader, device)
        print(f"Validation Loss: {val_loss:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}, F1: {val_f1:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            count = 0
        else:
            count += 1
            if count >= patience:
                print("Early stopping triggered.")
                break

    if best_model_state_dict:
        model.load_state_dict(best_model_state_dict)

    test_loss, test_precision, test_recall, test_f1 = evaluate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.4f}, Precision: {test_precision:.4f}, Recall: {test_recall:.4f}, F1: {test_f1:.4f}")

    if args.save_folder:
        model.save_pretrained(f'{args.save_folder}/margin_prediction_model')
        tokenizer.save_pretrained(f'{args.save_folder}/margin_prediction_model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a BERT model for multi-class text classification")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with CUDA.")
    parser.add_argument("--save_folder", type=str, default="./output", help="Directory to save the trained model.")
    args = parser.parse_args()
    main(args)

