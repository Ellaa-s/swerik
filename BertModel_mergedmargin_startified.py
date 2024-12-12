import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import torch.nn.functional as F
import argparse

def encode(df, tokenizer):
    input_ids, attention_masks = [], []

    for _, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
            row['text_line'],
            add_special_tokens=True,
            max_length=128,
            truncation=True,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt',
        )
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])

    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['merged'].tolist()).long()

    return input_ids, attention_masks, labels

def evaluate(model, loader, device, class_weights_tensor):
    loss, accuracy = 0.0, []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=input_mask, labels=labels)
        batch_loss = F.cross_entropy(output.logits, labels, weight=class_weights_tensor)
        loss += batch_loss.item()
        preds_batch = torch.argmax(output.logits, axis=1)
        batch_acc = torch.mean((preds_batch == labels).float())
        accuracy.append(batch_acc)

    accuracy = torch.mean(torch.tensor(accuracy))
    return loss / len(loader), accuracy.item()

def get_predictions_with_threshold(model, loader, device, threshold=0.5):
    preds, logits = [], []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=input_mask)
        logits_batch = output.logits
        preds_batch = (F.softmax(logits_batch, dim=1)[:, 1] > threshold).long()
        preds.extend(preds_batch.tolist())
        logits.extend(logits_batch.tolist())

    return preds, logits

def precision(labels, preds):
    tp = sum((labels == 1) & (preds == 1))
    fp = sum(preds == 1) - tp
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0

def recall(labels, preds):
    tp = sum((labels == 1) & (preds == 1))
    fn = sum(labels == 1) - tp
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0

def F1(pre, rec):
    return 2 * (pre * rec) / (pre + rec) if (pre + rec) > 0 else 0.0

def accuracy(labels, preds):
    return sum(labels == preds) / len(labels)

def get_metrics(labels, preds):
    labels, preds = torch.tensor(labels), torch.tensor(preds)
    acc = accuracy(labels, preds)
    pre = precision(labels, preds)
    rec = recall(labels, preds)
    f_1 = F1(pre, rec)
    return acc, pre, rec, f_1

def tune_threshold(model, loader, device, labels):
    best_threshold, best_f1 = 0.5, 0.0
    thresholds = [x / 100 for x in range(10, 90, 5)]

    for threshold in thresholds:
        preds, _ = get_predictions_with_threshold(model, loader, device, threshold)
        pre = precision(labels, preds)
        rec = recall(labels, preds)
        f1 = F1(pre, rec)

        if f1 > best_f1:
            best_f1, best_threshold = f1, threshold

    print(f"Best Threshold: {best_threshold} with F1-score: {best_f1:.4f}")
    return best_threshold

def main(args):
    device = 'cuda' if args.cuda else 'cpu'
    model_dir = 'KB/bert-base-swedish-cased'

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=2).to(device)

    # Load datasets
    train_data = pd.read_csv(f'{args.data_folder}/train_merged_stratified.csv')
    val_data = pd.read_csv(f'{args.data_folder}/val_merged_stratified.csv')
    test_data = pd.read_csv(f'{args.data_folder}/test_merged_stratified.csv')

    train_input_ids, train_attention_masks, train_labels = encode(train_data, tokenizer)
    val_input_ids, val_attention_masks, val_labels = encode(val_data, tokenizer)
    test_input_ids, test_attention_masks, test_labels = encode(test_data, tokenizer)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=16)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=16)
    test_loader = DataLoader(test_dataset, shuffle=False, batch_size=16)

    class_weights = train_data['merged'].value_counts()
    class_weights_tensor = torch.tensor([1.0 / class_weights[0], 1.0 / class_weights[1]], dtype=torch.float32).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=5 * len(train_loader))

    for epoch in range(5):
        print(f"Epoch {epoch + 1}/5")
        model.train()
        for batch in tqdm(train_loader, total=len(train_loader)):
            input_ids = batch[0].to(device)
            attention_masks = batch[1].to(device)
            labels = batch[2].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_masks, labels=labels)
            loss = F.cross_entropy(outputs.logits, labels, weight=class_weights_tensor)
            loss.backward()
            optimizer.step()
            scheduler.step()

    best_threshold = tune_threshold(model, val_loader, device, val_data['merged'])

    for dataset_name, loader, labels in zip(['Train', 'Validation', 'Test'], [train_loader, val_loader, test_loader], [train_data['merged'], val_data['merged'], test_data['merged']]):
        preds, _ = get_predictions_with_threshold(model, loader, device, best_threshold)
        acc, pre, rec, f1 = get_metrics(labels, preds)
        print(f"{dataset_name} Metrics - Accuracy: {acc:.4f}, Precision: {pre:.4f}, Recall: {rec:.4f}, F1-score: {f1:.4f}")

    if args.save_predictions:
        train_data['preds'] = get_predictions_with_threshold(model, train_loader, device, best_threshold)[0]
        val_data['preds'] = get_predictions_with_threshold(model, val_loader, device, best_threshold)[0]
        test_data['preds'] = get_predictions_with_threshold(model, test_loader, device, best_threshold)[0]

        train_data.to_csv(f'{args.save_folder}/train_predictions.csv', index=False)
        val_data.to_csv(f'{args.save_folder}/val_predictions.csv', index=False)
        test_data.to_csv(f'{args.save_folder}/test_predictions.csv', index=False)

    model.save_pretrained(f'{args.save_folder}/merged_margin_prediction_model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type=str, default='./data_stratified_sampling')
    parser.add_argument("--save_folder", type=str, default="./output")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with CUDA.")
    parser.add_argument("--save_predictions", action="store_true", help="Set this flag to save predictions to csv.")
    parser.add_argument("--patience", type=int, default=2, help="Number of epochs to wait for improvement before early stopping.")
    args = parser.parse_args()
    main(args)