import pandas as pd
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def encode(df, tokenizer):
    input_ids = []
    attention_masks = []

    for ix, row in df.iterrows():
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
    labels = torch.tensor(df['marginal_text'].tolist()).long()

    return input_ids, attention_masks, labels

def evaluate(model, loader, device):
    loss, accuracy = 0.0, []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            output = model(input_ids,
                           token_type_ids=None,
                           attention_mask=input_mask,
                           labels=labels)
        loss += output.loss.item()
        preds_batch = torch.argmax(output.logits, axis=1)
        batch_acc = torch.mean((preds_batch == labels).float())
        accuracy.append(batch_acc)

    accuracy = torch.mean(torch.tensor(accuracy))
    return loss, accuracy

def get_predictions(model, loader, device):
    preds = []
    logits = []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
        with torch.no_grad():
            output = model(input_ids, attention_mask=input_mask, labels=labels)
        preds_batch = torch.argmax(output.logits, axis=1)
        logits_batch = output.logits

        preds.extend(preds_batch.tolist())
        logits.extend(logits_batch.tolist())

    return preds, logits

def precision(labels, preds):
    denominator = sum(preds == 1.0)
    if denominator == 0:
        return 0.0
    return sum((labels == 1.0) & (preds == 1.0)) / denominator

def recall(labels, preds):
    denominator = sum(labels == 1.0)
    if denominator == 0:
        return 0.0
    return sum((labels == 1.0) & (preds == 1.0)) / denominator

def F1(pre, rec):
    if pre + rec == 0:
        return 0.0
    return 2 / ((1 / pre) + (1 / rec))

def accuracy(labels, preds):
    return sum(labels == preds) / len(labels)

def get_metrics(labels, preds):
    acc = accuracy(labels, preds)
    pre = precision(labels, preds)
    rec = recall(labels, preds)
    f_1 = F1(pre, rec)
    return acc, pre, rec, f_1

n_epochs = 1  # Increase the number of epochs to allow more learning
batch_size = 16  # Reduce batch size to fit in GPU memory
num_workers = 2  # Reduce number of workers to save memory
learning_rate = 0.00001  # Use a common learning rate for transformers

def main(args):
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'

    id2label = {0: 'other', 1: 'marginal_text', 2: 'merged'}
    label2id = {'other': 0, 'marginal_text': 1, 'merged': 2}

    model_dir = 'KB/bert-base-swedish-cased'

    tok = AutoTokenizer.from_pretrained(model_dir)

    model = AutoModelForSequenceClassification.from_pretrained(model_dir,
                                                               num_labels=3,
                                                               id2label=id2label,
                                                               label2id=label2id).to(device)

    train_data = pd.read_csv(f'{args.data_folder}/train_set.csv')
    val_data = pd.read_csv(f'{args.data_folder}/val_set.csv')
    test_data = pd.read_csv(f'{args.data_folder}/test_set.csv')

    train_input_ids, train_attention_masks, train_labels = encode(train_data, tok)
    train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

    val_input_ids, val_attention_masks, val_labels = encode(val_data, tok)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)

    test_input_ids, test_attention_masks, test_labels = encode(test_data, tok)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    train_loader = DataLoader(train_dataset,
                              shuffle=True,
                              batch_size=batch_size,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers)
    test_loader = DataLoader(test_dataset,
                             shuffle=False,
                             batch_size=batch_size,
                             num_workers=num_workers)

    # Assign class weights to handle class imbalance
    class_weights = torch.tensor([1.0, 5.0, 2.0]).to(device)
    loss_fn = torch.nn.CrossEntropyLoss(weight=class_weights)

    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                  lr=learning_rate)

    num_training_steps = len(train_loader) * n_epochs
    num_warmup_steps = num_training_steps // 10

    scheduler = get_linear_schedule_with_warmup(optimizer=optimizer,
                                                num_warmup_steps=num_warmup_steps,
                                                num_training_steps=num_training_steps)

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    count = 0
    for epoch in range(n_epochs):
        print(f"Start epoch {epoch}!")
        train_loss = 0
        model.train()

        for i, batch in enumerate(tqdm(train_loader, total=len(train_loader))):
            model.zero_grad()

            input_ids = batch[0].to(device)
            input_mask = batch[1].to(device)
            labels = batch[2].to(device)

            output = model(input_ids,
                           token_type_ids=None,
                           attention_mask=input_mask)
            loss = loss_fn(output.logits, labels)
            train_loss += loss.item()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping to stabilize training
            optimizer.step()
            scheduler.step()

        # Evaluation
        val_loss, val_accuracy = evaluate(model, val_loader, device)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            count = 0
        else:
            count += 1
            if count == 5:
                print('Early Stopping!')
                break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} done!")
        print(f"Validation accuracy is {val_accuracy} and val loss is {val_loss}")

    model.load_state_dict(best_model_state_dict)

    train_eval_loader = DataLoader(train_dataset,
                                   shuffle=False,
                                   batch_size=batch_size,
                                   num_workers=num_workers)
    train_preds, train_logits = get_predictions(model, train_eval_loader, device)
    train_preds = pd.Series(train_preds)
    train_labels = train_data['marginal_text']

    val_preds, val_logits = get_predictions(model, val_loader, device)
    val_preds = pd.Series(val_preds)
    val_labels = val_data['marginal_text']

    test_preds, test_logits = get_predictions(model, test_loader, device)
    test_preds = pd.Series(test_preds)
    test_labels = test_data['marginal_text']

    if args.save_predictions:
        train_data['preds'] = train_preds
        val_data['preds'] = val_preds
        test_data['preds'] = test_preds

        train_data.to_csv(f'{args.save_folder}/train_predictions.csv', index=False)
        val_data.to_csv(f'{args.save_folder}/val_predictions.csv', index=False)
        test_data.to_csv(f'{args.save_folder}/test_predictions.csv', index=False)

    print(f'train metrics: \n {get_metrics(train_labels, train_preds)}')
    print(f'val metrics: \n {get_metrics(val_labels, val_preds)}')
    print(f'test metrics: \n {get_metrics(test_labels, test_preds)}')

    model.save_pretrained(f'{args.save_folder}/margin_prediction_model')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type=str, default='./data')
    parser.add_argument("--save_folder", type=str, default="./output")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")
    parser.add_argument("--save_predictions", action="store_true", help="Set this flag to save predictions to csv.")
    args = parser.parse_args()
    main(args)
