import torch
import torch.nn as nn
from transformers import BertModel, AutoConfig, AutoTokenizer, AutoModel
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
from network_pos_features import PositionalFFNN, create_tensordataset
from safetensors.torch import load_file
import os
from safetensors.torch import safe_open


def encode(df, tokenizer):
    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for ix, row in df.iterrows():
        encoded_dict = tokenizer.encode_plus(
                            row['text_line'],
                            add_special_tokens = True,
                            max_length = 128,
                            truncation=True,
                            padding = 'max_length',
                            return_attention_mask = True,
                            return_tensors = 'pt',
                       )

        # Add the encoded sentence to the list.
        input_ids.append(encoded_dict['input_ids'])

        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(df['marginal_text'].tolist()).float()

    return input_ids, attention_masks, labels
# Check if evaluation is correct
def evaluate(model, pos_loader,bert_loader, device,pos_weight):
    loss, accuracy = 0.0, []
    model.eval()
    for (pos_batch, bert_batch) in tqdm(zip(pos_loader, bert_loader), total=min(len(pos_loader), len(bert_loader))):
        pos_features = pos_batch[0].to(device)
        pos_labels = pos_batch[1].to(device)
        input_ids = bert_batch[0].to(device)
        input_mask = bert_batch[1].to(device)
        with torch.no_grad():
            output = model(input_ids,input_mask,pos_features)
        loss_function = torch.nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight]))        
        loss += loss_function(output, pos_labels)
        # preds_batch = torch.argmax(output.logits, axis=1)
        # batch_acc = torch.mean((preds_batch == bert_labels).float())
        # accuracy.append(batch_acc)
        preds_batch = (torch.sigmoid(output) > 0.5).long()  # Converts probabilities to labels (0 or 1)
        batch_acc = torch.mean((preds_batch == pos_labels).float())
        accuracy.append(batch_acc)

    accuracy = torch.mean(torch.tensor(accuracy))
    return loss, accuracy

def get_predictions(model, pos_loader, bert_loader, device):
    preds = []
    logits = []
    model.eval()
    for (pos_batch, bert_batch) in tqdm(zip(pos_loader, bert_loader), total=min(len(pos_loader), len(bert_loader))):
        pos_features = pos_batch[0].to(device)
        input_ids = bert_batch[0].to(device)
        input_mask = bert_batch[1].to(device)
        labels = bert_batch[2].to(device)
        with torch.no_grad():
            output = model(input_ids,input_mask,pos_features)
        #preds_batch = torch.argmax(output.logits, axis=1)
        preds_batch = (torch.sigmoid(output) > 0.5).long()
        #logits_batch = output.logits
        preds.extend(preds_batch.tolist())
        logits.extend(output.tolist())
    
    return preds, logits

def precision(labels, preds):
  return np.sum((labels == 1.0) & (preds == 1.0)) / np.sum(preds == 1.0)

def recall(labels, preds):
  return np.sum((labels == 1.0) & (preds == 1.0)) / np.sum(labels == 1.0)

def F1(pre, rec):
  return 2/((1/pre)+(1/rec))

def accuracy(labels, preds):
  return sum(labels == preds) / len(labels)

def get_metrics(labels, preds):
  acc = accuracy(labels, preds)
  pre = precision(labels, preds)
  rec = recall(labels, preds)
  f_1 = F1(pre, rec)
  return acc, pre, rec, f_1

n_epochs = 30
batch_size = 16
num_workers = 2
learning_rate = 0.00003

class BERTWithPositionalFeatures(nn.Module):
    def __init__(self):
        super().__init__()
        # to use basic Bert:
        # bert_model_name='KB/bert-base-swedish-cased'
        
        # Load pretrained
        bert_model_name = "./swerik/output/output/margin_prediction_model/"
        self.bert = BertModel.from_pretrained(bert_model_name)
        
        # # Set BERT layers to be trainable (all parameters by default)
        # for param in self.bert.parameters():
        #     param.requires_grad = True  # Enable gradient updates for all BERT layers
        
        # Pretrained PositionalFFNN
        self.positional_ffnn = PositionalFFNN()
        self.positional_ffnn.load_state_dict(torch.load(f'{args.model_folder}/positional_ffnn.pt',weights_only=True))
        self.positional_ffnn.eval()    # if the positional_ffnn should not be trained
        
        # Train positional model in here
        # self.input_layer = nn.Linear(9, 180)
        # self.hidden_layers = nn.Sequential(
        #     nn.Linear(180, 180),
        #     nn.GELU(),
        #     nn.Linear(180, 180),
        #     nn.GELU(),
        #     nn.Linear(180, 180),
        #     nn.GELU(),
        #     nn.Linear(180, 180),
        #     nn.GELU()
        # )
        
        self.batch_norm = nn.BatchNorm1d(948)
        self.network = nn.Sequential(
            nn.Linear(948, 948),
            nn.GELU(),
            nn.Linear(948, 948),
            nn.GELU(),
        )
        self.dropout = nn.Dropout(p=0.9)
        self.classifier = nn.Linear(948, 1)
    def forward(self, text_input, text_attention_mask, positional_features):
        # Textual data through BERT
        bert_output = self.bert(text_input, text_attention_mask).pooler_output # Shape: (batch_size, 768)

        # Positional data through FFNN without output layer
        with torch.no_grad():
            pos_output = self.positional_ffnn.extract_features(positional_features)  # Shape: (batch_size, 180)

        # Train Positional FFNN in here:
        # pos_output = self.input_layer(positional_features)
        # pos_output = self.hidden_layers(pos_output)
        
        # Concatenate BERT and positional features
        combined_output = torch.cat([bert_output, pos_output], dim=1)  # Shape: (batch_size, 948)
        # Batch normalization + classification
        normalized_comb_output = self.batch_norm(combined_output)
        x = self.network(normalized_comb_output)
        x = self.dropout(x)
        x = self.classifier(x)
        return x.squeeze(1)

def main(args):
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
    pos_features = ['posLeft', 'posUpper', 'posRight', 'posLower', "year", "relative_page_number","even_page", "second_chamber", "unicameral"]

    # Load your data
    train_data = pd.read_csv(f'{args.data_folder}/train_pos_set.csv')
    val_data = pd.read_csv(f'{args.data_folder}/val_pos_set.csv')
    test_data = pd.read_csv(f'{args.data_folder}/test_pos_set.csv')
    
    # Drop if nan values in positional features
    train_data = train_data.dropna(subset=pos_features,ignore_index=True)
    test_data = test_data.dropna(subset=pos_features,ignore_index=True)
    val_data = val_data.dropna(subset=pos_features,ignore_index=True)
    
    # Small set just to test if the script is running or not
    # train_data=train_data.iloc[:100,:]
    # test_data=train_data.iloc[:15,:]
    # val_data=val_data.iloc[:15,:]
  
    # Prepare positional datasets for PositionalFFNN
    train_pos_dataset = create_tensordataset(train_data)
    test_pos_dataset = create_tensordataset(test_data)
    val_pos_dataset = create_tensordataset(val_data)
    
    train_pos_loader = DataLoader(train_pos_dataset,
                            shuffle = True,
                            batch_size = batch_size,
                            num_workers = num_workers)
    val_pos_loader = DataLoader(val_pos_dataset,
                            shuffle = False,
                            batch_size = batch_size,
                            num_workers = num_workers)
    test_pos_loader = DataLoader(test_pos_dataset,
                            shuffle = False,
                            batch_size = batch_size,
                            num_workers = num_workers)

    # Prepare data for BERT input
    model_dir = 'KB/bert-base-swedish-cased'
    tok = AutoTokenizer.from_pretrained(model_dir)
    train_input_ids, train_attention_masks, train_labels = encode(train_data, tok)
    train_bert_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

    val_input_ids, val_attention_masks, val_labels = encode(val_data, tok)
    val_bert_dataset = TensorDataset(val_input_ids, val_attention_masks, val_labels)
    
    test_input_ids, test_attention_masks, test_labels = encode(test_data, tok)
    test_bert_dataset = TensorDataset(test_input_ids, test_attention_masks, test_labels)

    train_bert_loader = DataLoader(train_bert_dataset,
                            shuffle = True,
                            batch_size = batch_size,
                            num_workers = num_workers)
    val_bert_loader = DataLoader(val_bert_dataset,
                            shuffle = False,
                            batch_size = batch_size,
                            num_workers = num_workers)
    test_bert_loader = DataLoader(test_bert_dataset,
                            shuffle = False,
                            batch_size = batch_size,
                            num_workers = num_workers)

    model = BERTWithPositionalFeatures()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr = learning_rate)
    pos_weight = (np.sum(train_data["marginal_text"] == 0) / np.sum(train_data["marginal_text"] == 1))
    pos_weight = 1
    print("weight", pos_weight)
    criterion = nn.BCEWithLogitsLoss(pos_weight= torch.tensor([pos_weight])) # Loss for binary classification
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    count = 0
    for epoch in range(n_epochs):
        print(f"Start epoch {epoch}!")
        train_loss = 0
        model.train()

        for (pos_batch, bert_batch) in tqdm(zip(train_pos_loader, train_bert_loader), total=min(len(train_pos_loader), len(train_bert_loader))):
            model.zero_grad()
            
            # Extract the features and labels from the batch
            pos_features, pos_labels = pos_batch
            pos_features = pos_features.to(device)
            pos_labels = pos_labels.to(device)
            input_ids = bert_batch[0].to(device)
            input_mask = bert_batch[1].to(device)

            # Forward pass: get predictions from the model
            predictions = model(input_ids, text_attention_mask = input_mask, positional_features = pos_features)

            # Calculate the loss
            loss = criterion(predictions.squeeze(), pos_labels.float())
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Evaluation
        val_loss, val_accuracy = evaluate(model, val_pos_loader,val_bert_loader, device,pos_weight)
            
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            count = 0
        else:
            count += 1
            if count == 3:
                print('Early Stopping!')
                break

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch} done!")
        print(f"Validation accuracy is {val_accuracy} and val loss is {val_loss}")
    
    # use epoch with lowest validation loss
    model.load_state_dict(best_model_state_dict) 
    train_eval_pos_loader = DataLoader(train_pos_dataset,
                            shuffle = False,
                            batch_size = batch_size,
                            num_workers = num_workers)
    train_eval_bert_loader = DataLoader(train_bert_dataset,
                        shuffle = False,
                        batch_size = batch_size,
                        num_workers = num_workers)
    train_preds, train_logits = get_predictions(model, train_eval_pos_loader, train_eval_bert_loader, device)
    train_preds = pd.Series(train_preds).reset_index(drop=True)
    train_labels = train_data['marginal_text'].astype(int).reset_index(drop=True)
    
    val_preds, val_logits = get_predictions(model, val_pos_loader, val_bert_loader, device)
    val_preds = pd.Series(val_preds).reset_index(drop=True)
    val_labels = val_data['marginal_text'].astype(int).reset_index(drop=True)
    
    test_preds, test_logits = get_predictions(model, test_pos_loader, test_bert_loader, device)
    test_preds = pd.Series(test_preds).reset_index(drop=True)
    test_labels = test_data['marginal_text'].astype(int).reset_index(drop=True)
    
    if args.save_predictions:
        train_data['preds'] = train_preds
        val_data['preds'] = val_preds
        test_data['preds'] = test_preds

        train_data['logits'] = train_logits
        val_data['logits'] = val_logits
        test_data['logits'] = test_logits
        
        train_data.to_csv(f'{args.save_folder}/train_predictions.csv', index=False) 
        val_data.to_csv(f'{args.save_folder}/val_predictions.csv', index=False) 
        test_data.to_csv(f'{args.save_folder}/test_predictions.csv', index=False) 

    print(f'train metrics: \n {get_metrics(train_labels, train_preds)}')
    print(f'val metrics: \n {get_metrics(val_labels, val_preds)}')
    print(f'test metrics: \n {get_metrics(test_labels, test_preds)}')
    
    # save model locally
    torch.save(model.state_dict(),f'{args.save_folder}/Bert_with_positional_ffnn.pt')
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type = str, default = './swerik/data')
    parser.add_argument("--save_folder", type = str, default = "./swerik/model")
    parser.add_argument("--model_folder", type = str, default = "./swerik/network_results")
    parser.add_argument("--save_predictions", action="store_true", help="Set this flag to save predictions to csv.")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")

    args = parser.parse_args()
    main(args)