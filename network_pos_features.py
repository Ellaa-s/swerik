import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
from torch.utils.data import TensorDataset, DataLoader


def evaluate(model, loader, device):
    loss, accuracy = 0.0, []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        input_ids = batch[0].to(device)
        input_mask = batch[1].to(device)
        labels = batch[2].to(device)
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
        output = model(input_ids,
                       token_type_ids=None,
                       attention_mask=input_mask,
                       labels=labels)
        preds_batch = torch.argmax(output.logits, axis=1)
        logits_batch = output.logits
        preds.extend(preds_batch.tolist())
        logits.extend(logits_batch.tolist())
    
    return preds, logits

def precision(labels, preds):
  return sum((labels == 1.0) & (preds == 1.0)) / sum(preds == 1.0)

def recall(labels, preds):
  return sum((labels == 1.0) & (preds == 1.0)) / sum(labels == 1.0)

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

n_epochs = 1
batch_size = 16
num_workers = 2
learning_rate = 0.00003

"""
Feedforward Neural Network for positional features.
Args:
    input_dim (int): Number of input features (default: 9 positional features).
    hidden_dim (int): Number of units in each hidden layer (default: 180).
    num_hidden_layers (int): Number of hidden layers (default: 4).
"""
class PositionalFFNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=180, num_hidden_layers=4):

        super(PositionalFFNN, self).__init__()
        
        # Input layer (maps 9 → 180)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        
        # Hidden layers (4 layers, 180 → 180)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        
        # GELU activation function
        self.activation = nn.GELU()
    
    """
    Forward pass of the positional feedforward neural network.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim).
    """  
    def forward(self, x):
        # Input layer
        x = self.activation(self.input_layer(x))
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
        
        return x
    

class BERTWithPositionalFeatures(nn.Module):
    def __init__(self, bert_model_name='bert-base-swedish-cased'):
        super(BERTWithPositionalFeatures, self).__init__()
        self.bert = BertModel.from_pretrained(bert_model_name)
        self.positional_ffnn = PositionalFFNN()
        self.batch_norm = nn.BatchNorm1d(948)
        self.classifier = nn.Sequential(
            nn.Linear(948, 512),
            nn.GELU(),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Linear(256, 1)
        )

    def forward(self, text_input, text_attention_mask, positional_features):
        # Textual data through BERT
        bert_output = self.bert(input_ids=text_input, attention_mask=text_attention_mask)
        pooled_output = bert_output.pooler_output  # Shape: (batch_size, 768)

        # Positional data through FFNN
        pos_output = self.positional_ffnn(positional_features)  # Shape: (batch_size, 180)

        # Concatenate BERT and positional features
        combined_output = torch.cat([pooled_output, pos_output], dim=1)  # Shape: (batch_size, 948)

        # Batch normalization + classification
        combined_output = self.batch_norm(combined_output)
        logits = self.classifier(combined_output)

        return logits
    


def main(args):
    
    if args.cuda:
        device = 'cuda'
    else:
        device = 'cpu'
        
    # Load your data
    train_data = pd.read_csv(f'{args.data_folder}/train_set.csv')
    val_data = pd.read_csv(f'{args.data_folder}/val_set.csv')
    test_data = pd.read_csv(f'{args.data_folder}/test_set.csv')

    # Select positional features and labels
    num_features = ['posLeft', 'posUpper', 'posRight', 'posLower', 'page_number',"year", "relative_page_number"]
    boolean_features = ["even_page", "second_chamber", "unicameral"]
    
    # Normalize positional features
    scaler = StandardScaler()
    scaler.fit(train_data[num_features])
    train_data[num_features] = scaler.transform(train_data[num_features])
    val_data[num_features] = scaler.transform(val_data[num_features])
    test_data[num_features] = scaler.transform(test_data[num_features])

    # train_data_combined = np.hstack([train_data_scaled, train_data[boolean_features].values])
    # test_data_combined = np.hstack([test_data_scaled, test_data[boolean_features].values])
    # val_data_combined = np.hstack([val_data_scaled, val_data[boolean_features].values])
    
    # label vectors
    # y_train = train_data['marginal_text']
    # y_test = test_data['marginal_text']
    # y_val = val_data['marginal_text']
    
    train_dataset = TensorDataset(train_data["id"], train_data[num_features + boolean_features], train_data['marginal_text'])
    test_dataset = TensorDataset(test_data["id"], test_data[num_features + boolean_features], test_data['marginal_text'])
    val_dataset = TensorDataset(val_data["id"], val_data[num_features + boolean_features], val_data['marginal_text'])

    train_loader = DataLoader(train_dataset,
                            shuffle = True,
                            batch_size = batch_size,
                            num_workers = num_workers)
    val_loader = DataLoader(val_dataset,
                            shuffle = False,
                            batch_size = batch_size,
                            num_workers = num_workers)
    test_loader = DataLoader(test_dataset,
                            shuffle = False,
                            batch_size = batch_size,
                            num_workers = num_workers)
    # Convert to PyTorch tensors
    # X_train = torch.tensor(train_data_combined, dtype=torch.float32)
    # y_train = torch.tensor(y_train.values, dtype=torch.float32)
    # X_test = torch.tensor(test_data_combined, dtype=torch.float32)
    # y_test = torch.tensor(y_test.values, dtype=torch.float32)
    # X_val = torch.tensor(val_data_combined, dtype=torch.float32)
    # y_val = torch.tensor(y_val.values, dtype=torch.float32)

    model = PositionalFFNN(input_dim=9, hidden_dim=180, num_hidden_layers=4)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                                lr = learning_rate)
    criterion = nn.BCEWithLogitsLoss()  # Loss for binary classification
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    count = 0
    for epoch in range(n_epochs):
        print(f"Start epoch {epoch}!")
        train_loss = 0
        model.train()

        for i, batch in enumerate(tqdm(train_loader, total = len(train_loader))):
            model.zero_grad()

            # Extract the features and labels from the batch
            features, labels = batch  # Assuming `train_loader` gives you features and labels

            # Move data to the device (GPU/CPU)
            features = features.to(device)
            labels = labels.to(device)

            # Forward pass: get predictions from the model
            predictions = model(features)

            # Calculate the loss
            # Assuming MSE for regression or CrossEntropy for classification
            loss = criterion(predictions, labels)
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

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
    
    # use epoch with lowest validation loss
    model.load_state_dict(best_model_state_dict) 
    
    train_eval_loader = DataLoader(train_dataset,
                          shuffle = False,
                          batch_size = batch_size,
                          num_workers = num_workers)
    train_preds, train_logits = get_predictions(model, train_eval_loader, device)
    train_preds = pd.Series(train_preds)
    train_logits1 = pd.Series([x[0] for x in train_logits])
    train_logits2 = pd.Series([x[1] for x in train_logits])
    train_labels = train_data['title']
    
    val_preds, val_logits = get_predictions(model, val_loader, device)
    val_preds = pd.Series(val_preds)
    val_logits1 = pd.Series([x[0] for x in val_logits])
    val_logits2 = pd.Series([x[1] for x in val_logits])
    val_labels = val_data['title']
    
    test_preds, test_logits = get_predictions(model, test_loader, device)
    test_preds = pd.Series(test_preds)
    test_logits1 = pd.Series([x[0] for x in test_logits])
    test_logits2 = pd.Series([x[1] for x in test_logits])
    test_labels = test_data['title']
    
    if args.save_predictions:
        train_data['preds'] = train_preds
        val_data['preds'] = val_preds
        test_data['preds'] = test_preds

        train_data['logits1'] = train_logits1
        val_data['logits1'] = val_logits1
        test_data['logits1'] = test_logits1
        train_data['logits2'] = train_logits2
        val_data['logits2'] = val_logits2
        test_data['logits2'] = test_logits2
        
        train_data.to_csv(f'{args.save_folder}/train_predictions.csv', index=False) 
        val_data.to_csv(f'{args.save_folder}/val_predictions.csv', index=False) 
        test_data.to_csv(f'{args.save_folder}/test_predictions.csv', index=False) 
    
    print(f'train metrics: \n {get_metrics(train_labels, train_preds)}')
    print(f'val metrics: \n {get_metrics(val_labels, val_preds)}')
    print(f'test metrics: \n {get_metrics(test_labels, test_preds)}')
    
    # save model locally
    model.save_pretrained(f'{args.save_folder}/title_prediction_model')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type = str, default = './swerik/data')
    parser.add_argument("--save_folder", type = str, default = "./swerik/model")
    parser.add_argument("--save_predictions", action="store_true", help="Set this flag to save predictions to csv.")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")

    args = parser.parse_args()
    main(args)