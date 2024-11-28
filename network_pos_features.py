import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Check if evaluation is correct
def evaluate(model, loader, device):
    total_loss, accuracy = 0.0, []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        with torch.no_grad():
            output = model(inputs)
        # Check for NaN or Inf in the model output
        # if torch.isnan(output).any() or torch.isinf(output).any():
        #     print("NaN or Inf detected in model output!")
        #     print(f"Output: {output}")
        #     continue
        # Compute loss
        loss_fn = torch.nn.BCEWithLogitsLoss()  
        labels = labels.float()  
        loss = loss_fn(output, labels)
        # Check if loss is NaN
        # if torch.isnan(loss).any() or torch.isinf(loss).any():
        #     print("NaN or Inf loss detected")
        #     print(f"Output: {output}")
        #     print(f"Labels: {labels}")
        #     continue
        total_loss += loss.item()
        # Compute predictions
        #preds_batch = torch.argmax(output, axis=1)
        preds_batch = (output > 0.5).long().squeeze()  # Converts probabilities to labels (0 or 1)
        batch_acc = torch.mean((preds_batch == labels).float())
        accuracy.append(batch_acc)

    accuracy = torch.mean(torch.tensor(accuracy))
    return total_loss, accuracy

def get_predictions(model, loader, device):
    preds = []
    logits = []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        inputs = batch[0].to(device)
        #labels = batch[1].to(device)
        # Forward pass
        with torch.no_grad():  # Disable gradient computation for evaluation
            output = model(inputs)
        print(f"output: {output}")
        preds_batch = (output > 0.0).long().squeeze()
        #preds_batch = torch.argmax(output, axis=1)
        print(f"pres_batch {preds_batch}")
        #preds.extend(preds_batch.tolist())
        preds.append(preds_batch)
        logits.extend(output.tolist())
    
    return preds, logits

def precision(labels, preds):
  return np.sum((labels == 1.0) & (preds == 1.0)) / np.sum(preds == 1.0)

def recall(labels, preds):
  return np.sum((labels == 1.0) & (preds == 1.0)) / np.sum(labels == 1.0)

def F1(pre, rec):
  return 2/((1/pre)+(1/rec))

def accuracy(labels, preds):
  return np.sum(labels == preds) / len(labels)

def get_metrics(labels, preds):
  acc = accuracy(labels, preds)
  pre = precision(labels, preds)
  rec = recall(labels, preds)
  f_1 = F1(pre, rec)
  return acc, pre, rec, f_1

n_epochs = 1
batch_size = 8
num_workers = 2
learning_rate = 0.00000003

"""
Feedforward Neural Network for positional features.
Args:
    input_dim (int): Number of input features (default: 9 positional features).
    hidden_dim (int): Number of units in each hidden layer (default: 180).
    num_hidden_layers (int): Number of hidden layers (default: 4).
"""
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(9, 180),
            nn.ReLU(),
            nn.Linear(180, 180),
            nn.ReLU(),
            nn.Linear(180, 1),
        )

    def forward(self, x):
        #print(f"before input layer: {x}")
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits.squeeze(1)
class PositionalFFNN(nn.Module):
    def __init__(self, input_dim=9, hidden_dim=180, num_hidden_layers=4):

        super(PositionalFFNN, self).__init__()
        
        # Input layer (maps 9 → 180)
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        nn.init.xavier_normal_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        
        # Hidden layers (4 layers, 180 → 180)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim) for _ in range(num_hidden_layers)]
        )
        
        for layer in self.hidden_layers:
            nn.init.xavier_normal_(layer.weight)
            nn.init.zeros_(layer.bias)
            
        # GELU activation function
        self.activation = nn.GELU()

        self.output_layer = nn.Linear(hidden_dim, 1) 
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    """
    Forward pass of the positional feedforward neural network.
    Args:
        x (torch.Tensor): Input tensor of shape (batch_size, input_dim).
    Returns:
        torch.Tensor: Output tensor of shape (batch_size, hidden_dim).
    """  
    def forward(self, x):
        if torch.isnan(x).any():
            print("NaN detected before the first layer!")
        # Input layer
        x = self.input_layer(x)
        
        # Hidden layers
        for layer in self.hidden_layers:
            x = self.activation(layer(x))
            if torch.isnan(x).any():
                print(f"NaN detected after layer{layer}!")
        
        # Output layer for training
        x = self.output_layer(x)
        if torch.isnan(x).any():
            print("NaN detected at output layer!")
        return x

def create_tensordataset(dataset):
    # Encode id column
    #label_encoder = LabelEncoder()
    #encoded_ids = label_encoder.fit_transform(dataset["id"])
    
    # Select positional features and labels
    num_features = ['posLeft', 'posUpper', 'posRight', 'posLower', "year", "relative_page_number"]
    boolean_features = ["even_page", "second_chamber", "unicameral"]
    
    # Normalize positional features
    scaler = StandardScaler()
    scaler.fit(dataset[num_features])
    dataset[num_features] = scaler.transform(dataset[num_features])
    
    # Convert to tensors
    #id_tensor = torch.tensor(encoded_ids, dtype=torch.long) 
    numerical_tensor = torch.tensor(dataset[num_features].values, dtype=torch.float32)
    boolean_tensor = torch.tensor(dataset[boolean_features].values, dtype=torch.bool) 
    labels_tensor = torch.tensor(dataset["marginal_text"], dtype=torch.long) 
    
    # Concatenate the tensors into a single input tensor (shape=(N, 1 + num_features + boolean_features))
    input_tensor = torch.cat(( numerical_tensor, boolean_tensor), dim=1)  #id_tensor.unsqueeze(1),

    # Create the TensorDatasets
    return TensorDataset(input_tensor, labels_tensor)

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
    
    train_data = train_data.dropna(subset=pos_features)
    test_data = test_data.dropna(subset=pos_features)
    val_data = val_data.dropna(subset=pos_features)

    # Small set just to test if the script is running or not
    train_data=train_data.iloc[:100,:]
    test_data=train_data.iloc[:15,:]
    val_data=val_data.iloc[:15,:]
            
        
    train_dataset = create_tensordataset(train_data)
    test_dataset = create_tensordataset(test_data)
    val_dataset = create_tensordataset(val_data)
    #print(f"first row of trian data set after transforming it into tensors: {train_dataset[0]}")
    
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
    model = NeuralNetwork()
    #model = PositionalFFNN(input_dim=9, hidden_dim=180, num_hidden_layers=4)
    #optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr = learning_rate)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    #best_model_state_dict = {}
    best_model_state_dict = model.state_dict()

    count = 0
    for epoch in range(n_epochs):
        print(f"Start epoch {epoch}!")
        train_loss = 0
        model.train()

        for i, batch in enumerate(tqdm(train_loader, total = len(train_loader))):
            optimizer.zero_grad()

            # Extract the features and labels from the batch
            features, labels = batch  # Assuming `train_loader` gives you features and labels

            
            # Move data to the device (GPU/CPU)
            model = model.to(device)
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass: get predictions from the model
            predictions = model(features)
            print(f"predictions that come out of the model {predictions}")
            # if torch.isnan(predictions).any():
            #     print("Predictions contain NaN!")
            # Calculate the loss
            # Assuming MSE for regression or CrossEntropy for classification
            loss = criterion(predictions.squeeze(), labels.float())

            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            # for param in model.parameters():
            #     if torch.isnan(param.grad).any():
            #         print("NaN in gradients!")
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
    #print(f"train logits: {train_logits}")
    train_preds = pd.Series(train_preds)
    train_preds = [x.item() for sublist in train_preds for x in sublist]
    # train_logits1 = pd.Series([x[0] for x in train_logits])
    # train_logits2 = pd.Series([x[1] for x in train_logits])
    train_labels = train_data['marginal_text']
    
    val_preds, val_logits = get_predictions(model, val_loader, device)
    val_preds = pd.Series(val_preds)
    val_preds = [x.item() for sublist in val_preds for x in sublist]
    # val_logits1 = pd.Series([x[0] for x in val_logits])
    # val_logits2 = pd.Series([x[1] for x in val_logits])
    val_labels = val_data['marginal_text']
    
    test_preds, test_logits = get_predictions(model, test_loader, device)
    test_preds = pd.Series(test_preds)
    test_preds = [x.item() for sublist in test_preds for x in sublist]
    # test_logits1 = pd.Series([x[0] for x in test_logits])
    # test_logits2 = pd.Series([x[1] for x in test_logits])
    test_labels = test_data['marginal_text']
    
    if args.save_predictions:
        train_data['preds'] = train_preds
        val_data['preds'] = val_preds
        test_data['preds'] = test_preds

        # train_data['logits1'] = train_logits1
        # val_data['logits1'] = val_logits1
        # test_data['logits1'] = test_logits1
        # train_data['logits2'] = train_logits2
        # val_data['logits2'] = val_logits2
        # test_data['logits2'] = test_logits2
        
        train_data.to_csv(f'{args.save_folder}/train_predictions.csv', index=False) 
        val_data.to_csv(f'{args.save_folder}/val_predictions.csv', index=False) 
        test_data.to_csv(f'{args.save_folder}/test_predictions.csv', index=False) 
    
    print(f"train labels {train_labels}")
    print(f"train predictions {train_preds}")

    print(f'train metrics: \n {get_metrics(train_labels, train_preds)}')
    print(f'val metrics: \n {get_metrics(val_labels, val_preds)}')
    print(f'test metrics: \n {get_metrics(test_labels, test_preds)}')
    
    # save model locally
    torch.save(model.state_dict(),f'{args.save_folder}/network_pos_model')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type = str, default = './swerik/data')
    parser.add_argument("--save_folder", type = str, default = "./swerik/model")
    parser.add_argument("--save_predictions", action="store_true", help="Set this flag to save predictions to csv.")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")

    args = parser.parse_args()
    main(args)