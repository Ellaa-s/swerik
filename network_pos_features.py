import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
import argparse
from tqdm import tqdm
import pandas as pd
import numpy as np


# Check if evaluation is correct
def evaluate(model, loader, device):
    loss, accuracy = 0.0, []
    model.eval()
    for batch in tqdm(loader, total=len(loader)):
        inputs = batch[0].to(device)
        labels = batch[1].to(device)
        #print(f"dtype input, labels: {type(inputs),type(labels)}")
        with torch.no_grad():
            output = model(inputs)
        #print(f"model output: {output}")
        # Compute loss
        loss_function = torch.nn.BCEWithLogitsLoss()  
        loss += loss_function(output, labels)
        #print(f"loss: {loss}")
        # Compute predictions
        #preds_batch = torch.argmax(output, axis=1)
        preds_batch = (output > 0.0).long().squeeze()  # Converts probabilities to labels (0 or 1)
        batch_acc = torch.mean((preds_batch == labels).float())
        accuracy.append(batch_acc)

    accuracy = torch.mean(torch.tensor(accuracy))
    return loss, accuracy

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
        #print(f"output: {output}")
        preds_batch = (output > 0.0).long()#.squeeze()
        #preds_batch = torch.argmax(output, axis=1)
        #print(f"preds_batch {preds_batch}")
        preds.extend(preds_batch.tolist())
        #preds.append(preds_batch)
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
  return float(acc), float(pre), float(rec), float(f_1)

n_epochs = 30
batch_size = 8
num_workers = 2
learning_rate = 0.00003

#Feedforward Neural Network for positional features
class PositionalFFNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = nn.Linear(9, 180)
        self.hidden_layers = nn.Sequential(
            nn.Linear(180, 180),
            nn.GELU(),
            nn.Linear(180, 180),
            nn.GELU(),
            nn.Linear(180, 180),
            nn.GELU(),
            nn.Linear(180, 180),
            nn.GELU()
        )
        self.output_layer = nn.Linear(180, 1)

    def forward(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x.squeeze(1)
    
    # No output layer when combining the network with BERT
    def extract_features(self, x):
        x = self.input_layer(x)
        x = self.hidden_layers(x)
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
    dataset.loc[:, num_features] = scaler.transform(dataset[num_features])
    
    # Convert to tensors
    #id_tensor = torch.tensor(encoded_ids, dtype=torch.long) 
    numerical_tensor = torch.tensor(dataset[num_features].values, dtype=torch.float32)
    boolean_tensor = torch.tensor(dataset[boolean_features].values, dtype=torch.bool) 
    labels_tensor = torch.tensor(dataset["marginal_text"], dtype=torch.float) 
    
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
    #train_data=train_data.iloc[:1000,:]
    #test_data=train_data.iloc[:150,:]
    #val_data=val_data.iloc[:150,:]   
        
    train_dataset = create_tensordataset(train_data)
    test_dataset = create_tensordataset(test_data)
    val_dataset = create_tensordataset(val_data)
    
    # Load data in pytorch
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
    
    model = PositionalFFNN()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),lr = learning_rate)
    criterion = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    best_model_state_dict = model.state_dict()
    count = 0
    
    # Each epoch
    for epoch in range(n_epochs):
        print(f"Start epoch {epoch}!")
        train_loss = 0
        model.train()

        for i, batch in enumerate(tqdm(train_loader, total = len(train_loader))):
            optimizer.zero_grad()

            # Extract the features and labels from the batch
            features, labels = batch 
            
            # Move data to the device (GPU/CPU)
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass: get predictions from the model
            predictions = model(features)
            #print(f"predictions that come out of the model {predictions}")
            loss = criterion(predictions.squeeze(), labels.float())
            train_loss += loss.item()

            # Backpropagation
            loss.backward()
            optimizer.step()

        # Evaluation
        val_loss, val_accuracy = evaluate(model, val_loader, device)
        
        # Save best model so far   
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state_dict = model.state_dict()
            count = 0
        else: # early stopping if validation loss does not increase
            count += 1
            if count == 2:
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
    train_labels = train_data['marginal_text'].astype(int).reset_index(drop=True)
    
    val_preds, val_logits = get_predictions(model, val_loader, device)
    val_preds = pd.Series(val_preds)
    val_labels = val_data['marginal_text'].astype(int).reset_index(drop=True)
    
    test_preds, test_logits = get_predictions(model, test_loader, device)
    test_preds = pd.Series(test_preds)
    test_labels = test_data['marginal_text'].astype(int).reset_index(drop=True)
    
    # save predictions if flagged when running script
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
    
    #print(f"train labels {train_labels}")
    #print(f"train predictions {train_preds}")

    print(f'train metrics: \n {get_metrics(train_labels, train_preds)}')
    print(f'val metrics: \n {get_metrics(val_labels, val_preds)}')
    print(f'test metrics: \n {get_metrics(test_labels, test_preds)}')
    
    # save model locally
    torch.save(model.state_dict(),f'{args.save_folder}/positional_ffnn.pt')

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_folder", type = str, default = './swerik/data')
    parser.add_argument("--save_folder", type = str, default = "./swerik/model")
    parser.add_argument("--save_predictions", action="store_true", help="Set this flag to save predictions to csv.")
    parser.add_argument("--cuda", action="store_true", help="Set this flag to run with cuda.")

    args = parser.parse_args()
    main(args)