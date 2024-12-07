import torch
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from transformers import AutoTokenizer
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory
from small_text import (
    PoolBasedActiveLearner, 
    PredictionEntropy, 
    random_initialization_balanced, 
    TransformersDataset, 
    TransformerModelArguments
)
from scipy.stats import entropy
from torch.optim.lr_scheduler import StepLR

# Load the data
df_train = pd.read_csv('data_stratified_sampling/train_data_stratified.csv')
df_test = pd.read_csv('data_stratified_sampling/test_set_stratified.csv')

num_classes = np.unique(df_train['marginal_text']).shape[0]

# Load the tokenizer and create datasets
transformer_model_name = 'KB/bert-base-swedish-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)

target_labels = np.arange(num_classes)
train = TransformersDataset.from_arrays(df_train['text_line'], df_train['marginal_text'], tokenizer, max_length=60, target_labels=target_labels)
test = TransformersDataset.from_arrays(df_test['text_line'], df_test['marginal_text'], tokenizer, max_length=60, target_labels=target_labels)

# Define the model and active learner
transformer_model = TransformerModelArguments(transformer_model_name)
clf_factory = TransformerBasedClassificationFactory(
    transformer_model,
    num_classes,
    kwargs=dict({
        'device': 'cuda',
        'mini_batch_size': 32,
        'class_weight': 'balanced', # Adding dropout for regularization
    })
)

query_strategy = PredictionEntropy()
active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train)

# Initialize active learner
def initialize_active_learner(active_learner, y_train):
    indices_initial = random_initialization_balanced(y_train, n_samples=20)
    active_learner.initialize_data(indices_initial, y_train[indices_initial])
    return indices_initial

indices_labeled = initialize_active_learner(active_learner, train.y)

# Define evaluation with validation
def evaluate(active_learner, train, test, indices_labeled):
    y_pred_train = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)
    
    train_acc = accuracy_score(y_pred_train, train.y)
    test_acc = accuracy_score(y_pred_test, test.y)
    
    print(f'Train Accuracy: {train_acc:.4f}')
    print(f'Test Accuracy: {test_acc:.4f}')
    
    return train_acc, test_acc

# Early stopping
class EarlyStopping:
    def __init__(self, patience=5, delta=0.01):
        self.patience = patience
        self.delta = delta
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

# Training loop with improvements
results = []
early_stopping = EarlyStopping(patience=3)
num_queries = 5

for i in range(num_queries):
    indices_queried = active_learner.query(num_samples=5)
    y = train.y[indices_queried]
    active_learner.update(y)
    
    # Evaluate and log performance
    print(f'Iteration #{i+1}')
    train_acc, test_acc = evaluate(active_learner, train, test, indices_labeled)
    results.append((train_acc, test_acc))
    
    if early_stopping(-test_acc):
        print("Early stopping triggered")
        break
