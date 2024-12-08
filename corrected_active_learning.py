import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from small_text import (
    PoolBasedActiveLearner,
    PredictionEntropy,
    TransformersDataset,
    TransformerModelArguments,
)
from small_text.integrations.transformers.classifiers.factories import TransformerBasedClassificationFactory

# Dataset Paths
annotated_data_path = 'data/data_annotated.csv'

# Load Data
data = pd.read_csv(annotated_data_path)
annotated_data = data.iloc[:2424]
unannotated_data = data.iloc[2424:]

# Extract Features and Labels
X_annotated = annotated_data['text_line'].values
y_annotated = annotated_data['marginal_text'].values
X_unannotated = unannotated_data['text_line'].values

# Model and Tokenizer Configuration
transformer_model_name = 'KB/bert-base-swedish-cased'
tokenizer = AutoTokenizer.from_pretrained(transformer_model_name)
num_classes = len(np.unique(y_annotated))

# Convert Datasets into TransformersDataset
train_dataset = TransformersDataset.from_arrays(
    X_annotated, y_annotated, tokenizer, max_length=60, target_labels=np.arange(num_classes)
)
unannotated_dataset = TransformersDataset.from_arrays(
    X_unannotated, np.full(len(X_unannotated), -1), tokenizer, max_length=60, target_labels=np.arange(num_classes)
)

# Load the Pre-trained Model from .safetensors
def load_existing_model(model_path, transformer_model_name, num_classes):
    model = AutoModelForSequenceClassification.from_pretrained(
        transformer_model_name,
        num_labels=num_classes,
        state_dict=torch.load(model_path, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    )
    return model

existing_model_path = "margin_prediction_model/model.safetensors"  # Replace with actual path to your .safetensors file
pretrained_model = load_existing_model(existing_model_path, transformer_model_name, num_classes)

# Define Active Learner with Pre-trained Model
clf_factory = TransformerBasedClassificationFactory(
    TransformerModelArguments(transformer_model_name),
    num_classes,
    kwargs=dict(device='cuda', mini_batch_size=32, class_weight='balanced')
)
query_strategy = PredictionEntropy()
active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_dataset)

# Inject Pre-trained Model into Active Learner
active_learner.classifier.model = pretrained_model

# Active Learning Loop
results = []
num_queries = 5
batch_size = 50

for i in range(num_queries):
    print(f"Active Learning Iteration #{i + 1}")
    
    # Query the most uncertain samples
    query_indices = active_learner.query(num_samples=batch_size)
    queried_samples = unannotated_dataset[query_indices]
    
    # Simulate annotation using the original dataset
    queried_labels = unannotated_data.iloc[query_indices]['marginal_text'].values
    
    # Update the model with new labeled samples
    active_learner.update(queried_labels)
    
    # Evaluate on the annotated data
    y_pred = active_learner.classifier.predict(train_dataset)
    train_accuracy = accuracy_score(y_annotated, y_pred)
    train_f1 = f1_score(y_annotated, y_pred, average='weighted')  # Weighted F1-score for class imbalance
    
    print(f"Training Accuracy after iteration #{i + 1}: {train_accuracy:.4f}")
    print(f"Training F1-score after iteration #{i + 1}: {train_f1:.4f}")
    results.append((train_accuracy, train_f1))

    # Stop if performance plateaus
    if i > 1 and results[-1][0] <= results[-2][0]:  # Using accuracy for plateau detection
        print("Early stopping triggered.")
        break

# Final Results
print("Active Learning Results (Accuracy, F1-score):", results)
