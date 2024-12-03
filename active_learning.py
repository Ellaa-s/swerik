import numpy as np
from transformers import BertTokenizer
import torch
from modAL.models import ActiveLearner
from modAL.uncertainty import uncertainty_sampling
from tqdm import trange, tqdm
import pandas as pd
#from safetensors.torch import load_file
from torch.utils.data import TensorDataset, DataLoader
#from transformers import AutoModelForSequenceClassification,AutoTokenizer, get_linear_schedule_with_warmup
#from transformers import BertForSequenceClassification, Trainer, TrainingArguments
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
    labels = torch.tensor(df['marginal_text'].tolist()).long()

    return input_ids, attention_masks, labels


# Load pre-trained tokenizer and trained model
tokenizer = BertTokenizer.from_pretrained('KB/bert-base-swedish-cased')

model = torch.load('output/margin_prediction_model/model.safetensors')  # Load the pre-trained model from a file

df = pd.read_csv('./data/test_set.csv')
n_initial = 102

intial_idx =  np.random.choice(range(len(df)), size=n_initial, replace=False)
train_data = df[intial_idx]



train_input_ids, train_attention_masks, train_labels = encode(train_data, tokenizer)
train_dataset = TensorDataset(train_input_ids, train_attention_masks, train_labels)

