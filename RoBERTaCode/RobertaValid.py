import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
torch.manual_seed(0)

if os.path.exists('/home/sslater/data/valwindowsimproved.csv'):
    data_path = '/home/sslater/data/valwindowsimproved.csv'
else:
    data_path = '/Users/stevenslater/Desktop/FinalProject/Data/valwindowsimproved.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


val_df = pd.read_csv(data_path)

tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')
classifier = nn.Linear(model.config.hidden_size, 1)


# load model
model.load_state_dict(torch.load('newWindowsRoberta1epochWeight2.pth'))
model = model.to(device)
classifier = classifier.to(device)

# define sigmoid function
sigmoid = nn.Sigmoid()

def predict(df):
    model.eval()
    classifier.eval()
    
    predictions = []
    actuals = []

    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating"):
            conversation_chunk = row['conversation_chunks']
            label = row['labels']

            input_ids = tokenizer.encode(conversation_chunk, add_special_tokens=True, return_tensors='pt').to(device)

            output = model(input_ids)[0]
            output = output[:, 0, :]
            logits = classifier(output)
            logits = logits.squeeze(-1)

            probability = torch.sigmoid(logits)
            print(conversation_chunk,probability, label)
            probability = probability.squeeze(-1)

            binary_prediction = 1 if probability.item() > 0.50 else 0

            predictions.append(binary_prediction)
            actuals.append(label)

    return predictions, actuals

# predict on the validation set
predictions, actuals = predict(val_df)


accuracy = accuracy_score(actuals, predictions)
precision = precision_score(actuals, predictions)
recall = recall_score(actuals, predictions)
f1 = f1_score(actuals, predictions)
f1_pos = f1_score(actuals, predictions, pos_label=1)

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")
print(f"F1 Score - Positive Class {f1_pos}")