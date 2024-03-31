import pandas as pd
from transformers import RobertaTokenizer, RobertaModel
import torch
import torch.nn as nn
import torch.optim as optim
import os
from tqdm import tqdm
from transformers import RobertaTokenizer
import pandas as pd
import textwrap
torch.manual_seed(0)


if os.path.exists('/home/sslater/data/trainwindowsimproved.csv'):
    data_path = '/home/sslater/data/trainwindowsimproved.csv'
else:
    data_path = '/Users/stevenslater/Desktop/FinalProject/Data/trainwindowsimproved.csv'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_df = pd.read_csv(data_path)

# define the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaModel.from_pretrained('roberta-base')

# define the classifier
classifier = nn.Linear(model.config.hidden_size, 1)
sigmoid = nn.Sigmoid()

#put on GPU
model = model.to(device)
classifier = classifier.to(device)

#weight for the positive class - didnt need in end

#weight_for_pos = 2
#print(weight_for_pos)
#pos_weight = torch.tensor([weight_for_pos], dtype=torch.float32).to(device)

# EITHER PASS WEIGHT FOR WEIGHING POS OR NOTHING
#loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
loss_function = nn.BCEWithLogitsLoss() #this is used for when no weights

#lr
optimizer = optim.Adam(model.parameters(), lr=0.00002)

def run(train_df, batch_size=32, num_epochs=1):
    for epoch in range(num_epochs):
        print(f"Starting Epoch {epoch + 1}/{num_epochs}")

        
        df_shuffled = train_df.sample(frac=1).reset_index(drop=True)

        #batching
        for i in tqdm(range(0, len(train_df), batch_size), desc="Processing Batches"):
            
            batch = df_shuffled.iloc[i:i + batch_size]

            optimizer.zero_grad()

            total_loss = 0.0

            for _, row in batch.iterrows():
                
                conversation_chunk = row['conversation_chunks']
                label = row['labels']
                
                input_ids = tokenizer.encode(conversation_chunk, add_special_tokens=True, return_tensors='pt').to(device)
                
                output = model(input_ids)[0]
                
                output = output[:, 0, :]
                
                logits = classifier(output)
                logits = logits.squeeze(-1)

                label_tensor = torch.tensor([label], dtype=torch.float).to(device)
                loss = loss_function(logits, label_tensor)
                
                total_loss += loss
            
            average_loss = total_loss / len(batch)
            average_loss.backward()
            optimizer.step()
            print(f"Average loss for this batch: {average_loss.item()}")

run(train_df)
torch.save(model.state_dict(), 'newWindowsRoberta1epoch.pth')



