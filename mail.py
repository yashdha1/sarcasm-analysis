import torch
import torch.nn as nn
import pandas as pd
import re
from nltk.corpus import stopwords
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from transformers import AdamW
from tqdm import tqdm


# Dataset Class
class SarcasmDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        tokens = self.texts[idx]
        return {
            'input_ids': tokens['input_ids'].squeeze(0),
            'attention_mask': tokens['attention_mask'].squeeze(0),
            'labels': self.labels[idx]
        }


# Model Class
class BertForSarcasmClassification(nn.Module):
    def __init__(self, num_classes=4) :
        super(BertForSarcasmClassification, self).__init__()
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(in_features=self.bert.config.hidden_size,
                            out_features=num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(self.dropout(pooled_output))


# Preprocessing Function
def preprocessor(text):
    if not isinstance(text, str):
        text = str(text)
    text = re.sub(r'https?://\S+|www\.\S+', 'url', text)    
    text = re.sub(r'\S+@\S+', 'email', text)   
    text = re.sub(r'@\S+', 'user', text)   
    text = re.sub(r'\d+', '', text)   
    text = text.lower()   
    words = text.split()
    stop_words = set(stopwords.words('english'))
    return ' '.join([word for word in words if word not in stop_words])


# Tokenizer Function
def tokenize_function(text):
    return tokenizer(text, padding="max_length", truncation=True, max_length=128, return_tensors="pt")


# Set Device
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# Load Data
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

# Preprocess Data
train['tweets'] = train['tweets'].apply(preprocessor)
test['tweets'] = test['tweets'].apply(preprocessor)

# Map Labels
label_mapping = {"sarcasm": 0, "irony": 1, "figurative": 2, "regular": 3}
train['class'] = train['class'].map(label_mapping).fillna(0).astype(int)
test['class'] = test['class'].map(label_mapping).fillna(0).astype(int)

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Tokenize Texts
train_tokens = train['tweets'].apply(tokenize_function)
test_tokens = test['tweets'].apply(tokenize_function)

# Convert Labels to Tensors
train_labels = torch.tensor(train['class'].values, dtype=torch.long)
test_labels = torch.tensor(test['class'].values, dtype=torch.long)

# Create Dataset and DataLoader
train_dataset = SarcasmDataset(train_tokens, train_labels)
test_dataset = SarcasmDataset(test_tokens, test_labels)

train_dataloader = DataLoader(train_dataset,
                              batch_size=64,
                              shuffle=True)

test_dataloader = DataLoader(test_dataset,
                             batch_size=64,
                             shuffle=False)

# Initialize Model
model = BertForSarcasmClassification(num_classes=4).to(device)

# Optimizer and Loss Function
optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8)
loss_fn = nn.CrossEntropyLoss()

# Training and Testing Loop
EPOCHS = 3
best_accuracy = 0.0

for epoch in range(EPOCHS):
    print(f"Epoch {epoch + 1}/{EPOCHS}")
    print("--" * 30)

    # Training Loop
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0
    
    # TRAINING LOOP ---> 
    for batch in tqdm(train_dataloader, desc="Training") :
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
 
        outputs = model(input_ids, attention_mask)
        loss = loss_fn(outputs, labels)
        
        
        # Mandatory Four
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        train_correct += (outputs.argmax(1) == labels).sum().item()
        train_total += labels.size(0)

    train_accuracy = train_correct / train_total
    print(f"Training Loss: {train_loss:.4f}, Training Accuracy: {train_accuracy * 100:.2f}%")

    # Validation/tEST Loop
    model.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.inference_mode():
        for batch in tqdm(test_dataloader, desc="Validating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask)
            loss = loss_fn(outputs, labels)

            test_loss += loss.item()
            test_correct += (outputs.argmax(1) == labels).sum().item()
            test_total += labels.size(0)

    test_accuracy = test_correct / test_total
    print(f"Validation Loss: {test_loss:.4f}, Validation Accuracy: {test_accuracy * 100:.2f}%")

    # Save the Best Model
    if test_accuracy > best_accuracy:
        best_accuracy = test_accuracy
        torch.save(model.state_dict(), "best_model.pt")
        print("Model improved and saved.")

print(f"\nTraining completed. Best Validation Accuracy: {best_accuracy * 100:.2f}%")

# Acc = 77%
# Acc = 85% testing accuracy....!!!