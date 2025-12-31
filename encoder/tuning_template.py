# Tuning Types
# Freeze all encoder layers, only train the classifier head.
# Insert small low-rank matrices inside attention or FFN layers
# Add trainable tokens (embeddings) prepended to input

# This template demonstrates few-shot fine-tuning like hard fine tuning of a BERT model for sequence classification.
# Soft tuning usually targets encoder layers or classifier head in encoder model.
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertForSequenceClassification
from pathlib import Path

# --------------------------
# 1. Configuration

MODEL_NAME = "bert-base-uncased"
NUM_LABELS = 2        # change per task
EPOCHS = 30            
LR = 2e-5
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SAVE_DIR = "./my_sequence_classification_model"  # folder to save model

# --------------------------
# 2. Load tokenizer & model

tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)
model = BertForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)
model.to(DEVICE)

# --------------------------
# 3. Prepare few-shot dataset

train_texts = [
    "I want to buy a new phone",
    "Win a free iPhone now!",
    "Book a table for 2 at a restaurant",
    "Congratulations, you have won!",
    "Schedule a meeting with the team"
]
train_labels = [0, 1, 0, 1, 0]  # zero: non-spam, one: spam

train_encodings = tokenizer(
    train_texts,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

labels = torch.tensor(train_labels)

class FewShotDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels
    def __len__(self):
        return len(self.labels)
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = FewShotDataset(train_encodings, labels)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# --------------------------
# 4. Optimizer & loss

optimizer = AdamW(model.parameters(), lr=LR)

# --------------------------
# 5. Full-FineTuning training loop

model.train()
for epoch in range(EPOCHS):
    total_loss = 0
    for batch in train_loader:
        input_ids = batch['input_ids'].to(DEVICE)
        attention_mask = batch['attention_mask'].to(DEVICE)
        labels_batch = batch['labels'].to(DEVICE)

        optimizer.zero_grad()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss/len(train_loader):.4f}")

# --------------------------
# 6. Save model & tokenizer

Path(SAVE_DIR).mkdir(parents=True, exist_ok=True)
model.save_pretrained(SAVE_DIR)
tokenizer.save_pretrained(SAVE_DIR)

print(f"Model and tokenizer saved to {SAVE_DIR}")
