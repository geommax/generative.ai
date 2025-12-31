import torch
from transformers import BertTokenizer, BertForSequenceClassification

# --------------------------
# 1. Configuration

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_DIR = "./my_sequence_classification_model"  

# --------------------------
# 2. Load saved model & tokenizer

tokenizer = BertTokenizer.from_pretrained(MODEL_DIR)
model = BertForSequenceClassification.from_pretrained(MODEL_DIR)
model.to(DEVICE)
model.eval()  

# --------------------------
# 3. Prepare evaluation sentences

test_sentences = [
    "Free lottery winner!",
    "Book a cab to airport",
    "You have won $1000, click here",
    "Schedule meeting for tomorrow"
]

# Tokenize
test_enc = tokenizer(
    test_sentences,
    padding=True,
    truncation=True,
    return_tensors="pt"
)

# Move to device
test_enc = {k: v.to(DEVICE) for k, v in test_enc.items()}

# --------------------------
# 4. Inference

with torch.no_grad():
    outputs = model(**test_enc)
    predictions = torch.argmax(outputs.logits, dim=1)

# --------------------------
# 5. Display predictions
for text, pred in zip(test_sentences, predictions):
    print(text, "→", "Spam" if pred.item() == 1 else "Not Spam")
    

print("outputs.logits:", outputs.logits)
    
