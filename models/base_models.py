# models/intent_model.py
import torch
from torch import nn
from transformers import BertModel, BertTokenizer

class IntentClassifier(nn.Module):
    def __init__(self, num_intents, model_name="bert-base-chinese"):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_intents)
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.classifier(pooled_output)

    def predict(self, text, device="cpu"):
        self.eval()
        encoding = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=64
        )

        with torch.no_grad():
            input_ids = encoding["input_ids"].to(device)
            attention_mask = encoding["attention_mask"].to(device)
            outputs = self.forward(input_ids, attention_mask)

        probabilities = torch.softmax(outputs, dim=1)
        return probabilities.cpu().numpy()[0]