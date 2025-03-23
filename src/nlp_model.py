from transformers import BertTokenizer, BertForSequenceClassification
import torch
import os

class NLPModel:
    def __init__(self, model_dir="../trained_bert"):  # Path relative to src/
        # Resolve the absolute path for better debugging
        base_dir = os.path.dirname(os.path.abspath(__file__))  # Directory of nlp_model.py (src/)
        model_dir = os.path.join(base_dir, model_dir)  # Resolve relative path
        print(f"Looking for trained model in: {model_dir}")  # Debug print
        
        # Check if trained model exists, otherwise fall back to pre-trained
        if os.path.exists(model_dir):
            print("Found trained model. Loading...")
            self.tokenizer = BertTokenizer.from_pretrained(model_dir)
            self.model = BertForSequenceClassification.from_pretrained(model_dir)
        else:
            print("Trained model not found. Falling back to pre-trained BERT.")
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)
        self.intents = ['legal', 'safety', 'self_defense']

    def predict_intent(self, text):
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits
        predicted_class = torch.argmax(logits, dim=1).item()
        return self.intents[predicted_class]

# Example usage
if __name__ == "__main__":
    nlp = NLPModel()
    intent = nlp.predict_intent("How to defend myself?")
    print(f"Predicted intent: {intent}")