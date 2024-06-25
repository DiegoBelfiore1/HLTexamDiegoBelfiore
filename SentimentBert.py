from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Carica il modello e il tokenizer
model = BertForSequenceClassification.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')
tokenizer = BertTokenizer.from_pretrained('nlptown/bert-base-multilingual-uncased-sentiment')

model.eval()

corpus = []

# Max sequence length per BERT
max_sequence_length = 512

# TOKENIZATION
c = 1
while c <= 10:
    name = f"/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/HLT/python/Personality/Antisocial corpus/antisocial {c}.txt"
    with open(name, "r") as file:
        for line in file:
            if any(letter in line.lower() for letter in "abcdefghilmnopqurstuvwxyz"):
                # Tokenization per Sentiment Bert
                encoded_line = tokenizer.encode(line, add_special_tokens=True, max_length=max_sequence_length, truncation=True, return_tensors='pt')
                # Effettua l'analisi
                with torch.no_grad():
                    outputs = model(encoded_line)
                # Otteniamo i label
                sentiment_label = torch.argmax(outputs.logits).item()
                # Rendiamo i token leggibili
                tokens = tokenizer.convert_ids_to_tokens(encoded_line.squeeze())
                # Aggiungiamo il sentimento al corpus
                corpus.append((tokens, sentiment_label))
    c += 1

# Calcolo delle emozioni positive e negative
num_negative = sum(1 for _, label in corpus if label <= 2)
num_positive = sum(1 for _, label in corpus if label >= 3)

# Percentuale di emozioni positive e negative
total_sentences = len(corpus)
percent_negative = (num_negative / total_sentences) * 100 if total_sentences > 0 else 0
percent_positive = (num_positive / total_sentences) * 100 if total_sentences > 0 else 0

print("Positive emotions:", percent_positive)
print("Negative emotions:", percent_negative)
