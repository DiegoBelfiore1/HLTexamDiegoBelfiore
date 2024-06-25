import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from textblob import TextBlob
import matplotlib.pyplot as plt
from collections import Counter
from gensim import corpora, models
from functionwords import FunctionWords
import os

# Initialize FunctionWords
fw = FunctionWords(function_words_list='english')

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()

# Define filler words and function words
filler_words = set(['uh', 'um', 'well', 'actually', 'literally'])

# Function to preprocess and extract features
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in ["...", "’", "n't", "\'s", "\'m", "'re", "'ve"]]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    return tokens

def extract_features(text):
    tokens = preprocess(text)
    # Detect FunctionWords
    function_words_counts = fw.transform(' '.join(tokens))
    
    # Sentiment Analysis
    sentiment = TextBlob(text).sentiment
    
    # POS Tagging
    pos_tags = nltk.pos_tag(tokens)
    pos_counts = Counter(tag for word, tag in pos_tags)
    
    return {
        'tokens': tokens,
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        'function_words_counts': function_words_counts,
        'filler_word_counts': Counter(token for token in tokens if token in filler_words),
        'total_words': len(tokens),
        'pos_counts': pos_counts
    }

# Load the collected data
def load_data(file_paths, label):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            text = file.read()
            features = extract_features(text)
            features['label'] = label
            data.append(features)
    return data

# Paths
antisocial_files = [f"/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/HLT/python/Personality/Antisocial corpus/antisocial {i}.txt" for i in range(1, 11)]
nonantisocial_files = [f"/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/HLT/python/Personality/Normotype/Normotype{i}.txt" for i in range(1, 11)]

antisocial_data = load_data(antisocial_files, 1)
nonantisocial_data = load_data(nonantisocial_files, 0)

# Combine data
all_data = antisocial_data + nonantisocial_data

# Calculate total counts and percentages
def calculate_percentages(data):
    total_words = sum(d['total_words'] for d in data)
    total_function_words = sum(sum(d['function_words_counts']) for d in data)
    total_filler_words = sum(sum(d['filler_word_counts'].values()) for d in data)
    total_polarity = sum(d['polarity'] for d in data)
    total_subjectivity = sum(d['subjectivity'] for d in data)
    
    return {
        'total_words': total_words,
        'function_words_percentage': (total_function_words / total_words) * 100,
        'filler_words_percentage': (total_filler_words / total_words) * 100,
        'average_polarity': total_polarity / len(data),
        'average_subjectivity': total_subjectivity / len(data)
    }

antisocial_percentages = calculate_percentages(antisocial_data)
nonantisocial_percentages = calculate_percentages(nonantisocial_data)

print("Antisocial Percentages:")
print(antisocial_percentages)

print("Non-Antisocial Percentages:")
print(nonantisocial_percentages)

# LDA analysis
def lda_analysis(data, num_topics=5):
    corpus = [features['tokens'] for features in data]
    dictionary = corpora.Dictionary(corpus)
    corpus_bow = [dictionary.doc2bow(text) for text in corpus]
    
    # Train LDA model
    lda_model = models.LdaModel(corpus_bow, num_topics=num_topics, id2word=dictionary, passes=15)
    
    # Calculate word frequency distribution
    word_freq_dist = Counter(word for features in data for word in features['tokens'])
    
    topics = lda_model.print_topics()
    return topics, word_freq_dist

antisocial_topics, antisocial_word_freq_dist = lda_analysis(antisocial_data)
nonantisocial_topics, nonantisocial_word_freq_dist = lda_analysis(nonantisocial_data)

def print_topics_and_word_freq(topics, word_freq_dist, dataset_name):
    print(f"\n{dataset_name} Topics:")
    for topic in topics:
        print(f"Topic {topic[0]}: {topic[1]}")
    
    print(f"\n{dataset_name} Top 5 Most Common Words:")
    for word, freq in word_freq_dist.most_common(5):
        print(f'Word: {word}, Freq: {freq}')

print_topics_and_word_freq(antisocial_topics, antisocial_word_freq_dist, "Antisocial")
print_topics_and_word_freq(nonantisocial_topics, nonantisocial_word_freq_dist, "Non-Antisocial")

# POS analysis
def pos_analysis(data):
    pos_counts = Counter()
    for features in data:
        pos_counts.update(features['pos_counts'])
    return pos_counts

antisocial_pos_counts = pos_analysis(antisocial_data)
nonantisocial_pos_counts = pos_analysis(nonantisocial_data)

def print_pos_counts(pos_counts, dataset_name):
    print(f"\n{dataset_name} POS Counts:")
    for pos, count in pos_counts.most_common():
        print(f'POS: {pos}, Count: {count}')

print_pos_counts(antisocial_pos_counts, "Antisocial")
print_pos_counts(nonantisocial_pos_counts, "Non-Antisocial")
