#This code eliminate all fillers and function words to analyse topics for each class

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
from gensim import corpora, models
from functionwords import FunctionWords
from collections import Counter
import os

# Initialize FunctionWords
fw = FunctionWords(function_words_list='english')

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()

# Define filler words and extend stopwords
filler_words = set(['uh', 'um', 'well', 'actually', 'literally'])
stop_words = set(stopwords.words('english')).union(filler_words)

# Function to preprocess and extract features
def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [token for token in tokens if token not in string.punctuation]
    tokens = [token for token in tokens if token not in ["...", "’", "n't", "\'s", "\'m", "'re", "'ve"]]
    tokens = [token for token in tokens if token not in stop_words]
    tokens = [lemmatizer.lemmatize(token) for token in tokens]
    tokens = [token for token in tokens if token not in fw.function_words_list]
    return tokens

def extract_features(text):
    tokens = preprocess(text)
    return {'tokens': tokens}

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

print_topics_and_word_freq(antisocial_topics, antisocial_word_freq_dist, "Antisocial")
print_topics_and_word_freq(nonantisocial_topics, nonantisocial_word_freq_dist, "Non-Antisocial")
