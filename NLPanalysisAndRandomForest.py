#here we do Filler Words count, Function Words Count, Sentiment Analysis, Lemmatization and RD model training

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
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os


# Initialize FunctionWords
fw = FunctionWords(function_words_list='english')

# Download NLTK resources if not already downloaded
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')

# Initialize NLTK resources
lemmatizer = WordNetLemmatizer()

# Define filler words and function words
filler_words = set(['uh', 'um', 'well', 'actually', 'literally'])

# Initialize counters
function_word_counts = Counter()
filler_word_counts = Counter()
word_counts = Counter()

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
    
    
    for feature_name, count in zip(fw.get_feature_names(), function_words_counts):
        function_word_counts[feature_name] += count
    
    # Detect Fillers
    filler_word_counts.update(token for token in tokens if token in filler_words)
    word_counts.update(tokens)

    #Sentiment Analysis
    sentiment = TextBlob(text).sentiment
    return {
        'tokens': tokens,
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        **{f'func_word_{feature_name}': count for feature_name, count in zip(fw.get_feature_names(), function_words_counts)},
        **{f'filler_word_{word}': tokens.count(word) for word in filler_words}
    }

# Initialize counters for sentiment analysis
positive_count = 0
negative_count = 0

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

# Our total flying dataset
df = pd.DataFrame(antisocial_data + nonantisocial_data)

'''# if you want to save it
file_path = "/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/HLT/python/Personality.csv"
    df.to_csv(file_path, index=False)
    print("DataFrame CSV Saved.")
except Exception as e:
    print(f"Error: {e}")'''


# Random division of the dataset
X = df.drop(columns=['label', 'tokens'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model call
clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

# Training
clf.fit(X_train, y_train)

# Evaluation
y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()