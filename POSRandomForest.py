#Just the same algorithm as before but including POS tagging in the dataset which turned out to not work well

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
nltk.download('averaged_perceptron_tagger')


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
    pos_tags = nltk.pos_tag(tokens)
    return tokens, pos_tags

def extract_features(text):
    tokens, pos_tags = preprocess(text)
    function_words_counts = fw.transform(' '.join(tokens))
    
    for feature_name, count in zip(fw.get_feature_names(), function_words_counts):
        function_word_counts[feature_name] += count
    
    filler_word_counts.update(token for token in tokens if token in filler_words)
    word_counts.update(tokens)
    
    pos_counts = Counter(tag for word, tag in pos_tags) #POS tagging 

    
    sentiment = TextBlob(text).sentiment
    return {
        'tokens': tokens,
        'polarity': sentiment.polarity,
        'subjectivity': sentiment.subjectivity,
        **{f'func_word_{feature_name}': count for feature_name, count in zip(fw.get_feature_names(), function_words_counts)},
        **{f'filler_word_{word}': tokens.count(word) for word in filler_words},
        **{f'pos_{tag}': count for tag, count in pos_counts.items()}
    }

# Initialize counters for sentiment analysis
positive_count = 0
negative_count = 0

def load_data(file_paths, label):
    data = []
    for file_path in file_paths:
        with open(file_path, 'r') as file:
            text = file.read()
            features = extract_features(text)
            features['label'] = label
            data.append(features)
    return data

antisocial_files = [f"/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/HLT/python/Personality/Antisocial corpus/antisocial {i}.txt" for i in range(1, 11)]
nonantisocial_files = [f"/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/HLT/python/Personality/Normotype/Normotype{i}.txt" for i in range(1, 11)]

antisocial_data = load_data(antisocial_files, 1)
nonantisocial_data = load_data(nonantisocial_files, 0)

df = pd.DataFrame(antisocial_data + nonantisocial_data)
file_path = "/Users/diegobelfiore/Desktop/Università/Trento/Lezioni/HLT/python/PersonalityPOS.csv"

# Print the DataFrame with POS tagging
print(df.head())

try:
    df.to_csv(file_path, index=False)
    print("DataFrame saved as CSV.")
except Exception as e:
    print(f"Error saving DataFrame: {e}")

X = df.drop(columns=['label', 'tokens'])
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
scores = cross_val_score(clf, X, y, cv=5)
print("Cross-validation scores:", scores)
print("Average cross-validation score:", scores.mean())

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

corpus = [features['tokens'] for features in antisocial_data + nonantisocial_data]
dictionary = corpora.Dictionary(corpus)
corpus_bow = [dictionary.doc2bow(text) for text in corpus]

lda_model = models.LdaModel(corpus_bow, num_topics=5, id2word=dictionary, passes=15)

word_freq_dist = Counter(word_counts)

with open("topics.txt", "w", encoding="utf-8") as topics_file:
    topics = lda_model.print_topics()
    for topic in topics:
        topics_file.write(f"Topic {topic[0]}: {topic[1]}\n")
    for word, freq in word_freq_dist.most_common(5):
        topics_file.write(f'\nWord: {word}, Freq: {freq}\n')

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

# Plot Matrix
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g', xticklabels=['Predicted 0', 'Predicted 1'], yticklabels=['Actual 0', 'Actual 1'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted label')
plt.ylabel('True label')
plt.show()