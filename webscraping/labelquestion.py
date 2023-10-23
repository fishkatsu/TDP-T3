import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from pathlib import Path

def preprocess_text(text):
    nltk.download('punkt')
    nltk.download('stopwords')

    stop_words = set(stopwords.words('english'))
    # Tokenize the text
    words = nltk.word_tokenize(text.lower())

    # Remove punctuation and stop words
    words = [word for word in words if word.isalnum() and word not in stop_words]

    return " ".join(words)

def process_faq_csv():
    file_path = Path(".\\faq_data.csv")
    df = pd.read_csv(file_path)
    df['Processed_Question'] = df['Question'].apply(preprocess_text)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(df['Processed_Question'])

    # Topic predefined by model
    # Define the number of clusters as None to let the model determine them
    num_clusters = 10
    agg_clustering = AgglomerativeClustering(n_clusters=num_clusters)

    # Convert the sparse TF-IDF matrix to a dense NumPy array
    dense_tfidf_matrix = tfidf_matrix.toarray()
    df['Topic'] = agg_clustering.fit_predict(dense_tfidf_matrix)

    # Generate string labels for topics
    topic_labels = generate_topic_labels(df, 'Topic')
    df['Topic_Label'] = df['Topic'].map(topic_labels)

    # Extract the 'Topic_Label' values as a list
    topics = df['Topic_Label'].unique().tolist()

    return topics

# Define a function to automatically generate topic labels
def generate_topic_labels(df, topic_col):
    topic_labels = {}

    for topic in df[topic_col].unique():
        cluster_questions = df[df[topic_col] == topic]['Processed_Question']
        # Extract top keywords or terms for each cluster
        vectorizer = CountVectorizer()
        cluster_word_counts = vectorizer.fit_transform(cluster_questions)
        sum_word_counts = cluster_word_counts.sum(axis=0)
        word_freq = [(word, sum_word_counts[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        sorted_word_freq = sorted(word_freq, key=lambda x: x[1], reverse=True)
        top_keywords = [word for word, _ in sorted_word_freq[:10]]  # Adjust the number of top keywords
        unique_top_keywords = list(set(top_keywords))
        topic_labels[topic] = f"Topic: {' | '.join(unique_top_keywords)}"
    return topic_labels

# Print the questions and their assigned topics
# def main():
#     df = process_faq_csv()
#     print(df["Topic_Label"].unique())
#     # continue to use the data frame after this
# main()