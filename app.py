from flask import Flask, request, render_template
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the CSV file
df = pd.read_csv('movies.csv')

# Fill NaN values
for feature in ['cast', 'Genre', 'director']:
    df[feature] = df[feature].fillna('')

# Function to combine features into a single string
def combined_features(row):
    return row['cast'] + " " + row['Genre'] + " " + row['director']

# Create a new column with the combined features
df['combined_features'] = df.apply(combined_features, axis=1)

# Load pre-trained GloVe embeddings
def load_glove_embeddings(filepath):
    embeddings_index = {}
    with open(filepath, encoding="utf8") as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

glove_embeddings = load_glove_embeddings('glove.6B.100d.txt')  # Adjust the file path as needed

# Function to create a vector for a text using GloVe
def get_glove_vector(text, embeddings, dim=100):
    words = text.split()
    word_vectors = [embeddings[word] for word in words if word in embeddings]
    if not word_vectors:
        return np.zeros(dim)
    return np.mean(word_vectors, axis=0)

# Create GloVe vectors for all combined features
df['glove_vector'] = df['combined_features'].apply(lambda x: get_glove_vector(x, glove_embeddings))

# Stack the vectors into a matrix
glove_matrix = np.stack(df['glove_vector'].values)

# Calculating the cosine similarity
cosine_sim = cosine_similarity(glove_matrix)

# Function to get the title from the index
def get_title_from_index(index):
    return df[df.index == index]['title'].values[0]

# Function to get the index from the title (case-insensitive)
def get_index_from_title(title):
    title = title.lower()
    df['title_lower'] = df['title'].str.lower()
    return df[df.title_lower == title].index.values[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    movie_user_likes = request.form['movie']
    try:
        movie_index = get_index_from_title(movie_user_likes)
    except IndexError:
        return render_template('index.html', recommendations=[], error='Movie not found in the database')

    # Get a list of similar movies in the form (movie_index, similarity_score)
    similar_movies = list(enumerate(cosine_sim[movie_index]))

    # Sort the list of similar movies in descending order of similarity score
    sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    # Get the titles of the first 15 similar movies
    recommended_movies = [get_title_from_index(movie[0]) for movie in sorted_similar_movies[1:16]]

    return render_template('index.html', recommendations=recommended_movies, error='')

if __name__ == '__main__':
    app.run(debug=True)
