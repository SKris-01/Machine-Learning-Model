import pandas as pd
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from tabulate import tabulate

# OMDB API key
API_KEY = 'b8f0bdcf'

# Function to fetch genres from OMDB API
def fetch_genres_from_omdb(movie_name):
    url = f"http://www.omdbapi.com/?t={movie_name}&apikey={API_KEY}"
    response = requests.get(url)
    data = response.json()
    if response.status_code == 200 and 'Genre' in data:
        genres = data['Genre'].lower().split(', ')
        return genres
    else:
        return []

# Load the data with appropriate encoding handling
def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')

train_data = load_data('Train-data.csv')
test_data = load_data('Test-data.csv')
test_data_solutions = load_data('Test-data-solutions.csv')

# Preprocess the data
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char in text if char.isalpha() or char.isspace()])
        # Limit description length for display (adjust as needed)
        text = text[:60] + '...' if len(text) > 60 else text
    else:
        text = ''
    return text

train_data['Description'] = train_data['Description'].apply(preprocess_text)
test_data['Description'] = test_data['Description'].apply(preprocess_text)

# Remove extra spaces from genre labels and split into lists
train_data['Genre'] = train_data['Genre'].apply(lambda x: [genre.strip() for genre in x.split(',')] if isinstance(x, str) else [])
test_data_solutions['Genre'] = test_data_solutions['Genre'].apply(lambda x: [genre.strip() for genre in x.split(',')] if isinstance(x, str) else [])

# Define the genre classes
genre_classes = [
    'drama', 'thriller', 'adult', 'action', 'adventure', 'horror', 'short', 
    'family', 'talk-show', 'game-show', 'music', 'musical', 'documentary', 
    'sport', 'comedy', 'romance', 'history', 'war', 'mystery', 'crime', 
    'fantasy', 'animation', 'reality-tv', 'western', 'biography', 'sci-fi', 
    'news'
]

# Initialize MultiLabelBinarizer with the defined classes
mlb = MultiLabelBinarizer(classes=genre_classes)

# Fit and transform the training data genres
y_train = mlb.fit_transform(train_data['Genre'])

# Extract features using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Description'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['Description'])

# Use OneVsRestClassifier to handle multi-label classification
model_nb = OneVsRestClassifier(MultinomialNB())
model_lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_svc = OneVsRestClassifier(LinearSVC())

# Train the models
model_nb.fit(X_train_tfidf, y_train)
model_lr.fit(X_train_tfidf, y_train)
model_svc.fit(X_train_tfidf, y_train)

# Predict on test data
predictions_nb = model_nb.predict(X_test_tfidf)
predictions_lr = model_lr.predict(X_test_tfidf)
predictions_svc = model_svc.predict(X_test_tfidf)

# Convert predictions back to genre labels
predicted_genres_nb = mlb.inverse_transform(predictions_nb)
predicted_genres_lr = mlb.inverse_transform(predictions_lr)
predicted_genres_svc = mlb.inverse_transform(predictions_svc)

# Fetch predicted genres from OMDB API
test_data['Predicted_Genre'] = test_data['Name'].apply(fetch_genres_from_omdb)

# Compare with correct genres from Test-data-solutions
test_data['Correct_Genre'] = test_data_solutions['Genre']

# Create DataFrame with results
results_lr = pd.DataFrame({
    'ID': test_data['ID'],
    'Name': test_data['Name'],
    'Description': test_data['Description'],
    'Predicted_Genre': [' '.join(genres) for genres in test_data['Predicted_Genre']],
    'Correct_Genre': [' '.join(genres) for genres in test_data['Correct_Genre']]
})

# Display results in chunks with 'more' command
chunk_size = 10
start = 0

while start < len(results_lr):
    end = min(start + chunk_size, len(results_lr))
    print(tabulate(results_lr.iloc[start:end], headers='keys', tablefmt='psql', showindex=False))
    start = end
    if start < len(results_lr):
        user_input = input("Type 'more' to see more results: ").strip().lower()
        if user_input != 'more':
            break

# Save results to CSV
results_lr.to_csv('Movie-predictions-lr.csv', index=False)
