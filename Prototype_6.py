import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from tabulate import tabulate

# Load data function with encoding handling
def load_data(file_path):
    try:
        return pd.read_csv(file_path, encoding='utf-8')
    except UnicodeDecodeError:
        return pd.read_csv(file_path, encoding='latin1')

# Function to preprocess text and truncate descriptions
def preprocess_text(text):
    if isinstance(text, str):
        text = text.lower()
        text = ''.join([char for char in text if char.isalpha() or char.isspace()])
        # Limit description length for display (adjust as needed)
        text = text[:60] + '...' if len(text) > 60 else text
    else:
        text = ''
    return text

# Load train and test data
train_data = load_data('Train-data.csv')
test_data = load_data('Test-data.csv')
test_data_solutions = load_data('Test-data-solutions.csv')

# Preprocess descriptions
train_data['Description'] = train_data['Description'].apply(preprocess_text)
test_data['Description'] = test_data['Description'].apply(preprocess_text)

# Clean and preprocess genre labels
def clean_genre_labels(genres):
    if isinstance(genres, str):
        genres = [genre.strip() for genre in genres.split(',')]
        genres = [genre for genre in genres if genre in genre_classes]  # Filter unwanted genres
        return genres
    return []

# Define genres
genre_classes = [
    'drama', 'thriller', 'adult', 'action', 'adventure', 'horror', 'short', 
    'family', 'talk-show', 'game-show', 'music', 'musical', 'documentary', 
    'sport', 'comedy', 'romance', 'history', 'war', 'mystery', 'crime', 
    'fantasy', 'animation', 'reality-tv', 'western', 'biography', 'sci-fi', 
    'news'
]

# Clean and preprocess genre labels for training data
train_data['Genre'] = train_data['Genre'].apply(clean_genre_labels)

# Initialize MultiLabelBinarizer
mlb = MultiLabelBinarizer(classes=genre_classes)

# Transform genre labels
y_train = mlb.fit_transform(train_data['Genre'])

# Vectorize text using TF-IDF
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data['Description'])
X_test_tfidf = tfidf_vectorizer.transform(test_data['Description'])

# Train logistic regression model
model_lr = OneVsRestClassifier(LogisticRegression(max_iter=1000))
model_lr.fit(X_train_tfidf, y_train)

# Predict genres
predictions_lr = model_lr.predict(X_test_tfidf)
predicted_genres_lr = mlb.inverse_transform(predictions_lr)

# Prepare results DataFrame
results_lr = pd.DataFrame({
    'ID': test_data['ID'],
    'Name': test_data['Name'],
    'Description': test_data['Description'],
    'Predicted_Genre': [' '.join(genres) for genres in predicted_genres_lr]
})

# Add correct genres from solutions file
results_lr['Correct_Genre'] = test_data_solutions['Genre']

# Display table with limited columns and truncated descriptions
chunk_size = 10
start = 0

while start < len(results_lr):
    end = min(start + chunk_size, len(results_lr))
    # Truncate Description column for display
    results_lr_display = results_lr[['ID', 'Name', 'Description', 'Predicted_Genre', 'Correct_Genre']].iloc[start:end].copy()
    results_lr_display['Description'] = results_lr_display['Description'].apply(lambda x: x[:60] + '...' if len(x) > 60 else x)
    print(tabulate(results_lr_display, headers='keys', tablefmt='psql', showindex=False))
    start = end
    if start < len(results_lr):
        user_input = input("Type 'more' to see more results, or press Enter to exit: ").strip().lower()
        if user_input != 'more':
            break
