import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import string

# Function to load data from CSV
def load_data(filename):
    try:
        data = pd.read_csv(filename)
        return data
    except FileNotFoundError:
        print(f"Error: File '{filename}' not found!")
        return None

# Text Preprocessing function
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

# Function to train the model
def train_model(data_file):
    data = load_data(data_file)
    if data is not None:
        descriptions = data['Description']
        genres = data['Genre']

        descriptions = descriptions.apply(preprocess_text)

        # Feature Engineering - TF-IDF Vectorization
        vectorizer = TfidfVectorizer(stop_words='english')
        features = vectorizer.fit_transform(descriptions)

        # Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(features, genres, test_size=0.2, random_state=42)

        # Machine Learning Model - Multinomial Naive Bayes
        model = MultinomialNB()
        model.fit(X_train, y_train)

        return model, vectorizer  # Return both model and vectorizer for prediction
    else:
        return None, None

# Function to predict genre
def predict_genre(description, vectorizer, model):
    if description is None:
        return "Unknown"  # Default for missing descriptions
    description = preprocess_text(description)
    new_feature = vectorizer.transform([description])
    predicted_genre = model.predict(new_feature)[0]
    return predicted_genre

# Function to display movie data with predicted genres
def display_predictions(data_file, model, vectorizer):
    data = load_data(data_file)
    if data is not None:
        print(f"{'ID':<6}{'Name':<30}{'Description':<50}{'Genre'}")
        print("-" * 100)
        for index, row in data.iterrows():
            movie_id = row["ID"]
            movie_name = row["Name"]
            description = row["Description"]
            predicted_genre = predict_genre(description, vectorizer, model)
            description_snippet = description[:47] + "..." if description and len(description) > 47 else description
            print(f"{movie_id:<6}{movie_name[:28]:<30}{description_snippet:<50}{predicted_genre}")
    else:
        print("Data loading failed. Please check the CSV file path.")

# Hardcoded file paths for training and testing data
training_data_file = 'train_data.csv'
prediction_data_file = 'test_data.csv'

# Train the model
trained_model, trained_vectorizer = train_model(training_data_file)

# Check if training was successful
if trained_model is not None and trained_vectorizer is not None:
    # Predict genres for the second CSV file
    display_predictions(prediction_data_file, trained_model, trained_vectorizer)
else:
    print("Model training failed. Please check the training data file.")
