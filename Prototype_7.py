import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import string

# Sample Movie Data (replace with your actual data)
data = [
    {"ID": 1, "Name": "The Godfather", "Description": "Vito Corleone, the head of a powerful New York Mafia clan, tries to protect his family from rival gangs", "Genre": "Crime"},
    {"ID": 2, "Name": "The Shawshank Redemption", "Description": "Andy Dufresne, a seemingly ordinary man, is thrown into prison for the murder of his wife, a crime he didn't commit", "Genre": "Drama"},
    {"ID": 3, "Name": "The Dark Knight", "Description": "Mafia boss, The Joker, throws Gotham into chaos by introducing fear among the citizens", "Genre": "Thriller"},
    {"ID": 4, "Name": "The Lord of the Rings: The Fellowship of the Ring", "Description": "A hobbit named Frodo inherits the One Ring and embarks on a quest to destroy it", "Genre": "Fantasy"},
    {"ID": 5, "Name": "Star Wars: Episode IV - A New Hope", "Description": "A group of rebels fight against the evil Galactic Empire led by Darth Vader", "Genre": "Sci-Fi"},
]

# Convert to DataFrame
df = pd.DataFrame(data)

# Separate features (Description) and target variable (Genre)
descriptions = df['Description']
genres = df['Genre']

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

descriptions = descriptions.apply(preprocess_text)

# Feature Engineering - TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(descriptions)

# Train-Test Split (even though we're not using the test set in this example)
X_train, X_test, y_train, y_test = train_test_split(features, genres, test_size=0.2, random_state=42)

# Machine Learning Model - Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Function to predict genre
def predict_genre(description):
    if description is None:
        return "Unknown"  # Default for missing descriptions
    description = preprocess_text(description)
    new_feature = vectorizer.transform([description])
    predicted_genre = model.predict(new_feature)[0]
    return predicted_genre

# Function to get user input and display results
def get_user_input():
    user_data = []
    print("Enter movie details (type 'done' when finished):")
    while True:
        movie_id = input("ID: ").strip()
        if movie_id.lower() == 'done':
            break
        name = input("Name: ").strip()
        description = input("Description: ").strip()
        genre = predict_genre(description)
        user_data.append({"ID": movie_id, "Name": name, "Description": description, "Genre": genre})

    user_df = pd.DataFrame(user_data)
    
    # Align columns and print the DataFrame
    pd.set_option('display.colheader_justify', 'left')  # Left align column headers
    pd.set_option('display.width', 2000)  # Adjust the display width
    print("\nPredicted Movie Genres:")
    print(user_df.to_string(index=False))

# Get user input and display results
get_user_input()
