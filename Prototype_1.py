from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
import string

# Sample Movie Data (replace with your actual data)
data = [
    {"ID": 1, "Name": "The Godfather", "Description": "Vito Corleone, the head of a powerful New York Mafia clan, tries to protect his family from rival gangs"},
    {"ID": 2, "Name": "The Shawshank Redemption", "Description": "Andy Dufresne, a seemingly ordinary man, is thrown into prison for the murder of his wife, a crime he didn't commit"},
    {"ID": 3, "Name": "The Dark Knight", "Description": "Mafia boss, The Joker, throws Gotham into chaos by introducing fear among the citizens"},
    {"ID": 4, "Name": "The Lord of the Rings: The Fellowship of the Ring", "Description": "A hobbit named Frodo inherits the One Ring and embarks on a quest to destroy it"},
    {"ID": 5, "Name": "Star Wars: Episode IV - A New Hope", "Description": "A group of rebels fight against the evil Galactic Empire led by Darth Vader"},
    {"ID": 6, "Name": "Missing Movie", "Description": None},
]

# Separate features (Description) and target variable (Genre - Needs manual assignment)
descriptions = [movie["Description"] for movie in data if movie["Description"] is not None]
genres = ["Crime", "Drama", "Thriller", "Fantasy", "Sci-Fi"]

# Text Preprocessing
def preprocess_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

descriptions = [preprocess_text(text) for text in descriptions if text is not None]

# Feature Engineering - TF-IDF Vectorization
vectorizer = TfidfVectorizer(stop_words='english')
features = vectorizer.fit_transform(descriptions)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, genres, test_size=0.2, random_state=42)

# Machine Learning Model - Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction Function
def predict_genre(description):
    if description is None:
        return "Unknown"  # Default for missing descriptions
    description = preprocess_text(description)
    new_feature = vectorizer.transform([description])
    predicted_genre = model.predict(new_feature)[0]
    return predicted_genre

# Print the data in a table format
print("ID\tName\t\t\t\t\t\tDescription\t\t\t\tGenre")
print("-" * 100)
for movie in data:
    description = movie["Description"]
    predicted_genre = predict_genre(description)
    description_snippet = description[:47] + "..." if description and len(description) > 47 else description
    print(f"{movie['ID']}\t{movie['Name'][:24]:<24}\t{description_snippet:<50}\t{predicted_genre}")

