from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split

# Sample Movie Data (replace with your actual data)
data = [
    {"ID": 1, "Name": "The Godfather", "Description": "Vito Corleone, the head of a powerful New York Mafia clan, tries to protect his family from rival gangs"},
    {"ID": 2, "Name": "The Shawshank Redemption", "Description": "Andy Dufresne, a seemingly ordinary man, is thrown into prison for the murder of his wife, a crime he didn't commit"},
    {"ID": 3, "Name": "The Dark Knight", "Description": "Mafia boss,  The Joker, throws Gotham into chaos by introducing fear among the citizens"},
    {"ID": 4, "Name": "The Lord of the Rings: The Fellowship of the Ring", "Description": "A hobbit named Frodo inherits the One Ring and embarks on a quest to destroy it"},
    {"ID": 5, "Name": "Star Wars: Episode IV - A New Hope", "Description": "A group of rebels fight against the evil Galactic Empire led by Darth Vader"},
    # Add an entry with missing description
    {"ID": 6, "Name": "Missing Movie", "Description": None},
]

# Separate features (Description) and target variable (Genre - Needs manual assignment)
descriptions = [movie["Description"] for movie in data if movie["Description"] is not None]  # Handle missing descriptions
genres = ["Crime", "Drama", "Thriller", "Fantasy", "Sci-Fi"]  # Assign genres manually (replace with predicted values later)


# Text Preprocessing (Optional: You can explore more advanced techniques)
def preprocess_text(text):
  # Lowercase text
  text = text.lower()
  # Remove punctuation
  # ... (add more steps as needed)
  return text

descriptions = [preprocess_text(text) for text in descriptions if text is not None]  # Handle missing descriptions

# Feature Engineering - TF-IDF Vectorization
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(descriptions)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(features, genres, test_size=0.2)

# Machine Learning Model - Multinomial Naive Bayes
model = MultinomialNB()
model.fit(X_train, y_train)

# Prediction Function (Replace with actual description for prediction)
def predict_genre(description):
  # Preprocess the description (if needed)
  description = preprocess_text(description)
  # Convert description to a vector
  new_feature = vectorizer.transform([description])
  # Predict genre
  predicted_genre = model.predict(new_feature)[0]
  return predicted_genre

# Print the data in a table format
print("ID\tName\t\t\t\t\t\tDescription\t\t\t\tGenre")
print("-"*100)
for i, movie in enumerate(data):
  # Handle missing descriptions (predict or assign default value)
  if movie["Description"] is None:
      predicted_genre = "Missing Description"  # Or predict using a different strategy
  else:
      predicted_genre = predict_genre(movie["Description"])
  # Truncate description with ellipsis (...)
  description_snippet = movie["Description"][:47] + "..." if len(movie["Description"]) > 47 else movie["Description"]
  print(f"{movie['ID']}\t{movie['Name'][:24]:<24}\t{description_snippet:<50}\t{predicted_genre}")
