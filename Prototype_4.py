import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB  # Using MultinomialNB here
from sklearn.model_selection import train_test_split  # Added for data splitting
from sklearn.pipeline import Pipeline  # Added for creating a pipeline

# Example training data (replace with your actual data)
train_data = """
ID,Name,Description,Genre
1,Oscar et la dame rose,...Drama
2,Cupid,...Thriller
... (more data with various genres)
"""

# Load training data
train_df = pd.read_csv(StringIO(train_data))

# Text preprocessing function
def preprocess_text(text):
  text = text.lower()
  text = text.translate(str.maketrans('', '', string.punctuation))
  return text

# Preprocess the descriptions in the training data
train_df['Description'] = train_df['Description'].apply(preprocess_text)

# Split data into training and testing sets (improves model generalizability)
X_train, X_test, y_train, y_test = train_test_split(train_df['Description'], train_df['Genre'], test_size=0.2, random_state=42)

# Define the text processing and classification pipeline
pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(stop_words='english')),
    ('clf', MultinomialNB())  # Using MultinomialNB here
])

# Train the model
pipeline.fit(X_train, y_train)

# Load prediction data from test_data.csv
test_data_path = 'test_data.csv'  # Change this to the path of your test_data.csv
if os.path.exists(test_data_path):
  predict_df = pd.read_csv(test_data_path)
else:
  print(f"The file {test_data_path} does not exist.")
  exit()

# Preprocess the descriptions in the prediction data
predict_df['Description'] = predict_df['Description'].apply(preprocess_text)

# Predict the genres
predictions = pipeline.predict(predict_df['Description'])

# Add predictions to the DataFrame
predict_df['Genre'] = predictions

# Display the prediction results
print(predict_df[['ID', 'Name', 'Genre']])

# You can now evaluate the model performance on the test set (y_test)
