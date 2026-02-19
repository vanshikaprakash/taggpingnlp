import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Tiny dataset
data = {
    "question": [
        "How to reverse a list in python?",
        "How to center a div in CSS?",
        "How to join two tables in SQL?",
        "How to create a class in java?"
    ],
    "tags": [
        ["python", "list"],
        ["css", "html"],
        ["sql", "database"],
        ["java", "oop"]
    ]
}

df = pd.DataFrame(data)

# Create TF-IDF vectorizer
vectorizer = TfidfVectorizer()

# Convert text into numerical vectors
X = vectorizer.fit_transform(df["question"])

print("Shape of feature matrix:", X.shape)
