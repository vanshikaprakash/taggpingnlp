import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

# ---------------------------------
# Dataset
# ---------------------------------
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

# ---------------------------------
# TF-IDF
# ---------------------------------
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(df["question"])

# ---------------------------------
# Multi-label Encoding
# ---------------------------------
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(df["tags"])

print("Feature shape:", X.shape)
print("Label shape:", y.shape)
print("All possible tags:", mlb.classes_)

# ---------------------------------
# Train Model
# ---------------------------------
model = OneVsRestClassifier(
    LogisticRegression(max_iter=1000, C=10)
)

model.fit(X, y)

print("\nModel trained successfully!")

# ---------------------------------
# Predict New Question
# ---------------------------------
new_question = ["How to sort a list in python?"]
new_X = vectorizer.transform(new_question)

probs = model.predict_proba(new_X)[0]

print("\nProbabilities:\n", probs)

# Top 2 selection
top_k = 2
top_indices = np.argsort(probs)[-top_k:]

prediction = np.zeros_like(probs)
prediction[top_indices] = 1

# ✅ FIX: reshape correctly
prediction = prediction.reshape(1, -1)

predicted_tags = mlb.inverse_transform(prediction)

print("\nTest Question:", new_question[0])
print("Predicted Tags:", predicted_tags)
