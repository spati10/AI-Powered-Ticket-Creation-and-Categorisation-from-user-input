import joblib

vec = joblib.load("models/category_vectorizer.pkl")
print("Vocabulary size:", len(vec.vocabulary_))
