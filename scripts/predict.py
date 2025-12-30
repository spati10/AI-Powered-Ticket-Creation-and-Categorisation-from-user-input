import joblib
from .clean_text import clean_text

cat_model = joblib.load("models/rf_category_model.pkl")
cat_vectorizer = joblib.load("models/category_vectorizer.pkl")
cat_encoder = joblib.load("models/category_label_encoder.pkl")

pri_model = joblib.load("models/rf_priority_model.pkl")
pri_vectorizer = joblib.load("models/priority_vectorizer.pkl")
pri_encoder = joblib.load("models/priority_label_encoder.pkl")

def predict_ticket(text):
    clean = clean_text(text)

    category = cat_encoder.inverse_transform(
        cat_model.predict(cat_vectorizer.transform([clean]))
    )[0]

    priority = pri_encoder.inverse_transform(
        pri_model.predict(pri_vectorizer.transform([clean]))
    )[0]

    # ❗ MUST RETURN VALUES
    return category, priority
