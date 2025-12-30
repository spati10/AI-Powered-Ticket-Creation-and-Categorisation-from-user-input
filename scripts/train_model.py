import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# CONFIG

DATA_PATH = "data/IT Support Ticket Data.csv"
TEXT_COL = "Issue"
CATEGORY_COL = "Category"
PRIORITY_COL = "Priority"

# TEXT PREPROCESSING

def preprocess(text):
    text = str(text).lower()
    text = re.sub(r"http\S+|www.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

# LOAD DATA

df = pd.read_csv(DATA_PATH)
df = df[[TEXT_COL, CATEGORY_COL, PRIORITY_COL]].dropna()
df.columns = ["text", "category", "priority"]

df["clean_text"] = df["text"].apply(preprocess)

print("\nCategory distribution (%):")
print(df["category"].value_counts(normalize=True) * 100)


# CATEGORY MODEL 

cat_encoder = LabelEncoder()
y_cat = cat_encoder.fit_transform(df["category"])

cat_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3
)

X_cat = cat_vectorizer.fit_transform(df["clean_text"])

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_cat,
    y_cat,
    test_size=0.2,
    random_state=42,
    stratify=y_cat
)

rf_category = RandomForestClassifier(
    n_estimators=500,
    max_depth=30,
    min_samples_leaf=2,
    class_weight="balanced",  
    random_state=42,
    n_jobs=-1
)

rf_category.fit(Xc_train, yc_train)

cat_preds = rf_category.predict(Xc_test)

print("\nCATEGORY MODEL RESULTS")
print("Accuracy:", accuracy_score(yc_test, cat_preds))
print("\nClassification Report:")
print(classification_report(yc_test, cat_preds, target_names=cat_encoder.classes_))

print("\nPrediction distribution:")
print(pd.Series(cat_encoder.inverse_transform(cat_preds)).value_counts())

joblib.dump(rf_category, "models/rf_category_model.pkl")
joblib.dump(cat_vectorizer, "models/category_vectorizer.pkl")
joblib.dump(cat_encoder, "models/category_label_encoder.pkl")


# PRIORITY MODEL 

pri_encoder = LabelEncoder()
y_pri = pri_encoder.fit_transform(df["priority"])

pri_vectorizer = TfidfVectorizer(
    max_features=20000,
    ngram_range=(1, 2),
    min_df=3
)

X_pri = pri_vectorizer.fit_transform(df["clean_text"])

Xp_train, Xp_test, yp_train, yp_test = train_test_split(
    X_pri,
    y_pri,
    test_size=0.2,
    random_state=42,
    stratify=y_pri
)

rf_priority = RandomForestClassifier(
    n_estimators=400,
    max_depth=25,
    min_samples_leaf=2,
    class_weight="balanced",
    random_state=42,
    n_jobs=-1
)

rf_priority.fit(Xp_train, yp_train)

pri_preds = rf_priority.predict(Xp_test)

print("\nPRIORITY MODEL RESULTS")
print("Accuracy:", accuracy_score(yp_test, pri_preds))
print("\nClassification Report:")
print(classification_report(yp_test, pri_preds, target_names=pri_encoder.classes_))

joblib.dump(rf_priority, "models/rf_priority_model.pkl")
joblib.dump(pri_vectorizer, "models/priority_vectorizer.pkl")
joblib.dump(pri_encoder, "models/priority_label_encoder.pkl")

print("\n Training completed and models saved.")
