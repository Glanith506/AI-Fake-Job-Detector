import pandas as pd
import joblib
import re
import html

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ==================================================
# 1. TEXT CLEANING FUNCTION
# ==================================================

def clean_text(text):

    # Make sure value is a string
    text = str(text)

    # Convert HTML entities
    # Example: &amp; -> &
    text = html.unescape(text)

    # Remove <style> sections
    text = re.sub(
        r"<style.*?>.*?</style>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove <script> sections
    text = re.sub(
        r"<script.*?>.*?</script>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove HTML tags
    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    # Remove URLs
    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    # Remove CSS-like measurements
    # Examples:
    # 4pt
    # 10px
    # 0in
    # 2em
    text = re.sub(
        r"\b\d+(?:\.\d+)?(?:px|pt|in|em|rem)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Replace unnecessary symbols with spaces
    # Keep letters and numbers
    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    # Convert to lowercase
    text = text.lower()

    return text.strip()


# ==================================================
# 2. LOAD DATASET
# ==================================================

print("\nLoading dataset...")

df = pd.read_csv(
    "../dataset/fake_job_postings.csv"
)

print("Dataset loaded successfully.")

print(
    "Total rows:",
    len(df)
)


# ==================================================
# 3. TEXT COLUMNS
# ==================================================

text_columns = [
    "title",
    "company_profile",
    "description",
    "requirements",
    "benefits"
]


# ==================================================
# 4. HANDLE MISSING VALUES
# ==================================================

for column in text_columns:

    df[column] = df[column].fillna("")


print("\nMissing text values handled.")


# ==================================================
# 5. COMBINE JOB TEXT
# ==================================================

df["combined_text"] = (
    df["title"]
    + " "
    + df["company_profile"]
    + " "
    + df["description"]
    + " "
    + df["requirements"]
    + " "
    + df["benefits"]
)


print("Text columns combined successfully.")


# ==================================================
# 6. CLEAN TEXT
# ==================================================

print("\nCleaning job posting text...")


df["combined_text"] = (
    df["combined_text"]
    .apply(clean_text)
)


print(
    "Text cleaning completed."
)


# ==================================================
# 7. INPUT AND TARGET
# ==================================================

X = df["combined_text"]

y = df["fraudulent"]


print("\nClass distribution:")

print(
    y.value_counts()
)


# ==================================================
# 8. TRAIN / TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print(
    "\nTraining samples:",
    len(X_train)
)

print(
    "Testing samples:",
    len(X_test)
)


print(
    "\nTraining class distribution:"
)

print(
    y_train.value_counts()
)


# ==================================================
# 9. TF-IDF
# ==================================================
#
# ngram_range=(1, 2)
#
# Learns:
#
# Single words:
# hiring
# registration
#
# Two-word phrases:
# registration fee
# no experience
# aerospace engineering
#
# token_pattern removes purely numeric features
# such as:
#
# 00
# 10
# 000
#
# min_df=2 removes extremely rare features.
# ==================================================

print(
    "\nCreating TF-IDF features..."
)


tfidf = TfidfVectorizer(

    stop_words="english",

    ngram_range=(1, 2),

    max_features=10000,

    # Ignore features appearing in only one job
    min_df=2,

    # Ignore terms present in almost every document
    max_df=0.95,

    # Ignore purely numeric tokens
    token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z0-9]+\b",

    # Helps reduce dominance of very frequent words
    sublinear_tf=True
)


# Learn vocabulary from training data
X_train_tfidf = tfidf.fit_transform(
    X_train
)


# Transform test data
X_test_tfidf = tfidf.transform(
    X_test
)


print(
    "TF-IDF completed."
)


print(
    "Training shape:",
    X_train_tfidf.shape
)


print(
    "Testing shape:",
    X_test_tfidf.shape
)


# ==================================================
# 10. TRAIN LOGISTIC REGRESSION
# ==================================================

print(
    "\nTraining Logistic Regression model..."
)


model = LogisticRegression(

    max_iter=1000,

    # Dataset has far fewer fake jobs
    class_weight="balanced"
)


model.fit(
    X_train_tfidf,
    y_train
)


print(
    "Model training completed."
)


# ==================================================
# 11. MAKE PREDICTIONS
# ==================================================

y_pred = model.predict(
    X_test_tfidf
)


# ==================================================
# 12. ACCURACY
# ==================================================

accuracy = accuracy_score(
    y_test,
    y_pred
)


print(
    "\n----------------------------------"
)

print(
    "MODEL RESULTS"
)

print(
    "----------------------------------"
)


print(
    "\nAccuracy:",
    round(
        accuracy * 100,
        2
    ),
    "%"
)


# ==================================================
# 13. CLASSIFICATION REPORT
# ==================================================

print(
    "\nClassification Report:"
)


print(
    classification_report(

        y_test,

        y_pred,

        target_names=[
            "Real Job",
            "Fake Job"
        ]
    )
)


# ==================================================
# 14. CONFUSION MATRIX
# ==================================================

print(
    "\nConfusion Matrix:"
)


cm = confusion_matrix(
    y_test,
    y_pred
)


print(cm)


print(
    "\nConfusion Matrix Meaning:"
)


print(
    """
[[Real predicted Real, Real predicted Fake],
 [Fake predicted Real, Fake predicted Fake]]
"""
)


# ==================================================
# 15. SAVE MODEL
# ==================================================

joblib.dump(

    model,

    "fake_job_model.pkl"
)


# ==================================================
# 16. SAVE TF-IDF VECTORIZER
# ==================================================

joblib.dump(

    tfidf,

    "tfidf_vectorizer.pkl"
)


print(
    "\n----------------------------------"
)

print(
    "FILES SAVED SUCCESSFULLY"
)

print(
    "----------------------------------"
)


print(
    "fake_job_model.pkl"
)

print(
    "tfidf_vectorizer.pkl"
)


# ==================================================
# 17. SHOW SOME LEARNED BIGRAMS
# ==================================================

feature_names = (
    tfidf.get_feature_names_out()
)


phrases = [

    feature

    for feature in feature_names

    if " " in feature
]


print(
    "\nExample two-word phrases learned:"
)


print(
    phrases[:30]
)


print(
    "\nTraining finished successfully."
)