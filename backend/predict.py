import joblib
import re
import html


# ==================================================
# LOAD MODEL
# ==================================================

model = joblib.load(
    "fake_job_model.pkl"
)

tfidf = joblib.load(
    "tfidf_vectorizer.pkl"
)


# ==================================================
# TEXT CLEANING
# ==================================================

def clean_text(text):

    text = str(text)

    text = html.unescape(text)

    text = re.sub(
        r"<style.*?>.*?</style>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"<script.*?>.*?</script>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text
    )

    text = re.sub(
        r"https?://\S+|www\.\S+",
        " ",
        text
    )

    text = re.sub(
        r"\S+@\S+",
        " ",
        text
    )

    text = re.sub(
        r"\b\d+(?:\.\d+)?(?:px|pt|in|em|rem)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"[^a-zA-Z0-9\s]",
        " ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    text = text.lower()

    return text.strip()


# ==================================================
# GET USER INPUT
# ==================================================

job_text = input(
    "\nEnter job description:\n\n"
)


# ==================================================
# VALIDATION
# ==================================================

if len(job_text.strip()) < 50:

    print(
        "\nPlease enter a more complete "
        "job description."
    )

    exit()


# ==================================================
# CLEAN TEXT
# ==================================================

cleaned_text = clean_text(
    job_text
)


# ==================================================
# TF-IDF
# ==================================================

job_vector = tfidf.transform(
    [cleaned_text]
)


# ==================================================
# PREDICTION
# ==================================================

prediction = model.predict(
    job_vector
)[0]


probability = (
    model.predict_proba(
        job_vector
    )[0][1]
    * 100
)


# ==================================================
# RISK LEVEL
# ==================================================

if probability < 40:

    assessment = "Low Fraud Risk"

    risk_level = "Low"


elif probability < 70:

    assessment = "Potentially Suspicious"

    risk_level = "Medium"


else:

    assessment = "High Fraud Risk"

    risk_level = "High"


# ==================================================
# DISPLAY RESULT
# ==================================================

print(
    "\n------------------------------"
)

print(
    "JOB ANALYSIS"
)

print(
    "------------------------------"
)


print(
    "Assessment:",
    assessment
)


print(
    "Fraud Risk:",
    round(probability, 2),
    "%"
)


print(
    "Risk Level:",
    risk_level
)


print(
    "\nNote: This is a machine-learning "
    "risk assessment and does not prove "
    "that a job is fraudulent."
)