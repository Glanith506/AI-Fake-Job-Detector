from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import joblib
import pytesseract
import re
import html

from PIL import Image
from io import BytesIO


# ==================================================
# TESSERACT LOCATION
# ==================================================

pytesseract.pytesseract.tesseract_cmd = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe"
)


# ==================================================
# CREATE FASTAPI APP
# ==================================================

app = FastAPI(
    title="AI Fake Job Detector API",
    description="Detect potentially fraudulent job postings using NLP and Machine Learning.",
    version="1.0"
)


# ==================================================
# CORS
# ==================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==================================================
# LOAD TRAINED MODEL
# ==================================================

model = joblib.load("fake_job_model.pkl")

tfidf = joblib.load("tfidf_vectorizer.pkl")


# ==================================================
# TEXT CLEANING
# ==================================================
# IMPORTANT:
# This is the SAME type of cleaning used during
# model training.
# ==================================================

def clean_text(text):

    text = str(text)

    # Convert HTML entities
    text = html.unescape(text)

    # Remove style sections
    text = re.sub(
        r"<style.*?>.*?</style>",
        " ",
        text,
        flags=re.DOTALL | re.IGNORECASE
    )

    # Remove script sections
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

    # Remove CSS measurements
    text = re.sub(
        r"\b\d+(?:\.\d+)?(?:px|pt|in|em|rem)\b",
        " ",
        text,
        flags=re.IGNORECASE
    )

    # Remove unnecessary symbols
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

    # Lowercase
    text = text.lower()

    return text.strip()


# ==================================================
# REQUEST FORMAT
# ==================================================

class JobRequest(BaseModel):
    job_description: str


# ==================================================
# FEATURE EXPLANATION
# ==================================================

def explain_prediction(cleaned_text):

    job_vector = tfidf.transform(
        [cleaned_text]
    )

    feature_names = tfidf.get_feature_names_out()

    coefficients = model.coef_[0]

    vector = job_vector.toarray()[0]

    contributions = (
        vector * coefficients
    )


    # ----------------------------------------------
    # Features pushing toward Fake
    # ----------------------------------------------

    fake_indices = (
        contributions
        .argsort()[-5:][::-1]
    )

    fake_words = []

    for index in fake_indices:

        if contributions[index] > 0:

            fake_words.append(
                feature_names[index]
            )


    # ----------------------------------------------
    # Features pushing toward Real
    # ----------------------------------------------

    real_indices = (
        contributions
        .argsort()[:5]
    )

    real_words = []

    for index in real_indices:

        if contributions[index] < 0:

            real_words.append(
                feature_names[index]
            )


    return {
        "fake_indicators": fake_words,
        "real_indicators": real_words
    }


# ==================================================
# COMMON ANALYSIS FUNCTION
# ==================================================

def analyze_job(job_text):

    # Clean input exactly like training data
    cleaned_text = clean_text(job_text)


    if len(cleaned_text) < 30:

        raise HTTPException(
            status_code=400,
            detail="Not enough useful text was found for analysis."
        )


    # Convert text into TF-IDF features
    job_vector = tfidf.transform(
        [cleaned_text]
    )


    # Binary model prediction
    prediction = model.predict(
        job_vector
    )[0]


    # Probability of class 1 = fake job
    probability = (
        model.predict_proba(job_vector)[0][1]
        * 100
    )


    # ----------------------------------------------
    # Risk level
    # ----------------------------------------------

    if probability < 40:

        risk_level = "Low"

    elif probability < 70:

        risk_level = "Medium"

    else:

        risk_level = "High"


    # ----------------------------------------------
    # Original ML class
    # ----------------------------------------------

    model_prediction = (
        "Fake Job"
        if prediction == 1
        else "Real Job"
    )


    # ----------------------------------------------
    # Safer user-facing assessment
    # ----------------------------------------------

    if probability < 40:

        assessment = "Low Fraud Risk"

    elif probability < 70:

        assessment = "Potentially Suspicious"

    else:

        assessment = "High Fraud Risk"


    # ----------------------------------------------
    # Explain model
    # ----------------------------------------------

    explanation = explain_prediction(
        cleaned_text
    )


    return {

        "prediction": model_prediction,

        "assessment": assessment,

        "fraud_risk": round(
            probability,
            2
        ),

        "risk_level": risk_level,

        "fake_indicators":
            explanation["fake_indicators"],

        "real_indicators":
            explanation["real_indicators"]
    }


# ==================================================
# HOME ROUTE
# ==================================================

@app.get("/")
def home():

    return {
        "message":
        "AI Fake Job Detector backend is running."
    }


# ==================================================
# TEXT PREDICTION ROUTE
# ==================================================

@app.post("/predict")
def predict_job(request: JobRequest):

    job_text = (
        request.job_description.strip()
    )


    if len(job_text) < 50:

        raise HTTPException(
            status_code=400,
            detail="Please provide a more complete job description."
        )


    return analyze_job(
        job_text
    )


# ==================================================
# IMAGE PREDICTION ROUTE
# ==================================================

@app.post("/predict-image")
async def predict_image(
    file: UploadFile = File(...)
):

    # ----------------------------------------------
    # Make sure file is an image
    # ----------------------------------------------

    if (
        not file.content_type
        or
        not file.content_type.startswith("image/")
    ):

        raise HTTPException(
            status_code=400,
            detail="Please upload a valid image file."
        )


    # ----------------------------------------------
    # Read uploaded image
    # ----------------------------------------------

    image_bytes = await file.read()


    try:

        image = Image.open(
            BytesIO(image_bytes)
        )


        # ------------------------------------------
        # OCR PREPROCESSING
        # ------------------------------------------

        # Convert to grayscale
        image = image.convert("L")


        # Increase image size when very small
        if image.width < 1200:

            scale = 2

            image = image.resize(
                (
                    image.width * scale,
                    image.height * scale
                )
            )


    except Exception:

        raise HTTPException(
            status_code=400,
            detail="Unable to read the uploaded image."
        )


    # ----------------------------------------------
    # OCR
    # ----------------------------------------------

    try:

        extracted_text = (
            pytesseract.image_to_string(
                image,
                config="--psm 6"
            )
        )

    except Exception:

        raise HTTPException(
            status_code=500,
            detail="OCR processing failed."
        )


    extracted_text = (
        extracted_text.strip()
    )


    # ----------------------------------------------
    # Check OCR output
    # ----------------------------------------------

    if len(extracted_text) < 50:

        raise HTTPException(
            status_code=400,
            detail=(
                "Not enough text could be extracted "
                "from the poster. Try a clearer image."
            )
        )


    # ----------------------------------------------
    # ML ANALYSIS
    # ----------------------------------------------

    result = analyze_job(
        extracted_text
    )


    # Keep original OCR text for frontend display
    result["extracted_text"] = (
        extracted_text
    )


    return result