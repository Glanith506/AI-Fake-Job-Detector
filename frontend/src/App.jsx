import { useState } from "react"
import "./App.css"

function App() {
  const [jobDescription, setJobDescription] = useState("")

  const [assessment, setAssessment] = useState("")
  const [fraudRisk, setFraudRisk] = useState(null)
  const [riskLevel, setRiskLevel] = useState("")

  const [fakeIndicators, setFakeIndicators] = useState([])
  const [realIndicators, setRealIndicators] = useState([])

  const [image, setImage] = useState(null)
  const [extractedText, setExtractedText] = useState("")

  const [loading, setLoading] = useState("")

  // ---------------------------------------------
  // Save API result
  // ---------------------------------------------

  function saveResult(data) {
    setAssessment(data.assessment || "")
    setFraudRisk(data.fraud_risk)
    setRiskLevel(data.risk_level || "")
    setFakeIndicators(data.fake_indicators || [])
    setRealIndicators(data.real_indicators || [])
  }

  // ---------------------------------------------
  // Analyze text
  // ---------------------------------------------

  async function handleCheck() {
    if (jobDescription.trim() === "") {
      alert("Please enter a job description.")
      return
    }

    if (jobDescription.trim().length < 50) {
      alert("Please enter a more complete job description.")
      return
    }

    setLoading("text")

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict",
        {
          method: "POST",

          headers: {
            "Content-Type": "application/json",
          },

          body: JSON.stringify({
            job_description: jobDescription,
          }),
        }
      )

      const data = await response.json()

      if (!response.ok) {
        alert(data.detail || "Something went wrong.")
        return
      }

      saveResult(data)

      setExtractedText("")
    } catch (error) {
      console.error(error)

      alert("Unable to connect to the backend.")
    } finally {
      setLoading("")
    }
  }

  // ---------------------------------------------
  // Select image
  // ---------------------------------------------

  function handleImageChange(event) {
    const file = event.target.files[0]

    if (file) {
      setImage(file)
    }
  }

  // ---------------------------------------------
  // Analyze poster
  // ---------------------------------------------

  async function handleImageUpload() {
    if (!image) {
      alert("Please select a job poster image.")
      return
    }

    const formData = new FormData()

    formData.append("file", image)

    setLoading("image")

    try {
      const response = await fetch(
        "http://127.0.0.1:8000/predict-image",
        {
          method: "POST",
          body: formData,
        }
      )

      const data = await response.json()

      if (!response.ok) {
        alert(data.detail || "Unable to analyze image.")
        return
      }

      saveResult(data)

      setExtractedText(data.extracted_text || "")
    } catch (error) {
      console.error(error)

      alert("Unable to connect to the backend.")
    } finally {
      setLoading("")
    }
  }

  // ---------------------------------------------
  // Clear everything
  // ---------------------------------------------

  function clearAll() {
    setJobDescription("")
    setAssessment("")
    setFraudRisk(null)
    setRiskLevel("")
    setFakeIndicators([])
    setRealIndicators([])
    setImage(null)
    setExtractedText("")
  }

  // ---------------------------------------------
  // Risk styling
  // ---------------------------------------------

  function getRiskClass() {
    if (fraudRisk === null) {
      return ""
    }

    if (fraudRisk >= 70) {
      return "high"
    }

    if (fraudRisk >= 40) {
      return "medium"
    }

    return "low"
  }

  return (
    <div className="app">

      {/* HEADER */}

      <header className="navbar">
        <div className="brand">
          <div className="brand-icon">AI</div>

          <div>
            <h2>Fake Job Detector</h2>
            <span>AI-powered fraud risk analysis</span>
          </div>
        </div>

        <div className="status">
          <span className="status-dot"></span>
          AI Model Ready
        </div>
      </header>


      {/* HERO */}

      <section className="hero">

        <div className="hero-badge">
          Machine Learning • NLP • OCR
        </div>

        <h1>
          Detect Suspicious Job
          <span> Postings with AI</span>
        </h1>

        <p>
          Analyze job descriptions or upload recruitment posters
          to identify patterns associated with fraudulent job postings.
        </p>

      </section>


      {/* MAIN INPUT AREA */}

      <main className="container">

        <div className="input-grid">

          {/* JOB DESCRIPTION CARD */}

          <section className="card">

            <div className="card-header">

              <div className="icon-box">
                📝
              </div>

              <div>
                <h2>Job Description</h2>

                <p>
                  Paste the complete job description below.
                </p>
              </div>

            </div>


            <textarea
              className="job-textarea"
              placeholder="Paste job description here..."
              value={jobDescription}
              onChange={(event) =>
                setJobDescription(event.target.value)
              }
            />


            <div className="textarea-footer">

              <span>
                {jobDescription.length} characters
              </span>

              <span>
                Minimum 50 characters
              </span>

            </div>


            <button
              className="primary-button"
              onClick={handleCheck}
              disabled={loading !== ""}
            >

              {loading === "text"
                ? "Analyzing..."
                : "Analyze Job"}

            </button>

          </section>


          {/* POSTER CARD */}

          <section className="card">

            <div className="card-header">

              <div className="icon-box">
                🖼️
              </div>

              <div>
                <h2>Job Poster</h2>

                <p>
                  Upload a recruitment poster for OCR analysis.
                </p>
              </div>

            </div>


            <label className="upload-area">

              <div className="upload-icon">
                ↑
              </div>

              <h3>
                Upload Job Poster
              </h3>

              <p>
                PNG, JPG or JPEG
              </p>

              <input
                type="file"
                accept="image/*"
                onChange={handleImageChange}
              />

            </label>


            {image && (

              <div className="selected-file">

                <span>✓</span>

                <div>
                  <strong>
                    Selected file
                  </strong>

                  <p>
                    {image.name}
                  </p>
                </div>

              </div>

            )}


            <button
              className="primary-button"
              onClick={handleImageUpload}
              disabled={loading !== ""}
            >

              {loading === "image"
                ? "Reading Poster..."
                : "Analyze Poster"}

            </button>

          </section>

        </div>


        {/* RESULT */}

        {fraudRisk !== null && (

          <section className="result-card">

            <div className="result-top">

              <div>

                <p className="result-label">
                  ANALYSIS RESULT
                </p>

                <h2>
                  {assessment}
                </h2>

              </div>


              <div
                className={`risk-badge ${getRiskClass()}`}
              >

                {riskLevel} Risk

              </div>

            </div>


            {/* RISK SCORE */}

            <div className="risk-section">

              <div className="risk-title">

                <span>
                  Fraud Risk Score
                </span>

                <strong>
                  {fraudRisk}%
                </strong>

              </div>


              <div className="progress-background">

                <div
                  className={`progress-bar ${getRiskClass()}`}
                  style={{
                    width: `${fraudRisk}%`,
                  }}
                ></div>

              </div>

            </div>


            {/* EXPLANATION */}

            <div className="indicator-grid">

              <div className="indicator-box suspicious">

                <h3>
                  ⚠ Features Toward Fake
                </h3>

                <div className="tags">

                  {fakeIndicators.length > 0 ? (

                    fakeIndicators.map(
                      (word, index) => (

                        <span
                          key={index}
                          className="tag fake-tag"
                        >

                          {word}

                        </span>

                      )
                    )

                  ) : (

                    <p>
                      No strong suspicious features found.
                    </p>

                  )}

                </div>

              </div>


              <div className="indicator-box legitimate">

                <h3>
                  ✓ Features Toward Real
                </h3>

                <div className="tags">

                  {realIndicators.length > 0 ? (

                    realIndicators.map(
                      (word, index) => (

                        <span
                          key={index}
                          className="tag real-tag"
                        >

                          {word}

                        </span>

                      )
                    )

                  ) : (

                    <p>
                      No strong legitimate features found.
                    </p>

                  )}

                </div>

              </div>

            </div>


            <div className="model-note">

              <strong>Important:</strong>

              This is a machine-learning risk assessment based
              on patterns learned from training data. It does
              not prove that a job posting is fraudulent or
              genuine.

            </div>


            <button
              className="secondary-button"
              onClick={clearAll}
            >

              Analyze Another Job

            </button>

          </section>

        )}


        {/* OCR TEXT */}

        {extractedText && (

          <section className="ocr-card">

            <div className="card-header">

              <div className="icon-box">
                🔎
              </div>

              <div>

                <h2>
                  Extracted Poster Text
                </h2>

                <p>
                  Text detected by Tesseract OCR.
                </p>

              </div>

            </div>


            <div className="ocr-text">

              {extractedText}

            </div>

          </section>

        )}

      </main>


      {/* FOOTER */}

      <footer>

        <p>
          AI Fake Job Detector • Machine Learning Project
        </p>

        <p>
          Python • Scikit-learn • FastAPI • React • Tesseract OCR
        </p>

      </footer>

    </div>
  )
}

export default App