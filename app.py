import os
from pathlib import Path

import numpy as np
import tensorflow as tf
import gradio as gr
from PIL import Image

# ============================================================
# CROPIQ / SLAMBOT
# MAIN WEB APPLICATION
# UI REDESIGN — MATCHES CROPIQ MAIN WEBSITE
# Brand accent: #307D18
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

# ============================================================
# MODEL PATHS
# ============================================================

CROP_MODEL_PATH = MODEL_DIR / "slambot_crop_classifier_best.keras"

CONDITION_MODELS = {
    "Cardamom": MODEL_DIR / "slambot_cardamom_condition_best.keras",
    "Ginger": MODEL_DIR / "slambot_ginger_condition_best.keras",
    "Maize": MODEL_DIR / "slambot_maize_condition_best.keras",
    "Turmeric": MODEL_DIR / "slambot_turmeric_condition_best.keras",
}

# ============================================================
# CLASS NAMES
# ============================================================

CROP_CLASSES = [
    "Cardamom",
    "Ginger",
    "Maize",
    "Turmeric",
]

CONDITION_CLASSES = {
    "Cardamom": [
        "Blight",
        "Healthy",
        "Phylosticta_LS_1000",
    ],
    "Ginger": [
        "Damage-Pest",
        "Dehydrated",
        "Healthy",
        "Leaf-blight",
    ],
    "Maize": [
        "Blight",
        "Common_Rust",
        "Gray_Leaf_Spot",
        "Healthy",
    ],
    "Turmeric": [
        "Dry Leaf",
        "Healthy Leaf",
        "Leaf Blotch",
        "Rhizome Disease Root",
        "Rhizome Healthy Root",
    ],
}

DISPLAY_NAMES = {
    "Phylosticta_LS_1000": "Phylosticta Leaf Spot",
    "Common_Rust": "Common Rust",
    "Gray_Leaf_Spot": "Gray Leaf Spot",
    "Leaf-blight": "Leaf Blight",
    "Damage-Pest": "Damage / Pest",
    "Dry Leaf": "Dry Leaf",
    "Healthy Leaf": "Healthy Leaf",
    "Leaf Blotch": "Leaf Blotch",
    "Rhizome Disease Root": "Rhizome Disease",
    "Rhizome Healthy Root": "Healthy Rhizome",
}

DESCRIPTIONS = {
    "Blight": "The model detected symptoms associated with blight.",
    "Phylosticta_LS_1000": "The model detected symptoms associated with Phylosticta leaf spot.",
    "Damage-Pest": "The model detected symptoms consistent with pest or physical damage.",
    "Dehydrated": "The model detected symptoms associated with dehydration or water stress.",
    "Leaf-blight": "The model detected symptoms associated with ginger leaf blight.",
    "Common_Rust": "The model detected symptoms associated with maize common rust.",
    "Gray_Leaf_Spot": "The model detected symptoms associated with maize gray leaf spot.",
    "Dry Leaf": "The model detected symptoms associated with turmeric dry leaf.",
    "Leaf Blotch": "The model detected symptoms associated with turmeric leaf blotch.",
    "Rhizome Disease Root": "The model detected symptoms associated with turmeric rhizome disease.",
}

PRECAUTIONS = {
    "Blight": [
        "Inspect surrounding plants for similar symptoms.",
        "Remove and properly manage severely affected plant material.",
        "Maintain good field sanitation.",
        "Avoid prolonged leaf wetness where practical.",
    ],
    "Phylosticta_LS_1000": [
        "Inspect nearby plants for similar leaf spots.",
        "Remove severely affected plant material where practical.",
        "Maintain good field sanitation.",
        "Monitor the crop regularly for disease spread.",
    ],
    "Damage-Pest": [
        "Inspect the plant carefully for insects or physical damage.",
        "Check nearby plants for similar symptoms.",
        "Maintain field sanitation.",
        "Monitor the crop regularly.",
    ],
    "Dehydrated": [
        "Check soil moisture.",
        "Provide appropriate irrigation.",
        "Avoid prolonged water stress.",
        "Monitor the plant for continued wilting or drying.",
    ],
    "Leaf-blight": [
        "Inspect nearby plants for similar symptoms.",
        "Remove severely affected leaves where practical.",
        "Maintain good field sanitation.",
        "Monitor the crop for disease spread.",
    ],
    "Common_Rust": [
        "Inspect surrounding maize plants for rust symptoms.",
        "Remove or manage severely affected material where practical.",
        "Maintain good crop management.",
        "Use locally recommended disease-management practices.",
    ],
    "Gray_Leaf_Spot": [
        "Inspect nearby maize plants for similar leaf symptoms.",
        "Maintain good field sanitation.",
        "Monitor affected leaves regularly.",
        "Use locally recommended disease-management practices.",
    ],
    "Dry Leaf": [
        "Check soil moisture and irrigation.",
        "Monitor surrounding turmeric plants.",
        "Remove severely affected material where practical.",
        "Maintain good field hygiene.",
    ],
    "Leaf Blotch": [
        "Inspect nearby plants for similar leaf symptoms.",
        "Remove severely affected plant material where practical.",
        "Maintain good field sanitation.",
        "Monitor the crop regularly for disease spread.",
    ],
    "Rhizome Disease Root": [
        "Inspect affected plants and surrounding areas.",
        "Remove severely affected material where practical.",
        "Avoid spreading contaminated soil or plant material.",
        "Use locally recommended disease-management practices.",
    ],
}

RECOMMENDATIONS = {
    "Blight": [
        "Monitor the crop regularly for disease spread.",
        "Maintain appropriate field hygiene and crop management.",
        "Use locally recommended disease-management practices.",
        "Consult a local agricultural expert if symptoms continue to spread.",
    ],
    "Phylosticta_LS_1000": [
        "Monitor affected leaves and surrounding plants.",
        "Maintain good crop and field management.",
        "Use locally recommended disease-management practices.",
        "Consult an agricultural expert if symptoms spread rapidly.",
    ],
    "Damage-Pest": [
        "Inspect the plant closely for the responsible pest.",
        "Monitor surrounding plants.",
        "Use locally recommended integrated pest-management practices.",
        "Seek local agricultural advice if damage increases.",
    ],
    "Dehydrated": [
        "Maintain appropriate irrigation.",
        "Check soil moisture regularly.",
        "Reduce avoidable water stress.",
        "Monitor the plant after correcting moisture conditions.",
    ],
    "Leaf-blight": [
        "Monitor affected leaves and surrounding plants.",
        "Maintain good field hygiene.",
        "Use locally recommended disease-management practices.",
        "Consult an agricultural expert if symptoms spread rapidly.",
    ],
    "Common_Rust": [
        "Monitor the crop regularly.",
        "Maintain good crop management.",
        "Use locally recommended rust-management practices.",
        "Consult an agricultural expert if disease pressure increases.",
    ],
    "Gray_Leaf_Spot": [
        "Monitor affected leaves and surrounding plants.",
        "Maintain good field management.",
        "Use locally recommended disease-management practices.",
        "Consult an agricultural expert if symptoms continue to spread.",
    ],
    "Dry Leaf": [
        "Monitor plant moisture conditions.",
        "Maintain appropriate irrigation.",
        "Inspect nearby plants for similar symptoms.",
        "Consult an agricultural expert if symptoms persist.",
    ],
    "Leaf Blotch": [
        "Monitor affected leaves and surrounding plants.",
        "Maintain good crop management.",
        "Use locally recommended disease-management practices.",
        "Consult an agricultural expert if symptoms spread.",
    ],
    "Rhizome Disease Root": [
        "Monitor surrounding plants carefully.",
        "Maintain good field hygiene.",
        "Avoid moving contaminated plant material.",
        "Consult an agricultural expert for persistent disease problems.",
    ],
}

# ============================================================
# LOAD MODELS
# ============================================================

print("=" * 70)
print("CROPIQ / SLAMBOT")
print("Loading models...")
print("=" * 70)

for path in [CROP_MODEL_PATH, *CONDITION_MODELS.values()]:
    if not path.exists():
        raise FileNotFoundError(f"Model not found:\n{path}")

crop_model = tf.keras.models.load_model(CROP_MODEL_PATH)

condition_models = {}
for crop, path in CONDITION_MODELS.items():
    print(f"Loading {crop} model...")
    condition_models[crop] = tf.keras.models.load_model(path)

print("\nAll models loaded successfully.")

# ============================================================
# IMAGE PREPROCESSING
# ============================================================

def preprocess_image(image):
    if image is None:
        return None

    image = image.convert("RGB")
    image = image.resize((224, 224))

    image = np.array(image, dtype=np.float32)
    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# HTML HELPERS
# ============================================================

def esc(value):
    """Minimal HTML escaping for model-generated labels."""
    return (
        str(value)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&#39;")
    )


def list_html(items):
    return "".join(
        f'<li>{esc(item)}</li>'
        for item in items
    )


# ============================================================
# PREDICTION
# ============================================================

def slambot_predict(image):

    if image is None:
        return """
        <div class="empty-state">
            <div class="empty-icon">＋</div>
            <div class="empty-title">No image selected</div>
            <div class="empty-text">
                Upload a crop image to begin detection.
            </div>
        </div>
        """

    try:
        image_array = preprocess_image(image)

        # ----------------------------------------------------
        # CROP PREDICTION
        # ----------------------------------------------------

        crop_predictions = crop_model.predict(
            image_array,
            verbose=0
        )[0]

        crop_index = int(np.argmax(crop_predictions))
        crop = CROP_CLASSES[crop_index]
        crop_confidence = float(crop_predictions[crop_index]) * 100

        # ----------------------------------------------------
        # CONDITION PREDICTION
        # ----------------------------------------------------

        condition_model = condition_models[crop]

        condition_predictions = condition_model.predict(
            image_array,
            verbose=0
        )[0]

        condition_index = int(np.argmax(condition_predictions))
        condition = CONDITION_CLASSES[crop][condition_index]
        condition_confidence = (
            float(condition_predictions[condition_index]) * 100
        )

        display_condition = DISPLAY_NAMES.get(
            condition,
            condition
        )

        # ----------------------------------------------------
        # CONFIDENCE
        # ----------------------------------------------------

        if condition_confidence >= 80:
            confidence_level = "HIGH"
            confidence_class = "high"
        elif condition_confidence >= 60:
            confidence_level = "MEDIUM"
            confidence_class = "medium"
        else:
            confidence_level = "LOW"
            confidence_class = "low"

        # ----------------------------------------------------
        # STATUS
        # ----------------------------------------------------

        healthy_conditions = {
            "Healthy",
            "Healthy Leaf",
            "Rhizome Healthy Root",
        }

        if condition in healthy_conditions:
            status = "HEALTHY"
            status_class = "healthy"

            description = (
                "The model did not detect a disease condition "
                "in the selected image."
            )

            precautions = [
                "Continue regular crop monitoring.",
                "Maintain good field sanitation.",
                "Maintain appropriate irrigation and crop management.",
            ]

            recommendations = [
                "Continue routine crop inspection.",
                "Maintain appropriate field hygiene.",
                "Monitor for any new symptoms.",
            ]

        else:
            status = "DISEASE / STRESS"
            status_class = "disease"

            description = DESCRIPTIONS.get(
                condition,
                "The model detected symptoms associated with this condition."
            )

            precautions = PRECAUTIONS.get(
                condition,
                [
                    "Inspect surrounding plants.",
                    "Maintain good field sanitation.",
                    "Monitor the crop regularly.",
                ]
            )

            recommendations = RECOMMENDATIONS.get(
                condition,
                [
                    "Monitor the crop regularly.",
                    "Maintain good crop management.",
                    "Consult a local agricultural expert if symptoms spread.",
                ]
            )

        return f"""
        <div class="result-shell">

            <div class="result-topline">
                <span class="result-dot"></span>
                <span>ANALYSIS COMPLETE</span>
                <span class="result-time">SLAMBOT INFERENCE</span>
            </div>

            <div class="result-hero">
                <div>
                    <div class="eyebrow">CROP DETECTED</div>
                    <div class="hero-value">{esc(crop)}</div>
                    <div class="hero-sub">
                        Crop confidence
                        <strong>{crop_confidence:.2f}%</strong>
                    </div>
                </div>

                <div class="condition-block">
                    <div class="eyebrow">CONDITION</div>
                    <div class="condition-value">
                        {esc(display_condition)}
                    </div>
                    <div class="hero-sub">
                        Condition confidence
                        <strong>{condition_confidence:.2f}%</strong>
                    </div>
                </div>
            </div>

            <div class="metrics-grid">

                <div class="metric-card">
                    <div class="metric-label">STATUS</div>
                    <div class="metric-value {status_class}">
                        {esc(status)}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">CONFIDENCE</div>
                    <div class="metric-value {confidence_class}">
                        {confidence_level}
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">CROP SCORE</div>
                    <div class="metric-number">
                        {crop_confidence:.1f}<span>%</span>
                    </div>
                </div>

                <div class="metric-card">
                    <div class="metric-label">CONDITION SCORE</div>
                    <div class="metric-number">
                        {condition_confidence:.1f}<span>%</span>
                    </div>
                </div>

            </div>

            <div class="field-notes">

                <div class="notes-column">
                    <div class="section-kicker">FIELD NOTES</div>
                    <h3>Assessment</h3>
                    <p>{esc(description)}</p>
                </div>

                <div class="notes-column">
                    <div class="section-kicker">PRECAUTIONS</div>
                    <h3>What to check</h3>
                    <ul>{list_html(precautions)}</ul>
                </div>

                <div class="notes-column">
                    <div class="section-kicker">RECOMMENDATIONS</div>
                    <h3>Next actions</h3>
                    <ul>{list_html(recommendations)}</ul>
                </div>

            </div>

            <div class="result-footer">
                <span>AI-ASSISTED CROP INTELLIGENCE</span>
                <span>VERIFY WITH FIELD OBSERVATION</span>
            </div>

        </div>
        """

    except Exception as e:
        return f"""
        <div class="error-state">
            <div class="eyebrow">SYSTEM ERROR</div>
            <h3>Analysis could not be completed</h3>
            <p>{esc(str(e))}</p>
        </div>
        """


# ============================================================
# CROPIQ WEBSITE CSS
# ============================================================

CUSTOM_CSS = r"""
:root {
    --crop-green: #307D18;
    --crop-green-light: #55a83a;
    --crop-green-soft: rgba(48, 125, 24, 0.16);
    --crop-black: #0b0d0b;
    --crop-panel: #121612;
    --crop-panel-2: #171b17;
    --crop-border: rgba(255,255,255,0.11);
    --crop-muted: #8d968d;
    --crop-text: #f2f4ef;
}

html,
body,
.gradio-container {
    background: var(--crop-black) !important;
    color: var(--crop-text) !important;
}

.gradio-container {
    max-width: 1440px !important;
    margin: 0 auto !important;
    padding: 0 !important;
    font-family:
        Inter,
        ui-sans-serif,
        system-ui,
        -apple-system,
        BlinkMacSystemFont,
        "Segoe UI",
        sans-serif !important;
}

/* Hide default Gradio footer */
footer {
    display: none !important;
}

/* Main wrapper */
#app-shell {
    background:
        radial-gradient(
            circle at 75% 12%,
            rgba(48,125,24,0.12),
            transparent 28%
        ),
        linear-gradient(
            180deg,
            #0a0c0a 0%,
            #0b0d0b 50%,
            #0d100d 100%
        );
    min-height: 100vh;
}

/* Header */
#top-nav {
    height: 74px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0 38px;
    border-bottom: 1px solid var(--crop-border);
    background: rgba(9,11,9,0.88);
    backdrop-filter: blur(18px);
}

.brand {
    display: flex;
    align-items: center;
    gap: 13px;
}

.brand-name {
    font-size: 20px;
    font-weight: 650;
    letter-spacing: -0.03em;
}

.brand-tag {
    border: 1px solid rgba(255,255,255,0.18);
    border-radius: 999px;
    padding: 5px 9px;
    color: #aeb5ae;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.nav-links {
    display: flex;
    gap: 9px;
    align-items: center;
}

.nav-link {
    color: #9ba39a;
    padding: 9px 15px;
    border-radius: 999px;
    font-size: 13px;
}

.nav-link.active {
    color: #f4f6f1;
    background: rgba(255,255,255,0.08);
}

.nav-status {
    display: flex;
    align-items: center;
    gap: 8px;
    color: #9ba39a;
    font-size: 10px;
    letter-spacing: 0.13em;
    text-transform: uppercase;
}

.status-led {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--crop-green-light);
    box-shadow: 0 0 13px rgba(85,168,58,0.7);
}

/* Hero */
#hero {
    padding: 82px 7vw 58px;
    position: relative;
    overflow: hidden;
}

.hero-grid {
    display: grid;
    grid-template-columns: minmax(0, 1.25fr) minmax(320px, 0.75fr);
    gap: 70px;
    align-items: end;
}

.eyebrow {
    color: var(--crop-green-light);
    font-size: 10px;
    font-weight: 700;
    letter-spacing: 0.22em;
    text-transform: uppercase;
}

.hero-title {
    margin: 15px 0 22px;
    max-width: 790px;
    font-size: clamp(52px, 7vw, 100px);
    line-height: 0.88;
    font-weight: 500;
    letter-spacing: -0.065em;
}

.hero-title .dim {
    color: #6f766f;
}

.hero-copy {
    max-width: 580px;
    color: #b0b7af;
    font-size: 17px;
    line-height: 1.55;
}

.telemetry {
    display: flex;
    gap: 42px;
    margin-top: 35px;
    padding-top: 19px;
    border-top: 1px solid var(--crop-border);
}

.telemetry-item span {
    display: block;
    color: #697169;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.telemetry-item strong {
    display: block;
    margin-top: 7px;
    color: #dce2da;
    font-size: 12px;
    font-weight: 500;
}

.scan-panel {
    min-height: 340px;
    border: 1px solid rgba(255,255,255,0.14);
    border-radius: 17px;
    background:
        linear-gradient(
            145deg,
            rgba(48,125,24,0.12),
            rgba(255,255,255,0.025)
        );
    box-shadow:
        0 28px 70px rgba(0,0,0,0.32),
        inset 0 1px 0 rgba(255,255,255,0.05);
    display: flex;
    align-items: center;
    justify-content: center;
    position: relative;
    overflow: hidden;
}

.scan-panel::before {
    content: "";
    position: absolute;
    inset: 20px;
    border: 1px dashed rgba(85,168,58,0.27);
    border-radius: 12px;
}

.scan-cross {
    position: absolute;
    width: 150px;
    height: 150px;
    border: 1px solid rgba(85,168,58,0.28);
    border-radius: 50%;
}

.scan-cross::before,
.scan-cross::after {
    content: "";
    position: absolute;
    background: rgba(85,168,58,0.28);
}

.scan-cross::before {
    width: 1px;
    height: 190px;
    top: -20px;
    left: 74px;
}

.scan-cross::after {
    height: 1px;
    width: 190px;
    left: -20px;
    top: 74px;
}

.scan-label {
    z-index: 2;
    text-align: center;
}

.scan-label .big {
    font-size: 13px;
    letter-spacing: 0.2em;
    text-transform: uppercase;
}

.scan-label .small {
    margin-top: 9px;
    color: #747d74;
    font-size: 11px;
}

/* Detector */
#detector {
    margin: 0 7vw;
    padding: 48px 0 85px;
    border-top: 1px solid var(--crop-border);
}

.detector-heading {
    display: flex;
    justify-content: space-between;
    align-items: end;
    gap: 30px;
    margin-bottom: 28px;
}

.detector-heading h2 {
    margin: 7px 0 0;
    font-size: 38px;
    line-height: 1;
    font-weight: 500;
    letter-spacing: -0.04em;
}

.detector-heading p {
    max-width: 420px;
    margin: 0;
    color: #818981;
    font-size: 13px;
    line-height: 1.5;
}

.upload-box {
    border: 1px solid rgba(255,255,255,0.13);
    border-radius: 18px;
    overflow: hidden;
    background: #101310;
}

/* Gradio image component */
.upload-box .image-container {
    background: #101310 !important;
}

.upload-box .upload-container {
    min-height: 360px !important;
    border: 0 !important;
    background:
        radial-gradient(
            circle,
            rgba(48,125,24,0.08),
            transparent 46%
        ) !important;
}

.upload-box .wrap {
    border: 0 !important;
}

/* Buttons */
#analyze-btn {
    border: 1px solid rgba(85,168,58,0.5) !important;
    background: var(--crop-green) !important;
    color: #f5f8f3 !important;
    border-radius: 9px !important;
    min-height: 54px !important;
    font-size: 13px !important;
    font-weight: 650 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    box-shadow: 0 10px 30px rgba(48,125,24,0.18) !important;
    transition: all 180ms ease !important;
}

#analyze-btn:hover {
    background: #3a9220 !important;
    transform: translateY(-1px);
    box-shadow: 0 13px 35px rgba(48,125,24,0.27) !important;
}

/* Results */
#result-box {
    margin-top: 28px;
}

.result-shell {
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 18px;
    overflow: hidden;
    background:
        linear-gradient(
            145deg,
            #141914,
            #0f120f
        );
}

.result-topline {
    height: 46px;
    padding: 0 22px;
    display: flex;
    align-items: center;
    gap: 9px;
    border-bottom: 1px solid var(--crop-border);
    color: #aab2aa;
    font-size: 9px;
    letter-spacing: 0.19em;
    text-transform: uppercase;
}

.result-dot {
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: var(--crop-green-light);
    box-shadow: 0 0 10px rgba(85,168,58,0.65);
}

.result-time {
    margin-left: auto;
    color: #596159;
}

.result-hero {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0;
    border-bottom: 1px solid var(--crop-border);
}

.result-hero > div {
    padding: 38px 30px;
}

.result-hero > div + div {
    border-left: 1px solid var(--crop-border);
}

.hero-value {
    margin-top: 10px;
    font-size: clamp(38px, 5vw, 64px);
    line-height: 0.95;
    font-weight: 500;
    letter-spacing: -0.055em;
}

.condition-value {
    margin-top: 13px;
    font-size: clamp(25px, 3vw, 42px);
    line-height: 1;
    font-weight: 500;
    letter-spacing: -0.045em;
}

.hero-sub {
    margin-top: 15px;
    color: #727b72;
    font-size: 11px;
}

.hero-sub strong {
    color: #cbd2ca;
    margin-left: 5px;
}

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    border-bottom: 1px solid var(--crop-border);
}

.metric-card {
    padding: 23px 24px;
    min-height: 100px;
    border-right: 1px solid var(--crop-border);
}

.metric-card:last-child {
    border-right: 0;
}

.metric-label {
    color: #687168;
    font-size: 9px;
    letter-spacing: 0.18em;
    text-transform: uppercase;
}

.metric-value {
    margin-top: 13px;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 0.03em;
}

.metric-value.healthy,
.metric-value.high {
    color: #72bb55;
}

.metric-value.disease {
    color: #d4a26d;
}

.metric-value.medium {
    color: #c6b55d;
}

.metric-value.low {
    color: #c87867;
}

.metric-number {
    margin-top: 11px;
    color: #e5e9e2;
    font-size: 24px;
    letter-spacing: -0.04em;
}

.metric-number span {
    color: #687168;
    font-size: 12px;
}

.field-notes {
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
}

.notes-column {
    padding: 30px;
    min-height: 230px;
    border-right: 1px solid var(--crop-border);
}

.notes-column:last-child {
    border-right: 0;
}

.section-kicker {
    color: var(--crop-green-light);
    font-size: 9px;
    font-weight: 700;
    letter-spacing: 0.19em;
}

.notes-column h3 {
    margin: 10px 0 13px;
    color: #e9ede7;
    font-size: 17px;
    font-weight: 500;
}

.notes-column p,
.notes-column li {
    color: #858e85;
    font-size: 12px;
    line-height: 1.65;
}

.notes-column ul {
    margin: 0;
    padding-left: 18px;
}

.notes-column li {
    margin-bottom: 7px;
}

.result-footer {
    display: flex;
    justify-content: space-between;
    padding: 14px 22px;
    border-top: 1px solid var(--crop-border);
    color: #4f574f;
    font-size: 8px;
    letter-spacing: 0.16em;
}

/* Empty / error */
.empty-state,
.error-state {
    padding: 50px 30px;
    border: 1px solid var(--crop-border);
    border-radius: 16px;
    background: var(--crop-panel);
    text-align: center;
}

.empty-icon {
    width: 48px;
    height: 48px;
    margin: 0 auto 15px;
    border: 1px solid rgba(85,168,58,0.35);
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    color: var(--crop-green-light);
    font-size: 25px;
}

.empty-title {
    color: #e7ebe4;
    font-size: 16px;
}

.empty-text,
.error-state p {
    margin-top: 8px;
    color: #737b73;
    font-size: 12px;
}

.error-state {
    text-align: left;
    border-color: rgba(190,90,70,0.3);
}

.error-state h3 {
    color: #e4e9e2;
    font-size: 20px;
    font-weight: 500;
}

/* Disclaimer */
.disclaimer {
    margin: 0 7vw;
    padding: 25px 0 50px;
    border-top: 1px solid var(--crop-border);
    display: flex;
    justify-content: space-between;
    gap: 30px;
    color: #5f675f;
    font-size: 10px;
    line-height: 1.6;
}

.disclaimer strong {
    color: #7d877d;
}

/* Responsive */
@media (max-width: 900px) {

    #top-nav {
        padding: 0 20px;
    }

    .nav-links {
        display: none;
    }

    #hero {
        padding: 55px 22px 40px;
    }

    .hero-grid {
        grid-template-columns: 1fr;
        gap: 35px;
    }

    .hero-title {
        font-size: clamp(48px, 14vw, 78px);
    }

    .scan-panel {
        min-height: 260px;
    }

    #detector {
        margin: 0 22px;
    }

    .detector-heading {
        display: block;
    }

    .detector-heading p {
        margin-top: 15px;
    }

    .result-hero,
    .field-notes {
        grid-template-columns: 1fr;
    }

    .result-hero > div + div,
    .notes-column {
        border-left: 0;
        border-right: 0;
        border-top: 1px solid var(--crop-border);
    }

    .metrics-grid {
        grid-template-columns: 1fr 1fr;
    }

    .metric-card:nth-child(2) {
        border-right: 0;
    }

    .metric-card:nth-child(3),
    .metric-card:nth-child(4) {
        border-top: 1px solid var(--crop-border);
    }

    .disclaimer {
        margin: 0 22px;
        display: block;
    }

    .disclaimer span {
        display: block;
        margin-top: 10px;
    }
}
"""

# ============================================================
# GRADIO UI
# ============================================================

with gr.Blocks(
    title="CropIQ — SLAMBOT",
) as demo:

    with gr.Column(elem_id="app-shell"):

        # ----------------------------------------------------
        # HEADER
        # ----------------------------------------------------

        gr.HTML(
            """
            <header id="top-nav">
                <div class="brand">
                    <div class="brand-name">CropIQ</div>
                    <div class="brand-tag">SLAMBOT</div>
                </div>

                <nav class="nav-links">
                    <div class="nav-link active">Detection</div>
                    <div class="nav-link">Systems</div>
                    <div class="nav-link">Telemetry</div>
                </nav>

                <div class="nav-status">
                    <span class="status-led"></span>
                    AI ONLINE
                </div>
            </header>
            """
        )

        # ----------------------------------------------------
        # HERO
        # ----------------------------------------------------

        gr.HTML(
            """
            <section id="hero">
                <div class="hero-grid">

                    <div>
                        <div class="eyebrow">
                            CROP INTELLIGENCE / SLAMBOT
                        </div>

                        <h1 class="hero-title">
                            Know what<br>
                            your crop is<br>
                            <span class="dim">telling you.</span>
                        </h1>

                        <p class="hero-copy">
                            AI-assisted crop identification and condition
                            detection for Cardamom, Ginger, Maize and
                            Turmeric — built as the intelligence layer
                            of the CropIQ system.
                        </p>

                        <div class="telemetry">

                            <div class="telemetry-item">
                                <span>CROPS</span>
                                <strong>04</strong>
                            </div>

                            <div class="telemetry-item">
                                <span>INFERENCE</span>
                                <strong>IMAGE AI</strong>
                            </div>

                            <div class="telemetry-item">
                                <span>MODE</span>
                                <strong>SLAMBOT</strong>
                            </div>

                            <div class="telemetry-item">
                                <span>STATUS</span>
                                <strong>ONLINE</strong>
                            </div>

                        </div>
                    </div>

                    <div class="scan-panel">
                        <div class="scan-cross"></div>

                        <div class="scan-label">
                            <div class="big">
                                CROP SCAN
                            </div>
                            <div class="small">
                                Upload a plant image below
                            </div>
                        </div>
                    </div>

                </div>
            </section>
            """
        )

        # ----------------------------------------------------
        # DETECTOR
        # ----------------------------------------------------

        with gr.Column(elem_id="detector"):

            gr.HTML(
                """
                <div class="detector-heading">
                    <div>
                        <div class="eyebrow">LIVE DETECTION</div>
                        <h2>Run a crop scan.</h2>
                    </div>

                    <p>
                        Upload a clear image of the affected plant or
                        leaf. SLAMBOT will identify the crop first,
                        then route the image to the corresponding
                        condition model.
                    </p>
                </div>
                """
            )

            with gr.Column(elem_classes="upload-box"):

                image_input = gr.Image(
                    type="pil",
                    label="",
                    show_label=False,
                    height=380,
                )

                analyze_button = gr.Button(
                    "Analyze Image",
                    variant="primary",
                    elem_id="analyze-btn",
                )

            result_output = gr.HTML(
                value="""
                <div class="empty-state">
                    <div class="empty-icon">＋</div>
                    <div class="empty-title">Ready for detection</div>
                    <div class="empty-text">
                        Upload an image and start a CropIQ scan.
                    </div>
                </div>
                """,
                elem_id="result-box",
            )

            analyze_button.click(
                fn=slambot_predict,
                inputs=image_input,
                outputs=result_output,
            )

        # ----------------------------------------------------
        # FOOTER
        # ----------------------------------------------------

        gr.HTML(
            """
            <div class="disclaimer">
                <div>
                    <strong>CROPIQ / SLAMBOT</strong><br>
                    AI-assisted crop intelligence for field
                    decision support.
                </div>

                <span>
                    AI predictions are decision-support outputs.
                    Confirm serious or rapidly spreading crop
                    problems with qualified agricultural expertise.
                </span>
            </div>
            """
        )


# ============================================================
# LAUNCH
# ============================================================

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        css=CUSTOM_CSS,
    )