"""
Customer Complain Classifier — Hugging Face Spaces app.

Standalone script: loads the fine-tuned model + tokenizer + label encoder
from the 'deployment_artifacts' folder that must sit alongside this file.
"""
import spaces
import os
import pickle
import tempfile
from datetime import datetime

import numpy as np
import torch
import gradio as gr
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
MAX_LEN = 150
ARTIFACTS_DIR = "."

# ------------------------------------------------------------------
# 1. Load model, tokenizer, and label encoder
# ------------------------------------------------------------------
loaded_tokenizer = AutoTokenizer.from_pretrained(ARTIFACTS_DIR)
loaded_model = AutoModelForSequenceClassification.from_pretrained(ARTIFACTS_DIR)
loaded_model.eval()

with open(f"{ARTIFACTS_DIR}/label_encoder.pkl", "rb") as f:
    loaded_label_encoder = pickle.load(f)

ALL_LABELS = list(loaded_label_encoder.classes_)

# In-memory session history (used for both the table and the downloadable report)
prediction_history = []


# ------------------------------------------------------------------
# 2. Core inference helper
# ------------------------------------------------------------------
@spaces.GPU
def _predict_probs(text: str) -> np.ndarray:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = loaded_model.to(device)
    inputs = loaded_tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_LEN)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.nn.functional.softmax(logits, dim=-1).cpu().numpy()[0]
    return probs


# ------------------------------------------------------------------
# 3. Single-complaint prediction
# ------------------------------------------------------------------
def gradio_predict(complaint_text):
    if not complaint_text or not complaint_text.strip():
        empty_df = pd.DataFrame(columns=["Complaint (preview)", "Predicted Category", "Confidence", "Timestamp"])
        return "—", {}, "Please enter a complaint narrative.", empty_df, gr.update(visible=False)

    probs = _predict_probs(complaint_text)
    pred_id = int(np.argmax(probs))
    pred_label = loaded_label_encoder.inverse_transform([pred_id])[0]
    confidence = float(probs[pred_id])

    label_scores = {ALL_LABELS[i]: float(probs[i]) for i in range(len(ALL_LABELS))}

    char_count = len(complaint_text)
    word_count = len(complaint_text.split())
    info_text = f"Length: {char_count} characters / {word_count} words"

    prediction_history.insert(0, {
        "Complaint (preview)": complaint_text[:60] + ("..." if len(complaint_text) > 60 else ""),
        "Predicted Category": pred_label,
        "Confidence": f"{confidence:.1%}",
        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "_full_text": complaint_text,
        "_all_scores": label_scores,
    })
    del prediction_history[20:]

    display_df = pd.DataFrame(prediction_history)[["Complaint (preview)", "Predicted Category", "Confidence", "Timestamp"]]

    return pred_label, label_scores, info_text, display_df, gr.update(visible=True)


# ------------------------------------------------------------------
# 4. Batch prediction
# ------------------------------------------------------------------
def gradio_predict_batch(batch_text):
    if not batch_text or not batch_text.strip():
        return pd.DataFrame(columns=["Complaint", "Predicted Category", "Confidence"])

    lines = [line.strip() for line in batch_text.split("\n") if line.strip()]
    rows = []
    for line in lines:
        probs = _predict_probs(line)
        pred_id = int(np.argmax(probs))
        pred_label = loaded_label_encoder.inverse_transform([pred_id])[0]
        confidence = float(probs[pred_id])
        rows.append({
            "Complaint": line[:80] + ("..." if len(line) > 80 else ""),
            "Predicted Category": pred_label,
            "Confidence": f"{confidence:.1%}",
        })
    return pd.DataFrame(rows)


# ------------------------------------------------------------------
# 5. Downloadable report generation
# ------------------------------------------------------------------
def generate_report():
    """Build a plain-text report of every prediction made in this session and
    return a file path so Gradio can offer it as a download."""
    if not prediction_history:
        return None

    lines = [
        "=" * 60,
        "CUSTOMER COMPLAIN CLASSIFIER — SESSION REPORT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Total predictions this session: {len(prediction_history)}",
        "=" * 60,
        "",
    ]

    for i, entry in enumerate(reversed(prediction_history), start=1):
        lines.append(f"--- Prediction #{i} ({entry['Timestamp']}) ---")
        lines.append(f"Complaint text: {entry['_full_text']}")
        lines.append(f"Predicted category: {entry['Predicted Category']}")
        lines.append(f"Confidence: {entry['Confidence']}")
        lines.append("Confidence breakdown by category:")
        for label, score in sorted(entry["_all_scores"].items(), key=lambda x: -x[1]):
            lines.append(f"    {label:<25} {score:.1%}")
        lines.append("")

    report_text = "\n".join(lines)

    tmp_path = os.path.join(tempfile.gettempdir(), "complaint_classification_report.txt")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(report_text)

    return tmp_path


# ------------------------------------------------------------------
# 6. Theme & styling — creative teal/purple gradient look
# ------------------------------------------------------------------
custom_theme = gr.themes.Soft(
    primary_hue=gr.themes.colors.teal,
    secondary_hue=gr.themes.colors.purple,
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "ui-sans-serif", "sans-serif"],
).set(
    button_primary_background_fill="linear-gradient(90deg, #14b8a6, #8b5cf6)",
    button_primary_background_fill_hover="linear-gradient(90deg, #0d9488, #7c3aed)",
    button_primary_text_color="white",
    block_border_width="1px",
    block_radius="16px",
    block_shadow="*shadow_drop_lg",
)

custom_css = """
.gradio-container {
    max-width: 1150px !important;
    margin: auto;
}
#hero {
    text-align: center;
    padding: 1.5em 1em 0.5em 1em;
}
#hero h1 {
    font-size: 2.4em;
    font-weight: 700;
    background: linear-gradient(90deg, #14b8a6, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1em;
}
#hero p {
    color: var(--body-text-color-subdued);
    font-size: 1.05em;
}
.category-card {
    border-radius: 16px;
}
"""

EXAMPLE_COMPLAINTS = [
    "I have been trying to dispute a charge on my credit card for weeks with no response.",
    "A debt collector keeps calling me multiple times a day about a debt I do not owe.",
    "My mortgage payment was applied incorrectly and now I am being charged a late fee.",
    "There is an error on my credit report that I have not been able to get corrected.",
    "My bank closed my checking account without any prior notice or explanation.",
]

# ------------------------------------------------------------------
# 7. Interface
# ------------------------------------------------------------------
with gr.Blocks(theme=custom_theme, css=custom_css, title="Customer Complain Classifier") as demo:
    gr.Markdown(
        """
        <div id="hero">
        <h1>🗂️ Customer Complain Classifier</h1>
        <p>Turn raw complaint narratives into clear, actionable categories — powered by a fine-tuned Transformer.</p>
        </div>
        """
    )

    with gr.Tabs():
        # ---------------- Tab 1: Single complaint ----------------
        with gr.Tab("🔍 Classify a Complaint"):
            with gr.Row():
                with gr.Column(scale=1):
                    complaint_input = gr.Textbox(
                        lines=6,
                        label="Complaint narrative",
                        placeholder="Describe the customer complaint here...",
                    )
                    with gr.Row():
                        submit_btn = gr.Button("✨ Classify", variant="primary")
                        clear_btn = gr.Button("🗑️ Clear")
                    gr.Examples(examples=EXAMPLE_COMPLAINTS, inputs=complaint_input, label="Try an example")

                with gr.Column(scale=1, elem_classes="category-card"):
                    predicted_label = gr.Label(label="Predicted Category", num_top_classes=1)
                    score_breakdown = gr.Label(label="Confidence by Category", num_top_classes=5)
                    length_info = gr.Textbox(label="Text stats", interactive=False)

            gr.Markdown("### 📊 Recent Predictions (this session)")
            history_table = gr.Dataframe(
                headers=["Complaint (preview)", "Predicted Category", "Confidence", "Timestamp"],
                interactive=False,
            )

            with gr.Row(visible=False) as report_row:
                report_btn = gr.Button("📥 Download Session Report", variant="secondary")
                report_file = gr.File(label="Report file", visible=False)

            submit_btn.click(
                fn=gradio_predict,
                inputs=complaint_input,
                outputs=[predicted_label, score_breakdown, length_info, history_table, report_row],
            )
            complaint_input.submit(
                fn=gradio_predict,
                inputs=complaint_input,
                outputs=[predicted_label, score_breakdown, length_info, history_table, report_row],
            )
            clear_btn.click(
                fn=lambda: ("", {}, "", gr.update(), gr.update(visible=False)),
                outputs=[predicted_label, score_breakdown, length_info, history_table, report_row],
            )
            report_btn.click(
                fn=generate_report,
                outputs=report_file,
            ).then(
                fn=lambda path: gr.update(visible=path is not None),
                inputs=report_file,
                outputs=report_file,
            )

        # ---------------- Tab 2: Batch classification ----------------
        with gr.Tab("📚 Batch Classification"):
            gr.Markdown("Paste one complaint per line to classify several at once.")
            batch_input = gr.Textbox(
                lines=8,
                label="Complaints (one per line)",
                placeholder="Complaint 1...\nComplaint 2...\nComplaint 3...",
            )
            batch_btn = gr.Button("✨ Classify All", variant="primary")
            batch_output = gr.Dataframe(
                headers=["Complaint", "Predicted Category", "Confidence"],
                interactive=False,
            )
            batch_btn.click(fn=gradio_predict_batch, inputs=batch_input, outputs=batch_output)

        # ---------------- Tab 3: About ----------------
        with gr.Tab("ℹ️ About"):
            gr.Markdown(
                f"""
                ### About this app

                **Model:** Fine-tuned Transformer (HuggingFace)
                **Categories:** {", ".join(ALL_LABELS)}

                This app predicts the most likely category for a customer complaint narrative,
                shows the confidence score across the top categories, supports classifying many
                complaints at once, and lets you download a full report of your session.
                """
            )

if __name__ == "__main__":
    demo.launch()
