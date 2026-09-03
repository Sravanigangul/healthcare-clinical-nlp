import gradio as gr  # type: ignore[import-not-found]

from app.clinical_nlp import analyze_note

def run_analysis(text):
    if not text or not text.strip():
        return {
           "error": "Please enter a clinical note."
        }

    return analyze_note(text)

demo = gr.Interface(
    fn=run_analysis,

    inputs=gr.Textbox(
        lines=12,
        label="Clinical Note",
        placeholder = "Paste a synthetic or de-identified clinical note here..."
    ),
    outputs=gr.JSON(
        label="Clinical NLP results"
    ),

    title="Clinical NLP Review Assistant",

    description=(
        "Educational prototype for structured medication extraction"
        "and rule-based drug interaction review"
    )

)


if __name__ == "__main__":
    demo.launch()
