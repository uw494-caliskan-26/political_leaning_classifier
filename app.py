import gradio as gr
from classifier import load_model, classify, LABEL_MAP

print("Loading model (4-bit quantized)...")
model, tokenizer, number_token_ids = load_model(quantize_4bit=True)
print("Model loaded. Starting Gradio server...")


def predict(text: str):
    if not text or not text.strip():
        return "Please enter some text.", None

    result = classify(text, model, tokenizer, number_token_ids)
    label_output = f"**{result['label']}** (score: {result['score']})"
    confidences = result["probabilities"]
    return label_output, confidences


with gr.Blocks(title="Political Leaning Classifier") as demo:
    gr.Markdown(
        "# Political Leaning Classifier\n"
        "Paste a news article or text passage below. The model will estimate its "
        "political leaning on a 5-point scale from **Far Left** to **Far Right**.\n\n"
        "Model: [zhezhou1106/political-leaning-classifier-v2]"
        "(https://huggingface.co/zhezhou1106/political-leaning-classifier-v2)"
    )

    with gr.Row():
        with gr.Column(scale=2):
            text_input = gr.Textbox(
                label="Input Text",
                placeholder="Paste article or text here...",
                lines=12,
            )
            submit_btn = gr.Button("Classify", variant="primary")

        with gr.Column(scale=1):
            label_output = gr.Markdown(label="Prediction")
            confidence_output = gr.Label(label="Class Probabilities", num_top_classes=5)

    submit_btn.click(fn=predict, inputs=text_input, outputs=[label_output, confidence_output])
    text_input.submit(fn=predict, inputs=text_input, outputs=[label_output, confidence_output])

    gr.Examples(
        examples=[
            ["The government must expand social safety nets and raise taxes on the wealthy to reduce inequality."],
            ["Both parties have valid points; pragmatic compromise is needed to move forward on infrastructure."],
            ["Excessive regulation stifles free enterprise. Tax cuts drive growth and individual liberty."],
        ],
        inputs=text_input,
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
