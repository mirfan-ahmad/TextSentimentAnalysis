from transformers import pipeline
import gradio as gr


model = pipeline("text-classification", "ProsusAI/finbert")
def predict(text):
    try:
        result = model(text)
        return result[0]['label']
    except Exception as e:
        return str(e)

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(placeholder="Enter your text", lines=10),
    outputs="text",
    title="Text Sentiment Analysis",
    description="Enter text to analyze its sentiment (positive/negative/neutral)"
)

demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
    share=False,
    debug=True
)