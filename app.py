from transformers import pipeline
import gradio as gr


model = pipeline("text-classification", "ProsusAI/finbert")
def predict(text):
    try:
        result = model(text)
        return result[0]['label']
    except Exception as e:
        return str(e)

with gr.Blocks() as demo:
    textbox = gr.Textbox(placeholder="Enter your text", lines=10)
    demo = gr.Interface(fn=predict, inputs=textbox, outputs="text")
    
demo.launch()
