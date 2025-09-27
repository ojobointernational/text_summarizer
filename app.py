import torch
import gradio as gr
from transformers import pipeline 

# Use a pipeline as a high-level helper
from transformers import pipeline


text_summary = pipeline("summarization", model="sshleifer/distilbart-cnn-12-6", torch_dtype=torch.bfloat16)

# text ='''Nigeria has been home to several indigenous material cultures, precolonial states and kingdoms since the second millennium BC'''
# print (text_summary(text))

def summary(input):
    output = text_summary(input)
    return output [0]['summary_text']

gr.close_all()

#demo = gr.Interface(fn=summary, input="text", output="text")

demo = gr.Interface(fn=summary, 
                    inputs =[gr.Textbox(label="Input text to summarize", lines=6)],
                    outputs=[gr.Textbox(label="summarized text", lines=4)], title ="@My GenAI Project 16: Text-summarizer",
                    description ="This application would be used to summarize text")
                                
demo.launch()
