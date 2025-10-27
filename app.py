import gradio as gr
import time
from llm_helper import *

agent = create_agent()
file_uploaded = False

def run_agent(text, history, file=None):
    global agent, file_uploaded

    if file is None:
        # If user has removed a file
        file_uploaded = False

    if file is not None and file_uploaded is False:
        # If user has uploaded a file for the first time
        read_file(file)
        file_uploaded = True
        time.sleep(5)

    resp = get_llm_response(agent,text)

    return resp


# Create a gradio interface for chatbots
demo = gr.ChatInterface(
    fn = run_agent,
    additional_inputs=[
        gr.File(label="Upload a file")
    ],
    title="Chatbot Assisstant",
    description="Ask me anything or upload a document.",
    type="messages"
)


# Launch the interface
demo.launch()

delete_vector_db()