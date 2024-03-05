import gradio as gr
import os
from dotenv import load_dotenv
from main_class import PDFChatBot

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
user_name = os.getenv("USERNAME")
password = os.getenv("PASSWORD")

pdf_chatbot = PDFChatBot(api_key)

with gr.Blocks(title="RAG chatbot", theme="Soft") as demo:

    def upload_file(file):
        return file

    gr.Markdown(
        """
        # Retrieval Augmented Generation app
    
        Use Langchain´s OpenAI agent with retrieval tool with a memory to chat with your pdf document.
        """
    )

    with gr.Column():

        with gr.Row():
            chat_history = gr.Chatbot(value=[], elem_id='chatbot', height=680)

    with gr.Row():

        with gr.Column(scale=1):
            file_output = gr.File()
            uploaded_pdf = gr.UploadButton("📁 Upload PDF", file_types=[".pdf"])
            uploaded_pdf.upload(upload_file, inputs=uploaded_pdf, outputs=file_output)

        with gr.Column(scale=2):
            text_input = gr.Textbox(
                show_label=False,
                placeholder="Type here to ask your PDF",
                container=False)

        with gr.Column(scale=1):
            submit_button = gr.Button('Send')
            submit_button.click(pdf_chatbot.add_text, inputs=[chat_history, text_input], outputs=[chat_history], queue=False).\
                success(pdf_chatbot.generate_response, inputs=[chat_history, text_input, uploaded_pdf], outputs=[chat_history, text_input])

if __name__ == '__main__':
    demo.queue()
    demo.launch(share=True)