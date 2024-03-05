# Retrieval Augmented Generation pdf chatbot
### Libraries used: Langchain, OpenAI, FAISS, Gradio
[Try the application here on Hugging Face Spaces](https://huggingface.co/spaces/Wintersmith/RAG-pdf-chatbot)

![chatbot_vystrizek](https://github.com/koldamartin/RAG_pdf_chatbot/assets/68967537/78d5c8d8-1ac3-4e71-8c21-47d1f9fbd636)

### **Overview:**
This is a Retrieval Augmented Generation (RAG) project. You upload your desired pdf file and then you can chat, ask questions, summarize the document etc.

The chatbot uses OpenAI GPT-3.5 model with temperature value 0. It has its own memory of previously asked questions so you can ask follow up questions without any reference. 

There are many ways to implement memory in Langchain. I used an Agent with a retrieval tool.


### **Features:**
**Gradio:** Simple graphical user interface with necessary buttons, textblocks etc. The code for the application is in app.py

**Document loader:** PyPDFLoad is used

**Document splitter:** I use recursive character splitter to split document into multiple chunks of max. 1000 tokens with slight overlap.

**Embeddings:** I use OpenAI Embeddings

**Vectorstore:** Vectors of embedded document are stored in a local FAISS library. (Facebook AI Similarity Search)

**Retriever:** Two possible choices, either Basic retriever or Contextual compression retriever.

**Agent:** An agent specifically optimized for doing retrieval when necessary and also holding a conversation.

**Hosting:** The application is hosted for free on Hugging Face Spaces.

To run this app outside Hugging Face Spaces you need  to have a OpenAI API key stored in .env file.



### **References:**
https://github.com/mirabdullahyaser/Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit/tree/master
https://github.com/Niez-Gharbi/PDF-RAG-with-Llama2-and-Gradio/tree/master
https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents
https://python.langchain.com/docs/use_cases/question_answering/chat_history
