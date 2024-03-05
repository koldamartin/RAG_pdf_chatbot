from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.tools.retriever import create_retriever_tool
from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
import os
import gradio as gr

# The Agent retriever is based on: https://python.langchain.com/docs/use_cases/question_answering/conversational_retrieval_agents?ref=blog.langchain.dev
# The chat history is based on: https://python.langchain.com/docs/use_cases/question_answering/chat_history
# Inspired by https://github.com/Niez-Gharbi/PDF-RAG-with-Llama2-and-Gradio/tree/master
# Inspired by https://github.com/mirabdullahyaser/Retrieval-Augmented-Generation-Engine-with-LangChain-and-Streamlit/tree/master

class PDFChatBot:
    # Initialize the class with the api_key and the model_name
    def __init__(self, api_key):
        self.processed = False
        self.final_agent = None
        self.chat_history = []
        self.api_key = api_key
        self.llm = ChatOpenAI(openai_api_key=self.api_key, temperature=0, model_name="gpt-3.5-turbo-0125")

    # add  text to Gradio text block (not needed without Gradio)
    def add_text(self, history, text):
        if not text:
            raise gr.Error("Please enter text.")
        history.append((text, ''))
        return history

    # Load a pdf document with langchain textloader
    def load_document(self, file_name):
        loader = PyPDFLoader(file_name)
        raw_document = loader.load()
        return raw_document

    # Split the document
    def split_documents(self, raw_document):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                       chunk_overlap=100,
                                                       length_function=len,
                                                       is_separator_regex=False,
                                                       separators=["\n\n", "\n", " ", ""])
        chunks = text_splitter.split_documents(raw_document)
        return chunks

    # Embed the document with OpenAI Embeddings & store it to vectorstore
    def create_retriever(self, chunks):
        embedding_func = OpenAIEmbeddings(openai_api_key=self.api_key)
        # Create a new vectorstore from the chunks
        vectorstore = FAISS.from_documents(chunks, embedding_func)

        # Create a retriever
        basic_retriever = vectorstore.as_retriever()
        compressor = LLMChainExtractor.from_llm(self.llm)
        compression_retriever = ContextualCompressionRetriever(base_compressor=compressor,
                                                               base_retriever=basic_retriever)
        return basic_retriever  # or compression_retriever

    # Create an agent
    def create_agent(self, retriever):
        tool = create_retriever_tool(retriever,
                                     f"search_document",
                                     f"Searches and returns excerpts from the provided document.")
        tools = [tool]
        prompt = hub.pull("hwchase17/openai-tools-agent")
        agent = create_openai_tools_agent(self.llm, tools, prompt)
        self.final_agent = AgentExecutor(agent=agent, tools=tools)

    # Process files
    def process_file(self, file_name):
        documents = self.load_document(file_name)
        texts = self.split_documents(documents)
        db = self.create_retriever(texts)
        self.create_agent(db)
        print("Files successfully processed")

    # Generate a response and write to memory
    def generate_response(self, history, query, path):
        if not self.processed:
            self.process_file(path)
            self.processed = True
        result = self.final_agent.invoke({'input': query, 'chat_history': self.chat_history})['output']
        self.chat_history.extend((query, result))
        for char in result: # history argument and the subsequent code is only for the purpose of Gradio
            history[-1][1] += char
        return history, " "

