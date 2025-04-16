#Instruction to run the app 
# Step 1: Create a virtual Environment for python.
# Step 2: install dependencies by running."pip install -r requirements.txt"
# Step 3: Install Ollama locally and run "ollama run llama3.2"
# step 4: start the app by "streamlit run app.py"
# Step 5: Load the documnet submit and process then ask questions about it.
# Thank you 
# Mayank Rajput.


import streamlit as st
import os
import io
from dotenv import load_dotenv

# Try to use pdfplumber for better extraction. If not available, use PyPDF2.
try:
    import pdfplumber
except ImportError:
    pdfplumber = None
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.llms import Ollama
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

load_dotenv()

def get_pdf_text(pdf_docs):
    all_text = ""
    for pdf in pdf_docs:
        pdf_stream = io.BytesIO(pdf.read())
        if pdfplumber:
            with pdfplumber.open(pdf_stream) as pdf_file:
                for i, page in enumerate(pdf_file.pages):
                    page_text = page.extract_text()
                    if page_text:
                        
                        all_text += f"\n--- Page {i+1} ---\n" + page_text + "\n"
        else:
            pdf_reader = PdfReader(pdf_stream)
            for i, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text:
                    all_text += f"\n--- Page {i+1} ---\n" + page_text + "\n"
    return all_text

def get_text_chunks(text):
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  
        chunk_overlap=200, 
        separators=["\n\n", "\n", " ", ""]
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    
def get_conversational_chain():
    
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    Make sure to provide all the details; if the answer is not in the provided context, 
    just say: "answer is not available in the context". Do not provide an incorrect answer.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """
    
    model = Ollama(model="llama3.2:latest", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    docs = new_db.similarity_search(user_question)
    chain = get_conversational_chain()
    response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
    st.write("Reply: ", response["output_text"])

def main():
    st.set_page_config(page_title="Chat with Multiple PDFs")
    st.header("Chat with Multiple PDFs using Ollama")
    
    user_question = st.text_input("Ask a question from the PDF files")
    if user_question:
        user_input(user_question)
        
    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF files", type=["pdf"], accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing PDFs..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("PDF processing completed!")
                
if __name__ == "__main__":
    main()
