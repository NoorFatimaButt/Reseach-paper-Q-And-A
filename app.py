import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
import tempfile

from groq import Groq

# Hardcoded Groq API Key
API_KEY = "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"
MODEL_NAME = "llama-3.2-3b-preview"

# Initialize Groq client with the API key
client = Groq(api_key=API_KEY)

# Function to load PDF and split into chunks
def load_and_split_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20)
    return text_splitter.split_documents(docs)

# Function to initialize FAISS index
def initialize_faiss(documents):
    embedding_model = HuggingFaceEmbeddings(model_name="all-mpnet-base-v2")
    db = FAISS.from_documents(documents, embedding=embedding_model)
    return db

# Function to query the Groq API using the Groq library
def query_groq_api(prompt):
    try:
        # Create the chat completion request
        completion = client.chat.completions.create(
            model=MODEL_NAME,  # Use your model here, e.g., "llama-3.2-1b-preview"
            messages=[{"role": "user", "content": prompt}],
            temperature=1,
            max_tokens=1024,
            top_p=1,
            stream=True,
            stop=None,
        )

        # Collect the response content in a stream
        answer = ""
        for chunk in completion:
            answer += chunk.choices[0].delta.content or ""
        
        return answer

    except Exception as e:
        return f"Error: {str(e)}"

# Function to load LLM
def load_llm():
    def groq_pipeline(prompt):
        response = query_groq_api(prompt)
        return response  # Directly return the answer from Groq API
    return groq_pipeline

# Function to get the answer from FAISS + LLM
def get_answer(question, db, llm):
    retriever = db.as_retriever()
    context_docs = retriever.get_relevant_documents(question)  # Corrected the function to use get_relevant_documents
    context = "\n".join(doc.page_content for doc in context_docs)
    
    prompt = f"""
    Answer the following question based only on the provided context.
    Think step by step before providing a detailed answer.
    <context>
    {context}
    </context>
    Question: {question}
    """
    
    # Get the answer from the Groq API using the updated query function
    return llm(prompt)  # llm is now a pipeline using query_groq_api

# Streamlit app
st.title('Research Paper Q&A App')

# Upload PDF file
uploaded_file = st.file_uploader("Upload your research paper", type=["pdf"])

if uploaded_file is not None:
    st.write("Processing the document...")
    documents = load_and_split_pdf(uploaded_file)
    
    # Initialize FAISS and LLM
    db = initialize_faiss(documents)
    llm = load_llm()
    
    # User input for the question
    question = st.text_input("Ask a question about the paper:")
    
    if question:
        st.write("Fetching the answer...")
        try:
            answer = get_answer(question, db, llm)
            st.write("Answer:", answer)
        except Exception as e:
            st.error(f"Error generating answer: {e}")
