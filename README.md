# Reseach-paper-Q-And-A
A Streamlit-based web application that allows users to upload research papers in PDF format and ask questions related to the content of the paper. The application processes the uploaded PDF, indexes it using FAISS, and answers user queries using the Groq API and a language model.

## Features

- Upload and process research papers in PDF format.
- Split documents into chunks for better indexing.
- Use FAISS for document retrieval and Groq API for generating answers.
- Ask questions and get answers based on the context of the uploaded paper.
- Built with Streamlit for a simple web interface.

## Technologies Used

- **Streamlit**: For creating the web app interface.
- **LangChain**: For document processing and language model integration.
- **FAISS**: For efficient vector-based document retrieval.
- **Groq API**: For generating answers using a language model.
- **Transformers (Hugging Face)**: For pre-trained embeddings and language models.
