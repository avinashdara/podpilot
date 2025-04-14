import streamlit as st
import os
from datetime import datetime
from dotenv import load_dotenv
import openai
from PyPDF2 import PdfReader
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def read_file(uploaded_file):
    """Read content from uploaded file (PDF, TXT, or SRT)"""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PdfReader(uploaded_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    else:  # TXT or SRT
        return uploaded_file.getvalue().decode("utf-8")

def process_transcript(text):
    """Split transcript into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def create_vector_store(chunks):
    """Create FAISS vector store from text chunks"""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_texts(chunks, embeddings)
    return vector_store

def generate_response(context_chunks, query):
    """Generate response using OpenAI API"""
    context = "\n\n".join(context_chunks)
    prompt = f"""Given the following podcast transcript excerpts, answer the question. 
    If the answer cannot be found in the excerpts, say "I cannot find information about that in the transcript."
    
    Transcript excerpts:
    {context}
    
    Question: {query}
    
    Answer:"""
    
    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=500
        )
        return response.choices[0].message['content']
    except Exception as e:
        st.error(f"Error generating response: {str(e)}")
        return None

# Initialize session state
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None

# App title
st.title("Podcast Transcript Search and Q&A")

# File upload
uploaded_file = st.file_uploader("Upload podcast transcript (PDF, TXT, or SRT)", type=["pdf", "txt", "srt"])

if uploaded_file:
    # Process the uploaded file
    text = read_file(uploaded_file)
    st.session_state.chunks = process_transcript(text)
    st.session_state.vector_store = create_vector_store(st.session_state.chunks)
    st.success("Transcript processed successfully!")

# Question input and response
if st.session_state.vector_store is not None:
    question = st.text_input("Ask a question about the podcast:")
    
    if st.button("Get Answer"):
        if question.strip():
            # Get relevant chunks
            relevant_docs = st.session_state.vector_store.similarity_search(question, k=3)
            context_chunks = [doc.page_content for doc in relevant_docs]
            
            # Generate response
            response = generate_response(context_chunks, question)
            
            if response:
                st.markdown("### Answer:")
                st.write(response)
                
                # Show relevant transcript chunks
                st.markdown("### Relevant Transcript Excerpts:")
                for i, chunk in enumerate(context_chunks, 1):
                    with st.expander(f"Excerpt {i}"):
                        st.write(chunk)
        else:
            st.warning("Please enter a question.") 