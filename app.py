import streamlit as st
import PyPDF2
import io
import re
from typing import List, Tuple
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import logging
import os
from dotenv import load_dotenv
import openai

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize OpenAI client
api_key = os.getenv('OPENAI_API_KEY')
if api_key:
    try:
        openai.api_key = api_key
        logger.info("Successfully initialized OpenAI client")
    except Exception as e:
        logger.error(f"Error initializing OpenAI client: {str(e)}")
        st.error(f"Error initializing OpenAI client: {str(e)}")

# Initialize the sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Successfully loaded sentence transformer model")
except Exception as e:
    logger.error(f"Error loading sentence transformer model: {str(e)}")
    st.error(f"Error loading sentence transformer model: {str(e)}")

def create_faiss_index(chunks: List[str]) -> Tuple[faiss.IndexFlatIP, List[str]]:
    """
    Create a FAISS index from text chunks.
    
    Args:
        chunks: List of text chunks
        
    Returns:
        Tuple of (FAISS index, original chunks)
    """
    try:
        # Encode chunks into vectors
        vectors = model.encode(chunks)
        vectors = vectors.astype('float32')
        
        # Create and populate FAISS index
        dimension = vectors.shape[1]
        index = faiss.IndexFlatIP(dimension)  # Using Inner Product (cosine similarity)
        index.add(vectors)
        
        logger.info(f"Successfully created FAISS index with {len(chunks)} chunks")
        return index, chunks
    except Exception as e:
        logger.error(f"Error creating FAISS index: {str(e)}")
        st.error(f"Error creating FAISS index: {str(e)}")
        raise

def get_similar_chunks(query: str, index: faiss.IndexFlatIP, chunks: List[str], k: int = 3) -> List[Tuple[str, float]]:
    """
    Retrieve top k similar chunks for a given query.
    
    Args:
        query: User query string
        index: FAISS index containing chunk embeddings
        chunks: Original text chunks
        k: Number of similar chunks to retrieve
        
    Returns:
        List of tuples containing (chunk_text, similarity_score)
    """
    try:
        # Encode query
        query_vector = model.encode([query])
        query_vector = query_vector.astype('float32')
        
        # Search for similar chunks
        distances, indices = index.search(query_vector, k)
        
        # Get the similar chunks and their scores
        similar_chunks = []
        for idx, score in zip(indices[0], distances[0]):
            similar_chunks.append((chunks[idx], float(score)))
        
        logger.info(f"Found {len(similar_chunks)} similar chunks for query: {query}")
        return similar_chunks
    except Exception as e:
        logger.error(f"Error getting similar chunks: {str(e)}")
        st.error(f"Error getting similar chunks: {str(e)}")
        raise

def read_pdf(file):
    try:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        logger.info("Successfully read PDF file")
        return text
    except Exception as e:
        logger.error(f"Error reading PDF file: {str(e)}")
        st.error(f"Error reading PDF file: {str(e)}")
        raise

def clean_srt_content(content):
    try:
        # Split content into subtitle blocks
        blocks = re.split(r'\n\n+', content.strip())
        cleaned_blocks = []
        current_paragraph = []
        
        for block in blocks:
            # Split block into lines
            lines = block.strip().split('\n')
            if len(lines) < 3:  # Skip invalid blocks
                continue
            
            # Get only the text lines (skip subtitle number and timestamp)
            text_lines = []
            for line in lines:
                # Skip if line is a number (subtitle number)
                if line.strip().isdigit():
                    continue
                # Skip if line contains timestamp (matches pattern like 00:00:00,000 --> 00:00:00,000)
                if re.match(r'\d{2}:\d{2}:\d{2},\d{3}\s*-->\s*\d{2}:\d{2}:\d{2},\d{3}', line):
                    continue
                # Add non-timestamp, non-number lines
                text_lines.append(line)
            
            # Join the text lines
            text = ' '.join(text_lines)
            # Remove any remaining HTML tags or formatting
            text = re.sub(r'<[^>]+>', '', text)
            # Remove any special characters that might be left
            text = re.sub(r'[^\w\s.,!?-]', '', text)
            # Remove any extra whitespace
            text = ' '.join(text.split())
            
            if text:  # Only add non-empty text
                current_paragraph.append(text)
            
            # Group every 5 subtitle blocks into a paragraph
            if len(current_paragraph) >= 5:
                paragraph = ' '.join(current_paragraph)
                cleaned_blocks.append(paragraph)
                current_paragraph = []
        
        # Add any remaining paragraphs
        if current_paragraph:
            paragraph = ' '.join(current_paragraph)
            cleaned_blocks.append(paragraph)
        
        logger.info(f"Successfully cleaned SRT content with {len(cleaned_blocks)} paragraphs")
        return '\n\n'.join(cleaned_blocks)
    except Exception as e:
        logger.error(f"Error cleaning SRT content: {str(e)}")
        st.error(f"Error cleaning SRT content: {str(e)}")
        raise

def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences using regex pattern.
    Handles common sentence endings (.!?) and special cases.
    """
    try:
        # Split on sentence endings, but preserve them
        sentences = re.split(r'(?<=[.!?])\s+', text)
        # Filter out empty sentences
        result = [s.strip() for s in sentences if s.strip()]
        logger.info(f"Split text into {len(result)} sentences")
        return result
    except Exception as e:
        logger.error(f"Error splitting into sentences: {str(e)}")
        st.error(f"Error splitting into sentences: {str(e)}")
        raise

def split_into_chunks(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into chunks of approximately chunk_size words with overlap words between chunks.
    Ensures chunks don't split in the middle of sentences.
    
    Args:
        text: The input text to split
        chunk_size: Target number of words per chunk
        overlap: Number of words to overlap between chunks
        
    Returns:
        List of text chunks
    """
    try:
        # First split into sentences
        sentences = split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_word_count = 0
        
        for sentence in sentences:
            sentence_words = sentence.split()
            sentence_word_count = len(sentence_words)
            
            # If adding this sentence would exceed chunk size and we already have some content,
            # start a new chunk
            if current_word_count + sentence_word_count > chunk_size and current_chunk:
                # Join current chunk and add to chunks list
                chunks.append(' '.join(current_chunk))
                
                # Start new chunk with overlap
                if overlap > 0:
                    # Calculate how many words to take from the end of the current chunk
                    overlap_words = min(overlap, current_word_count)
                    # Get the last 'overlap_words' words from the current chunk
                    overlap_text = ' '.join(' '.join(current_chunk).split()[-overlap_words:])
                    current_chunk = [overlap_text]
                    current_word_count = len(overlap_text.split())
                else:
                    current_chunk = []
                    current_word_count = 0
            
            # Add sentence to current chunk
            current_chunk.append(sentence)
            current_word_count += sentence_word_count
        
        # Add the last chunk if it's not empty
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    except Exception as e:
        logger.error(f"Error splitting into chunks: {str(e)}")
        st.error(f"Error splitting into chunks: {str(e)}")
        raise

def read_srt(file):
    try:
        content = file.read().decode()
        logger.info("Successfully read SRT file")
        return clean_srt_content(content)
    except Exception as e:
        logger.error(f"Error reading SRT file: {str(e)}")
        st.error(f"Error reading SRT file: {str(e)}")
        raise

def generate_response(question: str, context_chunks: List[Tuple[str, float]]) -> str:
    """
    Generate a response using OpenAI's ChatCompletion based on the question and context.
    
    Args:
        question: User's question
        context_chunks: List of tuples containing (chunk_text, similarity_score)
        
    Returns:
        Generated response from OpenAI
    """
    try:
        # Format the context from chunks
        formatted_context = "\n\n".join([chunk[0] for chunk in context_chunks])
        
        # Create the prompt
        prompt = f"""You're an assistant helping users understand a podcast.
Using the following transcript context, answer the question clearly and quote if needed.

Context:
{formatted_context}

Question:
{question}

Please provide a clear and concise answer, using quotes from the transcript when relevant to support your response."""

        # Call OpenAI API
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that provides clear answers about podcast content, using quotes from the transcript when relevant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500
        )
        
        return response.choices[0]['message']['content']
        
    except Exception as e:
        logger.error(f"Error generating response: {str(e)}")
        return f"Error generating response: {str(e)}"

def main():
    st.title("Podcast Transcript Search and Q&A")
    st.write("Upload a transcript file (.txt, .pdf, or .srt) to search and ask questions about its content")

    # Check for OpenAI API key
    if not os.getenv('OPENAI_API_KEY'):
        st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
        return

    # File uploader
    uploaded_file = st.file_uploader("Choose a file", type=['txt', 'pdf', 'srt'])

    if uploaded_file is not None:
        try:
            # Display file details
            file_details = {
                "Filename": uploaded_file.name,
                "File size": f"{uploaded_file.size} bytes",
                "File type": uploaded_file.type
            }
            st.write("### File Details")
            for key, value in file_details.items():
                st.write(f"**{key}:** {value}")

            # Read and process file content
            content = None
            if uploaded_file.type == "text/plain":
                content = uploaded_file.read().decode()
            elif uploaded_file.type == "application/pdf":
                content = read_pdf(uploaded_file)
            elif uploaded_file.name.endswith('.srt'):
                content = read_srt(uploaded_file)
            
            if content is None:
                st.error("Failed to read file content")
                return
            
            # Split content into chunks
            chunks = split_into_chunks(content)
            
            # Create FAISS index
            index, chunks = create_faiss_index(chunks)
            
            # Display chunks
            with st.expander("View Transcript Chunks"):
                for i, chunk in enumerate(chunks, 1):
                    st.write(f"### Chunk {i}")
                    st.text_area("", chunk, height=200)
            
            # Add search and Q&A functionality
            st.write("### Ask Questions About the Podcast")
            query = st.text_input("Enter your question:")
            if query:
                # Get similar chunks
                similar_chunks = get_similar_chunks(query, index, chunks)
                
                # Generate response using OpenAI
                response = generate_response(query, similar_chunks)
                
                # Display response
                st.write("### Answer")
                st.write(response)
                
                # Display relevant chunks
                with st.expander("View Relevant Transcript Sections"):
                    st.write("### Most Relevant Chunks")
                    for i, (chunk, score) in enumerate(similar_chunks, 1):
                        st.write(f"#### Chunk {i} (Similarity Score: {score:.4f})")
                        st.text_area("", chunk, height=150)
                
        except Exception as e:
            logger.error(f"Error processing file: {str(e)}")
            st.error(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    main() 