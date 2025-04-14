# Podcast Transcript Search and Q&A

A Streamlit application that allows users to upload podcast transcripts (in TXT, PDF, or SRT format) and ask questions about the content. The app uses OpenAI's GPT-4 to generate responses based on relevant sections of the transcript.

## Features

- Support for multiple file formats (TXT, PDF, SRT)
- Semantic search using FAISS and OpenAI Embeddings
- Context-aware responses using OpenAI's GPT-4
- Interactive transcript chunk viewer
- Detailed answer generation with relevant quotes

## Setup

1. Clone the repository:
```bash
git clone <your-repository-url>
cd podcast-transcript-qa
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root and add your OpenAI API key:
```
OPENAI_API_KEY=your-api-key-here
```

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a transcript file (TXT, PDF, or SRT format)

4. Ask questions about the podcast content in the text input field

## How it Works

1. **File Processing**: The app reads and processes the uploaded transcript file, cleaning and formatting the content as needed.

2. **Chunking**: The transcript is split into manageable chunks while preserving context using RecursiveCharacterTextSplitter.

3. **Semantic Search**: When a question is asked, the app uses FAISS and OpenAI Embeddings to find the most relevant chunks of the transcript.

4. **Response Generation**: The relevant chunks are sent to OpenAI's GPT-4 along with the question to generate a comprehensive answer.

## Deployment

The app can be deployed on Streamlit Cloud:
1. Push your code to GitHub
2. Connect your repository to Streamlit Cloud
3. Add your OpenAI API key to Streamlit Cloud secrets

## Security Note

Make sure to never commit your `.env` file or expose your OpenAI API key in the code. 