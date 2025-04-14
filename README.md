# Podcast Transcript Search and Q&A

A Streamlit application that allows users to upload podcast transcripts (in TXT, PDF, or SRT format) and ask questions about the content. The app uses OpenAI's GPT-4 to generate responses based on relevant sections of the transcript.

## Features

- Support for multiple file formats (TXT, PDF, SRT)
- Semantic search using FAISS and Sentence Transformers
- Context-aware responses using OpenAI's GPT-4
- Interactive transcript chunk viewer
- Detailed answer generation with relevant quotes

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd <repository-name>
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
```bash
# Copy the template environment file
cp .env.template .env

# Edit .env file with your actual API key
# IMPORTANT: Never commit your .env file to version control!
```

Then edit the `.env` file and replace `your_openai_api_key_here` with your actual OpenAI API key.

⚠️ **IMPORTANT: Security Notes**
- Never commit your `.env` file to version control
- Never share your API keys publicly
- The `.env` file is already in `.gitignore` to prevent accidental commits
- If you accidentally commit API keys, rotate them immediately

## Usage

1. Run the Streamlit app:
```bash
streamlit run app.py
```

2. Open your web browser and go to the URL shown in the terminal (usually http://localhost:8501)

3. Upload a transcript file (TXT, PDF, or SRT format)

4. Ask questions about the podcast content in the text input field

## Deployment on Streamlit Cloud

1. Push your code to GitHub (make sure your `.env` file is not included)
2. Go to [Streamlit Cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Add your OpenAI API key as a secret in the Streamlit Cloud dashboard:
   - Go to your app settings
   - Add a secret with the key `OPENAI_API_KEY`
   - Set its value to your API key

## How it Works

1. **File Processing**: The app reads and processes the uploaded transcript file, cleaning and formatting the content as needed.

2. **Chunking**: The transcript is split into manageable chunks while preserving sentence integrity.

3. **Semantic Search**: When a question is asked, the app uses FAISS and Sentence Transformers to find the most relevant chunks of the transcript.

4. **Response Generation**: The relevant chunks are sent to OpenAI's GPT-4 along with the question to generate a comprehensive answer.

## Requirements

- Python 3.8+
- OpenAI API key
- See requirements.txt for full list of dependencies

## License

MIT License 