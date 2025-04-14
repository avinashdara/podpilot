from dotenv import load_dotenv
import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import chromadb

load_dotenv()

# Initialize embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize ChromaDB
collection_name = "competitor_analysis"
persist_directory = "chroma_db"

# Create vector store
vectorstore = Chroma(
    collection_name=collection_name,
    embedding_function=embedding_model,
    persist_directory=persist_directory
)

def store_report(report_text, week_id):
    vectorstore.add_texts(
        texts=[report_text],
        ids=[week_id],
        metadatas=[{"week": week_id}]
    )
    vectorstore.persist()  # Save to disk
    print(f"✅ Stored report for {week_id} in ChromaDB.")

def fetch_last_report():
    results = vectorstore.similarity_search(
        "latest competitor report",
        k=1
    )
    if results:
        print("✅ Historical report found.")
        return results[0].page_content
    else:
        print("❌ No historical report found.")
        return None 