from pinecone import Pinecone
from dotenv import load_dotenv
import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

load_dotenv()

# Initialize Pinecone with V3 API
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "competitor-analysis"

# First, delete the existing index if it exists (since we can't modify dimensions)
if index_name in pc.list_indexes().names():
    pc.delete_index(index_name)

# Create index with correct dimensions for OpenAI ada-002 model (1536)
pc.create_index(
    name=index_name,
    dimension=1536,  # OpenAI ada-002 embedding dimension
    metric="cosine",
    spec={
        "serverless": {
            "cloud": "gcp",
            "region": "us-central1"  # Free tier region
        }
    }
)

print("✅ Pinecone setup complete.")

embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding_model,
    text_key="text",
    namespace=""  # Use default namespace
)

def store_report(report_text, week_id):
    vectorstore.add_texts([report_text], ids=[week_id])
    print(f"✅ Stored report for {week_id} in Pinecone.")

def fetch_last_report():
    results = vectorstore.similarity_search("latest competitor report", k=1)
    if results:
        print("✅ Historical report found.")
        return results[0].page_content
    else:
        print("❌ No historical report found.")
        return None
