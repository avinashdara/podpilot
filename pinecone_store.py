import pinecone
from dotenv import load_dotenv
import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import Pinecone as LangchainPinecone

load_dotenv()

# Initialize Pinecone
pinecone.init(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV")
)

index_name = "competitor-analysis"

# Create index if it doesn't exist
if index_name not in pinecone.list_indexes():
    pinecone.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine"
    )

# Get the index
index = pinecone.Index(index_name)
print("✅ Pinecone setup complete.")

embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = LangchainPinecone.from_existing_index(
    index_name=index_name,
    embedding=embedding_model,
    text_key="text"
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
