from dotenv import load_dotenv
import os
from langchain_community.embeddings.openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
import pickle

load_dotenv()

# Initialize embeddings
embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

# Initialize or load FAISS index
index_file = "faiss_index"
if os.path.exists(index_file):
    vectorstore = FAISS.load_local(index_file, embedding_model)
    print("✅ Loaded existing FAISS index.")
else:
    vectorstore = FAISS.from_texts(
        texts=["Initial empty index"],
        embedding=embedding_model,
        metadatas=[{"week": "initial"}]
    )
    print("✅ Created new FAISS index.")

def store_report(report_text, week_id):
    vectorstore.add_texts(
        texts=[report_text],
        metadatas=[{"week": week_id}]
    )
    # Save to disk
    vectorstore.save_local(index_file)
    print(f"✅ Stored report for {week_id} in FAISS.")

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