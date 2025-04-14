from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

index_name = "competitor-analysis"

if index_name not in pc.list_indexes().names():
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec={"cloud": "aws", "region": os.getenv("PINECONE_ENV")}
    )

index = pc.Index(index_name)
print("✅ Pinecone v3 setup complete.")


embedding_model = OpenAIEmbeddings(openai_api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = LangchainPinecone(
    index,
    embedding_model,
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
