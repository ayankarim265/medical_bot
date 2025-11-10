# ==========================================
# 🧩 Vector Store Setup for Medical Chatbot (Pinecone + LangChain)
# ==========================================

from src.helper import load_pdf_files, split_text_into_chunks, get_huggingface_embeddings
from pinecone import Pinecone
from pinecone import ServerlessSpec
from langchain_community.vectorstores import Pinecone as PineconeVectorStore
from dotenv import load_dotenv
import os

# ==========================================
# 🔹 Load Environment Variables
# ==========================================
load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# ==========================================
# 📘 Step 1: Load and Process PDFs
# ==========================================
data_path = "data/"  # folder containing medical PDFs
extracted_data = load_pdf_files(folder_path=data_path)

# Split into smaller chunks for embeddings
text_chunks = split_text_into_chunks(extracted_data)

# Load embedding model
embeddings = get_huggingface_embeddings()  # returns MiniLM (384-dim)

# ==========================================
# 🧠 Step 2: Connect to Pinecone
# ==========================================
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medicalbot"

# Check if index already exists
existing_indexes = [i["name"] for i in pc.list_indexes()]
if index_name not in existing_indexes:
    print(f"🪶 Creating new Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # for MiniLM
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
else:
    print(f"✅ Pinecone index '{index_name}' already exists.")

# ==========================================
# ⚙️ Step 3: Create Vector Store and Upload Embeddings
# ==========================================
print("📤 Uploading text chunks to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("✅ Embeddings uploaded successfully to Pinecone index:", index_name)
