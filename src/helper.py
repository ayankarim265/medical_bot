# ==========================================
# 📘 helper.py — Utility Functions for RAG Medical Chatbot
# ==========================================

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings

# ------------------------------------------------------
# 1️⃣ Load all PDFs from a given folder
# ------------------------------------------------------
def load_pdf_files(folder_path: str):
    """
    Loads all PDF files from the specified folder.

    Args:
        folder_path (str): Path to the folder containing PDF files.

    Returns:
        documents (List[Document]): Loaded documents.
    """
    loader = DirectoryLoader(
        folder_path,
        glob="*.pdf",
        loader_cls=PyPDFLoader
    )
    documents = loader.load()
    print(f"✅ Loaded {len(documents)} pages from PDFs in '{folder_path}'")
    return documents


# ------------------------------------------------------
# 2️⃣ Split the text into manageable chunks
# ------------------------------------------------------
def split_text_into_chunks(documents, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Splits documents into overlapping text chunks for embedding.

    Args:
        documents (List[Document]): The loaded documents.
        chunk_size (int): Size of each chunk.
        chunk_overlap (int): Overlap between chunks.

    Returns:
        text_chunks (List[Document]): Chunked text data.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    text_chunks = text_splitter.split_documents(documents)
    print(f"✅ Split into {len(text_chunks)} text chunks (size={chunk_size}, overlap={chunk_overlap})")
    return text_chunks


# ------------------------------------------------------
# 3️⃣ Load HuggingFace Embeddings
# ------------------------------------------------------
def get_huggingface_embeddings(model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
    """
    Downloads and initializes HuggingFace embeddings model.

    Args:
        model_name (str): HuggingFace sentence-transformer model.

    Returns:
        embeddings (HuggingFaceEmbeddings): Embedding model instance.
    """
    embeddings = HuggingFaceEmbeddings(model_name=model_name)
    print(f"✅ Loaded HuggingFace embedding model: {model_name}")
    return embeddings
