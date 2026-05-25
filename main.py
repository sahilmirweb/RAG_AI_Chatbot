from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

loader = PyPDFLoader("data/notes-1.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=100,
    chunk_overlap=20
)

chunks = splitter.split_documents(docs)

embedding_model = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

db = FAISS.from_documents(
    chunks,
    embedding_model
)

while True:
    query = input("\nAsk (type exit to stop): ")

    if query.lower() == "exit":
        print("Chatbot closed")
        break

    results = db.similarity_search(query)

    print("\nAnswer:")
    print(results[0].page_content)