import streamlit as st
import google.generativeai as genai

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

st.title("RAG AI Chatbot")

genai.configure(api_key="AIzaSyBDJk6sWYLlLPg7a2yXByW60shnPPvMgbo")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type="pdf"
)

if uploaded_file is not None:

    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    text = ""

    for doc in docs:
        text += doc.page_content

    if text.strip() == "":
        st.error("This PDF is image/scan based")
        st.stop()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.create_documents([text])

    embedding_model = HuggingFaceEmbeddings(
        model_name="all-MiniLM-L6-v2"
    )

    db = FAISS.from_documents(
        chunks,
        embedding_model
    )

    query = st.text_input("Ask a question")

    if query:

        results = db.similarity_search(query, k=3)

        context = "\n".join(
            [doc.page_content for doc in results]
        )

        model = genai.GenerativeModel(
            "gemini-1.5-flash"
        )

        response = model.generate_content(
            f"""
            Context:
            {context}

            Question:
            {query}
            """
        )

        st.subheader("Answer")
        st.write(response.text)