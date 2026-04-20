from langchain_community.document_loaders import PyPDFLoader
from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_ollama import OllamaEmbeddings
import streamlit as st

embeddings = OllamaEmbeddings(model="llama3")

st.title("PDF Searcher!!")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file is not None:
    pdf = PyPDFLoader(uploaded_file, mode="single")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
    texts = text_splitter.split_text(pdf.extract_text())
    vectorstore = InMemoryVectorStore.from_texts(texts, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    st.write("PDF uploaded and processed successfully!")
    input_text = st.text_input("Enter the question:")
    if st.button("Search"):
         retrieved_documents = retriever.invoke(input_text)
         st.write("Search Results:")


    # vectorstore = InMemoryVectorStore.from_texts(texts, embedding=embeddings)
    # retriever = vectorstore.as_retriever()
    # st.write("PDF uploaded and processed successfully!")
    # input_text = st.text_input("Enter the question:")
    # if st.button("Search"):
    #      retrieved_documents = retriever.invoke(input_text)
    #      st.write("Search Results:")
    #      for doc in retrieved_documents:
    #          st.write(doc.page_content)


# text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
# texts = text_splitter.split_text(pdf.pages[0].extract_text())



# retrieved_documents = retriever.invoke("Give me M K Verma's timetable for 2026-27")
# print(retrieved_documents[0].page_content)