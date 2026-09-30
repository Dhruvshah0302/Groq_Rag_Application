import streamlit as st 
import os
from langchain_groq import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import time

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv('GROQ_API_KEY')

if "vector" not in st.session_state:
    st.session_state.embeddings = HuggingFaceEmbeddings()
    st.session_state.loader = WebBaseLoader("https://fhir.epic.com/Documentation?docId=oauth2")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 200)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs[:50])
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
    
st.title("ChtGroq Application")
llm = ChatGroq(model = "meta-llama/llama-prompt-guard-2-22m", groq_api_key = groq_api_key)

prompt = ChatPromptTemplate("""
   
   Answer the question based on the provided context only. Please provide the most accurate response based on hte question
   <context>
   {context}                         
   </context>
   Questions: {input}
"""
)


document_chain = create_stuff_documents_chain(llm,prompt)
retiever = st.session_state.vectors.as_retriever()
retiever_chain = create_retriveal_Chain(retiever, document_chain)

app_prompt = st.text_input("Enter your prompt here")

if app_prompt:
    start = time.process_time()
    response = retiever_chain.invoke({"input":app_prompt})
    print("reponse time :", time.process_time() - start)
    st.write(response['answer'])
    
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("---------------------------------")
            
            


