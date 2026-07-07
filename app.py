import streamlit as st
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="PDF QA Agent", layout="wide")
st.title("📄 Upload & Chat with any PDF")

# Sidebar for API Key
api_key = st.sidebar.text_input("Google API Key", type="password", help="Enter your Google API Key here")
if api_key:
    os.environ['GOOGLE_API_KEY'] = api_key

# Main UI components
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
question = st.text_input("Ask a question about the uploaded paper:")

if st.button("Get Answer", type="primary"):
    if not os.environ.get('GOOGLE_API_KEY'):
        st.error("Please enter your Google API Key in the sidebar.")
    elif not uploaded_file:
        st.warning("Please upload a PDF document.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        with st.spinner("Processing PDF and searching for the answer..."):
            try:
                # 1. Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_pdf_path = tmp_file.name

                # 2. Extract Text & Chunk
                loader = PyPDFLoader(temp_pdf_path)
                documents = loader.load()
                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
                chunks = text_splitter.split_documents(documents)

                # 3. Build Vector Store & Retriever
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                vector_store = FAISS.from_documents(chunks, embeddings)
                retriever = vector_store.as_retriever(search_kwargs={"k": 3})

                # 4. Define LLM & RAG Chain
                llm = ChatGoogleGenerativeAI(model="gemma-4-31b-it", temperature=0)

                system_prompt = (
                    "You are an expert research assistant for question-answering tasks. "
                    "Use the following pieces of retrieved context to answer the question. "
                    "If you don't know the answer, say that you don't know. "
                    "Keep the answer clear, accurate, and concise.

"
                    "Context: {context}"
                )
                prompt = ChatPromptTemplate.from_messages([
                    ("system", system_prompt),
                    ("human", "{input}"),
                ])

                def format_docs(docs):
                    return "

".join(doc.page_content for doc in docs)

                rag_chain_from_docs = (
                    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"]))) 
                    | prompt 
                    | llm 
                    | StrOutputParser()
                )
                rag_chain = RunnableParallel(
                    {"context": retriever, "input": RunnablePassthrough()}
                ).assign(answer=rag_chain_from_docs)

                # 5. Generate Answer
                response = rag_chain.invoke(question)

                st.success("Done!")
                st.markdown("### 🤖 Answer")
                st.info(response["answer"])

            except Exception as e:
                st.error(f"An error occurred: {e}")
