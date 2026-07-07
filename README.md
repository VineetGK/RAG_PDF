# 📄 PDF QA Agent (Chat with PDF)

A Streamlit application that allows users to upload PDF documents and ask questions about their content. This project uses a Retrieval-Augmented Generation (RAG) pipeline built with LangChain, FAISS, HuggingFace embeddings, and Google Gemini.

## ✨ Features
- **Upload Any PDF**: Easily upload your research papers, manuals, or documents.
- **Fast Retrieval**: Uses FAISS vector database for lightning-fast similarity search.
- **Accurate Answers**: Powered by Google's Gemini LLM to generate precise answers based *only* on the document context.
- **Local Embeddings**: Uses `all-MiniLM-L6-v2` via HuggingFace for efficient, free, and local text embeddings.

## 🚀 Live Demo
*[Insert your Streamlit Cloud App URL here once deployed]*

## 🛠️ Installation & Local Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/your-repo-name.git
   cd your-repo-name
   ```

2. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the Streamlit app:**
   ```bash
   streamlit run app.py
   ```

## 🔑 Usage
1. Open the app in your browser (usually `http://localhost:8501`).
2. Enter your **Google API Key** in the sidebar.
3. Upload a PDF file.
4. Type a question about the document and click **Get Answer**.

## 📚 Tech Stack
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Google Gemini API](https://ai.google.dev/)
- [FAISS](https://faiss.ai/)
- [HuggingFace Embeddings](https://huggingface.co/)
