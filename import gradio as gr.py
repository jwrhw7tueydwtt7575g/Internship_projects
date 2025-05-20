import gradio as gr
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# -------------------- Document Extraction --------------------
def extract_text_from_file(file):
    text = ""
    filename = file.name if hasattr(file, "name") else file.filename
    if filename.endswith(".pdf"):
        pdf_reader = PdfReader(file)
        for page in pdf_reader.pages:
            content = page.extract_text()
            if content:
                text += content
    elif filename.endswith(".docx"):
        doc = Document(file)
        for para in doc.paragraphs:
            text += para.text + "\n"
    elif filename.endswith(".txt"):
        text += str(file)  # FIXED HERE
    return text

# -------------------- Text Chunking --------------------
def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_text(text)

# -------------------- Create Vector Store --------------------
def get_vector_store(chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    vector_store = FAISS.from_texts(chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# -------------------- Load Vector Store --------------------
def load_vector_store():
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    return FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)


# -------------------- LLM Chain Setup --------------------
def get_conversational_chain(model_name, groq_api_key):
    prompt_template = """
Answer the question as clearly and helpfully as possible using the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
    llm = ChatGroq(groq_api_key="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", model_name=model_name)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)
    return chain

# -------------------- Main Processing Functions --------------------
def process_documents(files, model_name, groq_api_key):
    if not files or not groq_api_key:
        return "❌ Please upload documents and provide your Groq API key."

    raw_text = ""
    for file in files:
        raw_text += extract_text_from_file(file)
    chunks = get_text_chunks(raw_text)
    get_vector_store(chunks)
    return "✅ Documents indexed successfully!"

def answer_question(question, model_name, groq_api_key):
    if not question or not groq_api_key:
        return "❌ Please enter a question and provide your Groq API key."

    try:
        vector_store = load_vector_store()
        docs = vector_store.similarity_search(question)
        chain = get_conversational_chain(model_name, groq_api_key)
        result = chain({"input_documents": docs, "question": question}, return_only_outputs=True)
        return result["output_text"]
    except Exception as e:
        return f"❌ Error: {e}"

# -------------------- Gradio Interface --------------------
with gr.Blocks() as demo:
    gr.Markdown("## 📄 AskMyDocs - Groq LLM Chatbot with RAG")

    groq_api_key = gr.Textbox(label="🔐 Enter your Groq API Key", type="password")
    model_option = gr.Dropdown(label="🧠 Choose LLM", choices=["mixtral-8x7b-32768", "llama3-70b-8192", "gemma-7b-it"], value="mixtral-8x7b-32768")
    uploaded_files = gr.File(label="📁 Upload Documents (PDF, DOCX, TXT)", file_types=[".pdf", ".docx", ".txt"], file_count="multiple")

    process_button = gr.Button("🔄 Process Documents")
    process_output = gr.Textbox(label="📌 Status", interactive=False)

    question = gr.Textbox(label="💬 Ask a question based on your documents:")
    answer_output = gr.Textbox(label="🤖 Answer", interactive=False)

    process_button.click(fn=process_documents,
                         inputs=[uploaded_files, model_option, groq_api_key],
                         outputs=process_output)

    question.submit(fn=answer_question,
                    inputs=[question, model_option, groq_api_key],
                    outputs=answer_output)

if __name__ == "__main__":
    demo.launch()