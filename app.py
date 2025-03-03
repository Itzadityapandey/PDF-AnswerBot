import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)

# Directory to save FAISS index
INDEX_PATH = "faiss_index"

def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        try:
            pdf_reader = PdfReader(pdf.name)
            for page in pdf_reader.pages:
                extracted_text = page.extract_text()
                if extracted_text:
                    text += extracted_text
        except Exception as e:
            return f"Error reading PDF: {str(e)}"
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

def create_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
        vector_store.save_local(INDEX_PATH)  # Save to disk
        return "PDFs processed successfully! Vector store saved. Now you can ask questions."
    except Exception as e:
        return f"Error creating vector store: {str(e)}"

def load_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
        if os.path.exists(INDEX_PATH):
            return FAISS.load_local(INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
        return None
    except Exception as e:
        return None

def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, respond with "answer is not available in the context".
    Do not provide incorrect information.
    
    Context:
    {context}
    
    Question: 
    {question}
    
    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

def query_pdf(user_question):
    vector_store = load_vector_store()
    if vector_store is None:
        return "Please process a PDF first by uploading and submitting it."
    
    try:
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        return f"Error querying the PDF: {str(e)}"

def process_pdfs(pdf_files):
    if not pdf_files:
        return "Please upload at least one PDF."
    
    raw_text = get_pdf_text(pdf_files)
    if isinstance(raw_text, str) and "Error" in raw_text:
        return raw_text
    if not raw_text.strip():
        return "No extractable text found in the uploaded PDFs."
    
    text_chunks = get_text_chunks(raw_text)
    result = create_vector_store(text_chunks)
    return result

# Gradio UI
with gr.Blocks(title="Chat with PDF") as demo:
    gr.Markdown("## Chat with PDF 💁")
    pdf_input = gr.File(file_types=[".pdf"], label="Upload PDF(s)", file_count="multiple")
    process_button = gr.Button("Submit & Process")
    status_output = gr.Textbox(label="Status", placeholder="Status updates will appear here...")
    question_input = gr.Textbox(label="Ask a Question from the PDF")
    answer_output = gr.Textbox(label="Reply", placeholder="Answers will appear here...")
    ask_button = gr.Button("Get Answer")
    
    process_button.click(process_pdfs, inputs=[pdf_input], outputs=[status_output])
    ask_button.click(query_pdf, inputs=[question_input], outputs=[answer_output])

if __name__ == "__main__":
    demo.launch()
