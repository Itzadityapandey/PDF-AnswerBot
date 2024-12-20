import gradio as gr
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import os
import numpy as np

# Load models
def initialize_models():
    qa_model_name = "tiiuae/falcon-7b-instruct"
    embedding_model_name = "all-MiniLM-L6-v2"
    
    # Question-answering model
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_pipeline = pipeline("text-generation", model=qa_model, tokenizer=qa_tokenizer)

    # Embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    
    return qa_pipeline, embedding_model

qa_pipeline, embedding_model = initialize_models()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    doc = fitz.open(pdf_file.name)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text

# Chunk text
def chunk_text(text, chunk_size=500):
    words = text.split()
    return [' '.join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]

# Find the most relevant chunk
def find_relevant_chunk(question, chunks, embedding_model):
    question_embedding = embedding_model.encode([question])[0]
    chunk_embeddings = embedding_model.encode(chunks)
    similarities = np.dot(chunk_embeddings, question_embedding)  # Faster similarity calculation
    best_chunk_index = np.argmax(similarities)
    return chunks[best_chunk_index]

# Answer the question
def answer_question_with_confidence(pdf_file, question):
    context = extract_text_from_pdf(pdf_file)
    chunks = chunk_text(context)
    relevant_chunk = find_relevant_chunk(question, chunks, embedding_model)
    
    # Get the answer using the QA pipeline
    response = qa_pipeline(question=question, context=relevant_chunk)
    answer = response["answer"]
    confidence = response.get("score", 0)
    return f"Answer: {answer}", f"Confidence: {confidence:.2%}"

# Create Gradio interface
def create_gradio_interface():
    interface = gr.Interface(
        fn=answer_question_with_confidence,
        inputs=[gr.File(label="Upload PDF", type="file", file_types=[".pdf"]),
                gr.Textbox(label="Ask a Question")],
        outputs=["text", "text"],
        title="Optimized PDF Q&A Bot",
        description="Upload a PDF and ask questions. This version is optimized for faster performance.",
    )
    return interface

# Launch Gradio
def launch_interface():
    interface = create_gradio_interface()
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)

launch_interface()
