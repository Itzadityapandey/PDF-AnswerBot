import os
import gradio as gr
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
import numpy as np
from sklearn.preprocessing import normalize

# Load lightweight models for faster local execution
def initialize_models():
    qa_model_name = "distilbert-base-uncased-distilled-squad"  # Lightweight QA model
    embedding_model_name = "all-MiniLM-L6-v2"  # Lightweight embedding model
    
    # Load QA pipeline
    qa_tokenizer = AutoTokenizer.from_pretrained(qa_model_name)
    qa_model = AutoModelForQuestionAnswering.from_pretrained(qa_model_name)
    qa_pipeline = pipeline("question-answering", model=qa_model, tokenizer=qa_tokenizer)

    # Load SentenceTransformer for embeddings
    embedding_model = SentenceTransformer(embedding_model_name)
    
    return qa_pipeline, embedding_model

qa_pipeline, embedding_model = initialize_models()

# Extract text from PDF
def extract_text_from_pdf(pdf_file):
    """Extracts all text from the uploaded PDF file."""
    doc = fitz.open(pdf_file)  # Open the PDF file using its filepath
    text = ""
    for page in doc:
        text += page.get_text()  # Extract text from each page
    doc.close()
    return text

# Split text into manageable chunks
def chunk_text(text, chunk_size=300):
    """Splits text into smaller chunks for processing."""
    words = text.split()
    return [' '.join(words[i:i + chunk_size]) for i in range(0, len(words), chunk_size)]

# Find the most relevant chunk of text
def find_relevant_chunk(question, chunks):
    """Finds the chunk of text most relevant to the given question."""
    question_embedding = normalize([embedding_model.encode(question)])
    chunk_embeddings = normalize(embedding_model.encode(chunks))
    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
    best_chunk_index = np.argmax(similarities)
    return chunks[best_chunk_index]

# Generate the answer to the user's question
def answer_question(pdf_file, question):
    """Answers the user's question based on the PDF content."""
    try:
        context = extract_text_from_pdf(pdf_file)
        chunks = chunk_text(context)
        relevant_chunk = find_relevant_chunk(question, chunks)

        # Get the answer using the QA pipeline
        response = qa_pipeline(question=question, context=relevant_chunk)
        answer = response.get("answer", "No answer found.")
        confidence = response.get("score", 0)
        return f"Answer: {answer}", f"Confidence: {confidence:.2%}"
    except Exception as e:
        return "Error processing your request.", str(e)

# Create the Gradio interface
def create_gradio_interface():
    """Sets up the Gradio interface for local deployment."""
    interface = gr.Interface(
        fn=answer_question,
        inputs=[
            gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"]),  # Use filepath for local file processing
            gr.Textbox(label="Ask a Question")  # User enters the question
        ],
        outputs=["text", "text"],
        title="Local PDF Q&A Bot",
        description=(
            "Upload a PDF and ask questions about its content. "
            "The bot will extract text from the PDF and provide answers based on its content."
        ),
        theme="default",
    )
    return interface

# Launch the Gradio interface
def launch_interface():
    """Launches the Gradio interface."""
    interface = create_gradio_interface()
    port = int(os.environ.get("PORT", 7860))
    interface.launch(server_name="0.0.0.0", server_port=port)
    
    interface.launch()

if __name__ == "__main__":
    launch_interface()
