import gradio as gr
from tika import parser
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer
import os

# Function to check file type and validate PDF
def validate_pdf_file(file_path):
    """Checks if the uploaded file is a valid PDF."""
    if not file_path.lower().endswith(".pdf"):
        raise ValueError("Invalid file format. Please upload a PDF file.")
    return True

# Load pre-trained model and tokenizer for question-answering once at the start
def load_qa_model():
    """Loads the pre-trained model and tokenizer for question-answering."""
    model_name = "distilbert-base-uncased-distilled-squad"  # Pre-trained model for QA
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return model, tokenizer

model, tokenizer = load_qa_model()

# Initialize the question-answering pipeline with the pre-trained model
def initialize_pipeline(model, tokenizer):
    """Initializes the question-answering pipeline."""
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = initialize_pipeline(model, tokenizer)

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    """Extracts text content from the uploaded PDF file using Tika."""
    raw = parser.from_file(pdf_file.name)  # Use the file path to extract text
    return raw['content']

# Function to process the uploaded PDF
def process_pdf(file_path):
    """Processes the uploaded PDF file and extracts its content."""
    validate_pdf_file(file_path)
    return extract_text_from_pdf(file_path)

# Function to answer questions based on the extracted text from PDF
def answer_question_with_confidence(pdf_file, question):
    """Extracts text from the uploaded PDF and answers the question with a confidence score."""
    # Extract text from the uploaded PDF
    context = extract_text_from_pdf(pdf_file)
    
    # Get the answer to the user's question from the extracted context
    result = qa_pipeline(question=question, context=context)
    
    # Answer and confidence score
    answer = result['answer']
    confidence = result.get('score', 0)  # Confidence score of the answer
    
    # Return the answer and confidence
    return f"Answer: {answer}", f"Confidence: {confidence:.2%}"

# Function to provide extra tips for question formulation
def provide_question_tips():
    """Provides tips for asking better questions to the model."""
    tips = """
    1. Be specific with your questions.
    2. Avoid asking very general questions like 'What is the document about?'
    3. Try asking questions related to sections or paragraphs of the document.
    4. Ensure that your question is clearly stated and grammatically correct.
    """
    return tips

# Create the Gradio interface
def create_gradio_interface():
    """Creates the Gradio interface for the PDF Q&A Model."""
    interface = gr.Interface(
        fn=answer_question_with_confidence,
        inputs=[
            gr.File(label="Upload PDF", type="filepath", file_types=[".pdf"]),  # Use 'filepath' here
            gr.Textbox(label="Ask a Question")
        ],
        outputs=["text", "text"],
        title="PDF Q&A Model",
        description="Upload a PDF document and ask questions about its content. For better results, follow these tips: " + provide_question_tips(),
        theme="huggingface"  # You can customize the theme if needed
    )
    return interface

# Launch the Gradio interface
def launch_interface():
    """Launches the Gradio interface."""
    interface = create_gradio_interface()
    interface.launch()

launch_interface()
