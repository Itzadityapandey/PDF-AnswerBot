import gradio as gr
from tika import parser
from transformers import pipeline, AutoModelForQuestionAnswering, AutoTokenizer

# Load pre-trained model and tokenizer for question-answering
model_name = "distilbert-base-uncased-distilled-squad"
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

# Function to extract text from the uploaded PDF
def extract_text_from_pdf(pdf_file):
    raw = parser.from_file(pdf_file.name)
    return raw['content']

# Function to answer questions based on the extracted text from PDF
def answer_question_with_confidence(pdf_file, question):
    context = extract_text_from_pdf(pdf_file)
    result = qa_pipeline(question=question, context=context)
    answer = result['answer']
    confidence = result.get('score', 0)
    return f"Answer: {answer}", f"Confidence: {confidence:.2%}"

def handler(event, context):
    # You would handle the request to your Gradio interface here
    interface = gr.Interface(fn=answer_question_with_confidence,
                             inputs=[gr.File(label="Upload PDF", type="filepath"),
                                     gr.Textbox(label="Ask a Question")],
                             outputs=["text", "text"],
                             title="PDF Q&A Model",
                             description="Upload a PDF and ask questions about its content.")
    return interface.launch(inline=True)
