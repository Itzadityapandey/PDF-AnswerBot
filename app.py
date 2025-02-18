import gradio as gr
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.preprocessing import normalize
import os
from dotenv import load_dotenv
import google.generativeai as genai

# Load environment variables (API Key for Gemini)
load_dotenv()

# Configure Gemini API
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

# Initialize lightweight models for embeddings
def initialize_embedding_model():
    embedding_model_name = "all-MiniLM-L6-v2"  # Lightweight embedding model
    embedding_model = SentenceTransformer(embedding_model_name)
    return embedding_model

embedding_model = initialize_embedding_model()

# Split text into manageable chunks
def chunk_text(text, chunk_size=300):
    """Splits text into smaller chunks for processing."""
    try:
        sentences = text.split('\n')
        chunks = []
        current_chunk = ""

        for sentence in sentences:
            if len(current_chunk.split()) + len(sentence.split()) <= chunk_size:
                current_chunk += " " + sentence
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    except Exception as e:
        print(f"Error during text chunking: {e}")
        return []

# Find the most relevant chunks of text
def find_relevant_chunks(question, chunks, top_k=3):
    """Finds the k most relevant chunks of text for the given question."""
    if not chunks:
        return []
    question_embedding = normalize([embedding_model.encode(question)])
    chunk_embeddings = normalize(embedding_model.encode(chunks))
    similarities = np.dot(chunk_embeddings, question_embedding.T).flatten()
    top_chunk_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_chunk_indices]

# Query the Gemini API using the client library
def query_gemini_api(question, context):
    """Sends the question and context to the Gemini API and retrieves the answer."""
    try:
        chat_session = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "text/plain",
            },
        ).start_chat(
            history=[
                {
                    "role": "user",
                    "parts": [
                        f"Given the following context: '{context}'. Answer this question: {question}"
                    ],
                },
            ]
        )
        response = chat_session.send_message("")  # Trigger the response
        return {"answer": response.text, "confidence": 1.0}  # Dummy confidence - can't get it directly
    except Exception as e:
        return {"error": f"Error during API call: {e}"}

def answer_question(pasted_text, question):
    """Answers the user's question based on the pasted text."""
    try:
        # Validate input
        if not pasted_text.strip():
            return "Error: Input text is empty.", "Please paste valid text to process."
        
        # Preprocess text
        preprocessed_text = preprocess_text(pasted_text)
        
        # Chunk text
        chunks = chunk_text(preprocessed_text)
        if not chunks:
            return "Error: Unable to process the text.", "No chunks were generated from the input."
        
        # Find relevant chunks
        relevant_chunks = find_relevant_chunks(question, chunks)
        if not relevant_chunks:
            return "Error: No relevant chunks found.", "Please try with more detailed text."
        
        # Combine chunks into context
        context = " ".join(relevant_chunks)
        print(f"Context being sent to Gemini API: {context}")
        
        # Ensure context is not empty
        if not context.strip():
            return "Error: Context is empty.", "The input text might not contain sufficient information."
        
        # Query Gemini API
        gemini_response = query_gemini_api(question, context)
        
        if "error" in gemini_response:
            return "Error processing your request.", gemini_response["error"]
        
        answer = gemini_response.get("answer", "No answer found.")
        confidence = gemini_response.get("confidence", 0)
        return f"Answer: {answer}", f"Confidence: {confidence:.2%}"
    
    except Exception as e:
        return "Error processing your request.", str(e)

# Create the Gradio interface
def create_gradio_interface():
    """Sets up the Gradio interface for local deployment."""
    interface = gr.Interface(
        fn=answer_question,
        inputs=[
            gr.Textbox(label="Paste Text"),  # User pastes text here
            gr.Textbox(label="Ask a Question")  # User enters the question
        ],
        outputs=["text", "text"],
        title="Text Q&A Bot with Gemini API",
        description=(
            "Paste text and ask questions about its content. "
            "The bot will provide answers based on the content using the Gemini API."
        ),
        theme="default",
    )
    return interface

# Launch the Gradio interface
def launch_interface():
    """Launches the Gradio interface with Render-compatible port settings."""
    interface = create_gradio_interface()
    
    # Ensure we use the port provided by Render (or default to 7860 locally)
    port = int(os.environ.get("PORT", 7860))
    
    # Launch the Gradio interface on Render with the specified port
    interface.launch(server_name="0.0.0.0", server_port=port)

if __name__ == "__main__":
    launch_interface()
