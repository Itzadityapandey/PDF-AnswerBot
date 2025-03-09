# Chat with PDF 

## 📽 Demo Video
<video src="media/Chat_with_pdf.mp4" controls width="600"></video>

## Overview
This project allows users to upload PDF files and chat with their content. It utilizes Google Gemini AI for generating embeddings and answering questions based on extracted text. The PDF content is processed and stored in a FAISS vector database for efficient retrieval.

## Features
- Upload and process multiple PDF files.
- Extract and store text in a FAISS vector store.
- Ask questions about the content and receive AI-generated responses.
- Uses Google Gemini AI (`gemini-pro`) for conversational responses.
- Gradio-based web UI for ease of use.

## Requirements
Make sure you have the following dependencies installed before running the project:

- Python 3.8+
- `gradio`
- `PyPDF2`
- `langchain`
- `langchain_google_genai`
- `google-generativeai`
- `langchain_community`
- `faiss-cpu`
- `python-dotenv`

Install the dependencies using:
```sh
pip install -r requirements.txt
```

## Setup
1. Clone the repository:
   ```sh
   git clone https://github.com/your-username/chat-with-pdf.git
   cd chat-with-pdf
   ```
2. Create a `.env` file and add your Google API key:
   ```
   GOOGLE_API_KEY=your_google_api_key_here
   ```
3. Run the application:
   ```sh
   python app.py
   ```
4. Open the Gradio UI in your browser and start interacting with your PDFs.

## How It Works
1. **Upload PDFs**: Select multiple PDFs to process.
2. **Extract & Store**: The text is extracted, split into chunks, embedded, and stored in a FAISS vector database.
3. **Ask Questions**: Query the processed text, and the AI will fetch the relevant context and generate an answer.
4. **Receive Answers**: The AI provides responses based on the extracted information.

## File Structure
```
📂 chat-with-pdf
├── 📜 app.py              # Main application script
├── 📜 requirements.txt    # List of dependencies
├── 📜 README.md           # Documentation
├── 📂 faiss_index         # FAISS vector store (auto-created after processing PDFs)
└── 📜 .env                # Environment variables (API keys, etc.)
```

## Future Enhancements
- Improve document parsing to handle scanned PDFs (OCR integration).
- Enhance the UI with real-time chat features.
- Add support for multiple language models.

## License
This project is licensed under the MIT License.



