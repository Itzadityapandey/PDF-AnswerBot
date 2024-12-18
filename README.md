Here's a tailored README for your project on GitHub, based on the provided link and the information you've shared:

---

# PDF-AnswerBot

**PDF-AnswerBot** is an intelligent question-answering system that allows users to interact with the content of PDF documents. Users can upload any PDF, ask questions about its content, and receive accurate answers based on the document's extracted text. The application uses a pre-trained DistilBERT model fine-tuned for question answering and leverages the Tika library for extracting text from PDFs.

## Features:
- **PDF Upload**: Easily upload PDF files for processing.
- **Ask Questions**: Users can ask specific questions about the document's content.
- **Confidence Scores**: Display confidence levels for each answer, helping users understand the reliability of the response.
- **Interactive Interface**: A user-friendly Gradio interface for easy interaction with the model.

## Technologies:
- **Python**: The core programming language used for development.
- **Gradio**: Framework for building the interactive web interface.
- **Tika**: Library for extracting text from PDF files.
- **Hugging Face Transformers**: Provides the pre-trained DistilBERT model for question answering.
- **TensorFlow**: Backend for executing the model.

## Installation:

### Step 1: Clone the repository
First, clone the repository to your local machine:
```bash
git clone https://github.com/Itzadityapandey/PDF-AnswerBot.git
cd PDF-AnswerBot
```

### Step 2: Install the required dependencies
Install all the necessary Python packages listed in the `requirements.txt` file:
```bash
pip install -r requirements.txt
```

### Step 3: Run the application
Once the dependencies are installed, you can run the application with the following command:
```bash
python app.py
```

The application will start a local server, and you'll be able to access it via the URL provided in your terminal (usually `http://127.0.0.1:7860`).

## Usage:
1. Upload a PDF document through the Gradio interface.
2. Type a question about the content of the uploaded PDF.
3. The system will provide an answer, along with a confidence score indicating the reliability of the answer.

## Example Questions:
- "What is the main topic of the document?"
- "What are the key findings in section 2?"
- "Summarize the introduction of the paper."

## Contributing:
We welcome contributions to this project! Feel free to fork the repository, create branches for new features or bug fixes, and submit pull requests. If you have any suggestions or encounter any issues, please open an issue on GitHub.

## License:
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


