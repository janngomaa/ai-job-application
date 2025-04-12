# AI Job Application Assistant

An intelligent application that helps users fill out job application forms using their resume. The application uses LLM-powered workflows to automate form filling while allowing human feedback and corrections.

## Features

- **Resume & Form Upload**: Upload your resume and job application form in PDF or DOCX format
- **Intelligent Form Filling**: Automatically extracts information from your resume to fill out job application forms
- **Human-in-the-Loop**: Review and provide feedback on the generated responses
- **Modern UI**: Clean and intuitive interface built with Streamlit
- **Robust Backend**: FastAPI backend with async support and proper error handling

## Architecture

The application follows a client-server architecture:

- **Frontend**: Streamlit application providing the user interface
- **Backend**: FastAPI server handling file processing and workflow management
- **Workflow Engine**: Custom RAG (Retrieval-Augmented Generation) workflow using LlamaIndex

## Prerequisites

- Python 3.10 or higher
- uv package manager

## Installation

1. Clone the repository:
```bash
git clone https://github.com/janngomaa/ai-job-application.git
cd ai-job-application
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows, use `.venv\Scripts\activate`
```

3. Install dependencies using uv:
```bash
uv pip install -r requirements.txt
```

## Configuration

1. Set up your environment variables:
```bash
export LLAMA_CLOUD_API_KEY=your_api_key_here
export OPENAI_API_KEY=your_openai_api_key_here
```

2. Create a `data` directory for file storage:
```bash
mkdir data
```

## Running the Application

1. Start the FastAPI backend:
```bash
uvicorn src.backend.api:app --reload
```

2. In a new terminal, start the Streamlit frontend:
```bash
streamlit run src/streamlit/app.py
```

3. Open your browser and navigate to http://localhost:8501

## Usage

1. Upload your resume (PDF or DOCX)
2. Upload the job application form (PDF or DOCX)
3. Click "Process Files" to start the automated form filling
4. Review the generated responses
5. Provide feedback if needed
6. Submit your feedback to improve the responses

## Project Structure

```
ai-job-application/
├── data/                  # Storage for uploaded files
├── src/
│   ├── backend/          # FastAPI backend
│   │   └── api.py       # API endpoints
│   ├── streamlit/        # Streamlit frontend
│   │   └── app.py       # UI implementation
│   ├── helper/          # Helper functions
│   └── job_application_human_in_loop.py  # Core workflow
├── requirements.txt      # Project dependencies
└── README.md            # Documentation
```

## Error Handling

The application includes comprehensive error handling:
- File upload validation
- Backend processing errors
- Network communication errors
- Workflow state management

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.
