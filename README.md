# AI-Powered Job Application Assistant

An intelligent job application assistant that helps automate the process of filling out job applications using a candidate's resume. Built with LlamaIndex and OpenAI.

## Features

- Automatic form field detection and parsing
- Resume analysis and information extraction
- Read job application form
- Fill in application form
- Interactive feedback (human-in-the-loop) and refinement
- Daily log rotation for better log management

## Requirements

- Python 3.10+
- OpenAI API key
- LlamaIndex
- Other dependencies (see requirements.txt)

## Project Structure

```
llamaindex/
├── src/
│   ├── helper/
│   │   ├── __init__.py
│   │   └── logger.py
│   └── job_application_human_in_loop.py
├── data/
│   ├── fake_resume.pdf
│   └── rc126-10b.pdf
├── logs/
└── storage/
```

## Setup

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up your environment variables:
   ```bash
   export OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the job application assistant:

```bash
python -m src.job_application_human_in_loop
```

The system will:
1. Parse the provided resume
2. Analyze the job application form
3. Generate appropriate responses
4. Allow for human feedback and refinement
5. Produce a completed application

## Logging

The system uses a custom logging framework that:
- Automatically detects and logs function names
- Rotates logs daily
- Supports multiple log levels
- Provides both file and console output

Logs are stored in the `logs/` directory with the naming format: `app_name_YYYYMMDD.log`
