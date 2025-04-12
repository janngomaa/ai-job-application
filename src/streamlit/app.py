"""Streamlit application for resume and job application form processing."""
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Callable, TypeVar, Coroutine, Union
import asyncio
import sys
import time
from functools import wraps
import requests
import functools
import json

import streamlit as st
import nest_asyncio
from streamlit.runtime.uploaded_file_manager import UploadedFile

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Enable nested event loops
nest_asyncio.apply()

from src.helper.logger import get_logger

# Configure logger
logger = get_logger("streamlit_app")

# FastAPI backend URL
API_BASE_URL = "http://localhost:8000"

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

def timing_decorator(func: Callable[..., R]) -> Callable[..., R]:
    """
    Decorator to measure and log the execution time of functions.
    
    Args:
        func: The function to be timed
        
    Returns:
        Wrapped function that logs execution time
    """
    @wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> R:
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Function '{func.__name__}' took {execution_time:.2f} seconds to execute")
    return wrapper

@timing_decorator
def upload_files_to_backend(resume_file: UploadedFile, form_file: UploadedFile) -> Dict[str, Any]:
    """
    Upload files to the FastAPI backend.
    
    Args:
        resume_file: The uploaded resume file
        form_file: The uploaded application form file
        
    Returns:
        Response from the backend
        
    Raises:
        requests.RequestException: If the upload fails
    """
    files = {
        "resume": (resume_file.name, resume_file.getvalue()),
        "application_form": (form_file.name, form_file.getvalue())
    }
    
    try:
        response = requests.post(f"{API_BASE_URL}/upload/", files=files)
        logger.debug(f"Response from backend: {response.text}")
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error uploading files to backend: {str(e)}")
        raise

@timing_decorator
def send_workflow_response(workflow_id: str, response_data: Dict[str, str]) -> Dict[str, Any]:
    """
    Send response to the workflow.
    
    Args:
        workflow_id: ID of the workflow
        response_data: Response data
        
    Returns:
        Response from the backend
        
    Raises:
        requests.RequestException: If the request fails
    """
    try:
        response = requests.post(
            f"{API_BASE_URL}/workflow/{workflow_id}/respond",
            json=response_data
        )
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        logger.error(f"Error sending workflow response: {str(e)}")
        raise

def initialize_session_state() -> None:
    """Initialize the session state variables if they don't exist."""
    if "processing_state" not in st.session_state:
        st.session_state.processing_state = "upload"
    if "workflow_id" not in st.session_state:
        st.session_state.workflow_id = None
    if "filled_form" not in st.session_state:
        st.session_state.filled_form = None
    if "feedback_prompt" not in st.session_state:
        st.session_state.feedback_prompt = None

def main():
    """Main application function."""
    st.title("AI Job Application Assistant")
    initialize_session_state()

    col1, col2 = st.columns([3, 5])

    with col1:
        st.header("Upload your resume and job application form")
        st.write("Upload your resume and the job application form to get started.")
            
        resume_file = st.file_uploader("Upload your resume", type=["pdf", "docx"])
        form_file = st.file_uploader("Upload the job application form", type=["pdf", "docx"])
            
        if resume_file and form_file and st.button("Process Files"):
            try:
                with st.spinner("Processing files..."):
                    response = upload_files_to_backend(resume_file, form_file)
                    logger.debug(f"Response from backend: {response}")
                    st.session_state.processing_state = "feedback"
                    st.session_state.workflow_id = response["workflow_id"]
                    st.session_state.filled_form = response["filled_form"]
                    st.session_state.feedback_prompt = response["feedback_prompt"]
                st.success("Files processed successfully!")
                st.rerun()
            except Exception as e:
                st.error(f"Error processing files: {str(e)}")
                logger.error("Error in file processing", exc_info=True)

    with col2:
        if st.session_state.processing_state == "feedback":
            st.header("Your feedback!")
            st.write("Your files are being processed. Please provide feedback when requested.")
            
            st.markdown("### Generated Form")
            st.markdown(st.session_state.filled_form)
            
            st.markdown("### Feedback Required")
            st.write(st.session_state.feedback_prompt)
            user_feedback = st.text_area("Your feedback:")
            
            if user_feedback and st.button("Submit Feedback"):
                try:
                    with st.spinner("Processing feedback..."):
                        response = send_workflow_response(
                            st.session_state.workflow_id,
                            {"feedback": user_feedback}
                        )
                        st.success("Feedback submitted successfully!")
                        st.session_state.feedback_prompt = None
                        st.rerun()
                except Exception as e:
                    st.error(f"Error submitting feedback: {str(e)}")
                    logger.error("Error submitting feedback", exc_info=True)

if __name__ == "__main__":
    main()
