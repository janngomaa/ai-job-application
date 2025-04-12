from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Dict, Optional, Any
import asyncio
from llama_index.core.workflow import (
    HumanResponseEvent,
    InputRequiredEvent
)
from src.job_application_human_in_loop import RAGWorkflow 
from src.helper.logger import get_logger
import os
from pathlib import Path
import uuid
from datetime import datetime

app = FastAPI()
logger = get_logger("fastapi-backend")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Get project root directory
project_root = Path(__file__).parent.parent.parent

# Store active workflows and their states
active_workflows: Dict[str, Dict[str, Any]] = {}

def save_uploaded_file(uploaded_file: UploadFile, file_type: str) -> Path:
    """
    Save the uploaded file to the data directory.
    
    Args:
        uploaded_file: The uploaded file from FastAPI
        file_type: Type of file (resume or job_application_form)
        
    Returns:
        Path: Path to the saved file
    """
    # Create timestamp for unique filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Clean filename and create new filename with timestamp
    original_filename = uploaded_file.filename  # Using filename instead of name
    file_extension = os.path.splitext(original_filename)[1]
    new_filename = f"{file_type}_{timestamp}{file_extension}"
    
    # Ensure data directory exists
    save_path = project_root / "data" / new_filename
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save the file
    try:
        with open(save_path, "wb") as f:
            f.write(uploaded_file.file.read())  # Using file.read() instead of getbuffer()
        logger.info(f"Successfully saved {file_type} file: {new_filename}")
        return save_path
    except Exception as e:
        logger.error(f"Error saving {file_type} file: {str(e)}")
        raise

@app.post("/upload/")
async def upload_files(
    resume: UploadFile = File(...),
    application_form: UploadFile = File(...)
) -> Dict[str, str]:

    try:
        # Save uploaded files
        resume_path = save_uploaded_file(resume, "resume")
        form_path = save_uploaded_file(application_form, "job_application_form")
        
        # Initialize workflow
        logger.debug("Initializing workflow")
        workflow = RAGWorkflow()
        workflow_id = str(uuid.uuid4())
        
        try:
            # Create handler and store it
            handler = workflow.run(
                resume_file=str(resume_path),
                application_form=str(form_path)
            )
            
            # Store workflow data
            active_workflows[workflow_id] = {
                "workflow": workflow,
                "handler": handler,
                "files": {
                    "resume": str(resume_path),
                    "form": str(form_path)
                }
            }
            
            # Process initial events
            async for event in handler.stream_events():
                logger.log_step(f"Processing event: {event}")
                if isinstance(event, InputRequiredEvent):
                    logger.log_step(f"Human feedback required: {event.prefix}")
                    logger.debug(f"Filled form: {event.result}")
                    # break  # Stop after first input required event
            
                    return {
                        "message": "Please provide your feedback",
                        "workflow_id": str(workflow_id),
                        "filled_form": str(event.result),
                        "feedback_prompt": str(event.prefix)
                    }
            
        except Exception as e:
            # Clean up files in case of workflow error
            os.remove(resume_path)
            os.remove(form_path)
            logger.error(f"Error in workflow execution: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
        
    except Exception as e:
        logger.error(f"Error processing files: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/workflow/{workflow_id}/respond")
async def handle_workflow_response(
    workflow_id: str,
    response: Dict[str, str]
) -> Dict[str, str]:
    try:
        if workflow_id not in active_workflows:
            raise HTTPException(status_code=404, detail="Workflow not found")
        
        workflow_data = active_workflows[workflow_id]
        handler = workflow_data["handler"]
        
        try:
            logger.info(f"Processing user feedback ... ")
            # Send human response event
            handler.ctx.send_event(
                HumanResponseEvent(response=response["feedback"])
            )
            
            # Continue processing events
            async for event in handler.stream_events():
                logger.log_step(f"Processing event: {event}")
                if isinstance(event, InputRequiredEvent):
                    logger.log_step(f"Human feedback required: {event.prefix}")
                    logger.debug(f"Filled form: {event.result}")
                    break  # Stop after first input required event
            
            return {"message": "Response processed successfully"}
            
        except Exception as e:
            logger.error(f"Error processing response in workflow: {str(e)}")
            raise HTTPException(status_code=500, detail=str(e))
            
    except Exception as e:
        logger.error(f"Error handling response: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def cleanup():
    """Clean up temporary files when the server shuts down."""
    for workflow_id, data in active_workflows.items():
        try:
            files = data.get("files", {})
            for file_path in files.values():
                if os.path.exists(file_path):
                    os.remove(file_path)
        except Exception as e:
            logger.error(f"Error cleaning up files for workflow {workflow_id}: {str(e)}")
