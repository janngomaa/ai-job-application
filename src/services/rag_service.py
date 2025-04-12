"""RAG service for processing resumes and job application forms."""
from pathlib import Path
from typing import Optional, Dict, Any, Callable, TypeVar, Coroutine, AsyncIterator
import sys
import asyncio
import time
from functools import wraps

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.helper.logger import get_logger
from src.job_application_human_in_loop import (
    RAGWorkflow, 
    InputRequiredEvent, 
    HumanResponseEvent
)
from llama_index.core.workflow import Context

# Configure logger
logger = get_logger("rag_service")

# Type variables for generic functions
T = TypeVar('T')
R = TypeVar('R')

def timing_decorator(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., Coroutine[Any, Any, T]]:
    """
    Decorator to measure and log the execution time of async functions.
    
    Args:
        func: The async function to be timed
        
    Returns:
        Wrapped async function that logs execution time
    """
    @wraps(func)
    async def wrapper(*args: Any, **kwargs: Any) -> T:
        start_time = time.time()
        try:
            result = await func(*args, **kwargs)
            return result
        finally:
            end_time = time.time()
            execution_time = end_time - start_time
            logger.info(f"Function '{func.__name__}' took {execution_time:.2f} seconds to execute")
    return wrapper

class RAGService:
    """Service class to handle RAG workflow operations."""
    
    def __init__(self) -> None:
        """Initialize the RAG service."""
        self.workflow: Optional[RAGWorkflow] = None
        self.handler: Optional[Context] = None
        self.loop: Optional[asyncio.AbstractEventLoop] = None

    @timing_decorator
    async def initialize_workflow(self) -> None:
        """Initialize the RAG workflow."""
        # Store the current event loop
        self.loop = asyncio.get_event_loop()
        self.workflow = RAGWorkflow(timeout=600, verbose=True)

    @timing_decorator
    async def process_files(self, resume_path: str, application_form_path: str) -> Dict[str, Any]:
        """
        Process the uploaded files with RAG workflow.
        
        Args:
            resume_path: Path to the resume file
            application_form_path: Path to the application form file
            
        Returns:
            Dict containing status and relevant data from the workflow
            
        Raises:
            RuntimeError: If workflow initialization fails
        """
        if not self.workflow:
            await self.initialize_workflow()
        
        if not self.workflow:  # Type guard
            raise RuntimeError("Failed to initialize workflow")
        
        self.handler = self.workflow.run(
            resume_file=resume_path,
            application_form=application_form_path
        )

        async for event in self.handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                logger.info("Human feedback required")
                return {
                    "status": "feedback_required",
                    "result": event.result,
                    "prefix": event.prefix
                }
        
        return {
            "status": "completed",
            "result": "No feedback required"
        }

    @timing_decorator
    async def submit_feedback(self, feedback: str) -> Dict[str, Any]:
        """
        Submit user feedback to the workflow.
        
        Args:
            feedback: User feedback string
            
        Returns:
            Dict containing status and relevant data from the workflow
            
        Raises:
            ValueError: If workflow is not initialized or no handler is available
        """
        logger.info(f"Submitting feedback: {feedback}")
    
        if not self.handler:
            raise ValueError("Workflow not initialized or no handler available")
    
        # Using the original loop for operations
        async def _send_feedback_and_process() -> Dict[str, Any]:
            # Send the feedback event
            self.handler.ctx.send_event(HumanResponseEvent(response=feedback))
            
            # Process the next event
            try:
                async for event in self.handler.stream_events():
                    if isinstance(event, InputRequiredEvent):
                        return {
                            "status": "feedback_required",
                            "result": event.result,
                            "prefix": event.prefix
                        }

            except StopAsyncIteration:
                return {"status": "completed", "result": "Workflow completed successfully"}
            
            return {"status": "completed", "result": "Workflow completed successfully"}
        
        return await self._run_in_event_loop(_send_feedback_and_process())

    @timing_decorator
    async def _run_in_event_loop(self, coroutine: Coroutine[Any, Any, T]) -> T:
        """
        Run a coroutine in the original event loop.
        
        Args:
            coroutine: The coroutine to run
            
        Returns:
            The result of the coroutine execution
            
        Raises:
            RuntimeError: If event loop handling fails
        """
        if self.loop is None:
            self.loop = asyncio.get_event_loop()
            
        if self.loop == asyncio.get_event_loop():
            # Already in the correct loop
            return await coroutine
        else:
            # Create a future in the original loop
            future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
            return await asyncio.wrap_future(future)