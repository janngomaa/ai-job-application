#job_application.py
import os
import nest_asyncio
from llama_parse import LlamaParse
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.core import VectorStoreIndex
from llama_index.llms.openai import OpenAI
from llama_index.core.tools import FunctionTool
from llama_index.core.agent import FunctionCallingAgent

from helper import get_llama_cloud_api_key, get_openai_api_key


"""
Need nested async (nest_asyncio)for this to work, so let's enable it here. 
It allows you to nest asyncio event loops within each other.
Note: In asynchronous programming, the event loop is like a continuous cycle that manages the execution of code.
"""
nest_asyncio.apply()

llama_cloud_api_key = get_llama_cloud_api_key()
openai_api_key = get_openai_api_key()

def query_resume(q: str) -> str:
    """This function answers questions about a specific resume."""
    # we're using the query engine we already created above
    response = query_engine.query(f"This is a question about the specific resume we have in our database: {q}")
    return response.response

# Read documents
documents = LlamaParse(
    api_key=llama_cloud_api_key,
    base_url=os.getenv("LLAMA_CLOUD_BASE_URL"),
    result_type="markdown",
    content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
).load_data("data/fake_resume.pdf",)

# Create the vector database (embedings)
index = VectorStoreIndex.from_documents(
    documents,
    embed_model=OpenAIEmbedding(
        model_name="text-embedding-3-small", 
        api_key= openai_api_key)
)
# Create a query engine
llm = OpenAI(model="gpt-4o-mini")
query_engine = index.as_query_engine(llm=llm, similarity_top_k=5)
response = query_engine.query("What is this person's name and what was their most recent job?")
# storage_dir = "./storage"
# index.storage_context.persist(persist_dir=storage_dir)

# Define an agent
resume_tool = FunctionTool.from_defaults(fn=query_resume)
agent = FunctionCallingAgent.from_tools(
    tools=[resume_tool],
    llm=llm,
    verbose=True
)

# Chat with the agent
response = agent.chat("How many years of experience does the applicant have?")
print(response)
