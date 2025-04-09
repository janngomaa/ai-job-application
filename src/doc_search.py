import os
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
response = query_engine.query("What are the bill date, the account number and the tola amount?")

print(response)