import os
import asyncio
import json
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Event,
    Context
)
from llama_index.core import (
    VectorStoreIndex,
    load_index_from_storage
)
from llama_index.core.storage.storage_context import StorageContext
from llama_parse import LlamaParse
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

import nest_asyncio
nest_asyncio.apply()

# Create Events
class ParseFormEvent(Event):
    application_form: str

class QueryEvent(Event):
    query: str

class ResponseEvent(Event):
    response: str


class RAGWorkflow(Workflow):
    storage_dir = "./storage"
    llm: OpenAI
    query_engine: VectorStoreIndex

    # the first step will be setup
    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseFormEvent:

        if not ev.resume_file:
            raise ValueError("No resume file provided")
        if not ev.application_form:
            raise ValueError("No application form provided")

        # define an LLM to work with
        self.llm = OpenAI(model="gpt-4o-mini")

        # ingest the data and set up the query engine
        if os.path.exists(self.storage_dir):
            # you've already ingested your documents
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            # parse and load your documents
            documents = LlamaParse(
                result_type="markdown",
                content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
            ).load_data(ev.resume_file)
            # embed and index the documents
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=OpenAIEmbedding(model_name="text-embedding-3-small")
            )
            index.storage_context.persist(persist_dir=self.storage_dir)

        # either way, create a query engine
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        # you no longer need a query to be passed in, 
        # you'll be generating the queries instead 
        # let's pass the application form to a new step to parse it
        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> QueryEvent:
        parser = LlamaParse(
            result_type="markdown",
            content_guideline_instruction="This is a job application form. Create a list of all the fields that need to be filled in.",
            formatting_instruction="Return a bulleted list of the fields ONLY."
        )
        # get the LLM to convert the parsed form into JSON
        result = parser.load_data(ev.application_form)[0]
        raw_json = self.llm.complete(
            f"""
            This is a parsed form. 
            Convert it into a JSON object containing only the list 
            of fields to be filled in, in the form {{ fields: [...] }}. 
            <form>{result.text}</form>. 
            Return JSON ONLY, no markdown.
            """)
        fields = json.loads(raw_json.text)["fields"]

        # generate one query for each of the fields, and fire them off
        for field in fields:
            # print(f"Sending event for this field: {field}")
            ctx.send_event(QueryEvent(
                field=field,
                query=f"How would you answer this question about the candidate? {field}"
            ))

        # store the number of fields so we know how many to wait for later
        await ctx.set("total_fields", len(fields))
        return
    
    # the second step will be to ask a question and return a result immediately
    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")
        return ResponseEvent(field=ev.field, response=response.response)

    # the third step will be to fill in the application
    @step
    async def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> StopEvent:
        # get the total number of fields to wait for
        total_fields = await ctx.get("total_fields")

        responses = ctx.collect_events(ev, [ResponseEvent] * total_fields)
        if responses is None:
            return None # do nothing if there's nothing to do yet

        # we've got all the responses!
        responseList = "\n".join("Field: " + r.field + "\n" + "Response: " + r.response for r in responses)

        result = self.llm.complete(f"""
            You are given a list of fields in an application form and responses to
            questions about those fields from a resume. Combine the two into a list of
            fields and succinct, factual answers to fill in those fields.

            <responses>
            {responseList}
            </responses>
        """)
        return StopEvent(result=result)
    
async def main():
    # run the workflow
    w = RAGWorkflow(timeout=120, verbose=False)
    result = await w.run(
        resume_file="./data/fake_resume.pdf",
        application_form="./data/rc126-10b.pdf",
        query="Here is this candidate from?"
        )
    print(result)
    

if __name__ == "__main__":
    asyncio.run(main())
