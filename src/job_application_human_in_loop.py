import os
import asyncio
import json
from src.helper.logger import get_logger
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    HumanResponseEvent,
    InputRequiredEvent,
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
    field: str
    
class ResponseEvent(Event):
    response: str

class FeedbackEvent(Event):
    feedback: str

class GenerateQuestionsEvent(Event):
    pass

class RAGWorkflow(Workflow):
    
    storage_dir = "./storage"
    llm: OpenAI
    query_engine: VectorStoreIndex
    logger = get_logger("job_application")

    @step
    async def set_up(self, ctx: Context, ev: StartEvent) -> ParseFormEvent:
        self.logger.log_step("Starting workflow setup")

        if not ev.resume_file:
            self.logger.error("No resume file provided")
            raise ValueError("No resume file provided")

        if not ev.application_form:
            self.logger.error("No application form provided")
            raise ValueError("No application form provided")

        self.logger.log_step("Initializing LLM")
        self.llm = OpenAI(model="gpt-4o-mini")

        if os.path.exists(self.storage_dir):
            self.logger.log_step("Loading existing index from storage")
            storage_context = StorageContext.from_defaults(persist_dir=self.storage_dir)
            index = load_index_from_storage(storage_context)
        else:
            self.logger.log_step("Creating new index from resume")
            documents = LlamaParse(
                result_type="markdown",
                content_guideline_instruction="This is a resume, gather related facts together and format it as bullet points with headers"
            ).load_data(ev.resume_file)
            
            self.logger.debug("Embedding and indexing documents")
            index = VectorStoreIndex.from_documents(
                documents,
                embed_model=OpenAIEmbedding(model_name="text-embedding-3-small")
            )
            index.storage_context.persist(persist_dir=self.storage_dir)
            self.logger.log_step("setup", "Index created and persisted")

        self.logger.debug("Creating query engine")
        self.query_engine = index.as_query_engine(llm=self.llm, similarity_top_k=5)

        self.logger.log_step("Setup completed successfully")
        return ParseFormEvent(application_form=ev.application_form)

    @step
    async def parse_form(self, ctx: Context, ev: ParseFormEvent) -> GenerateQuestionsEvent:
        self.logger.log_step("Starting form parsing")
        parser = LlamaParse(
            result_type="markdown",
            content_guideline_instruction="This is a job application form. Create a list of all the fields that need to be filled in.",
            user_prompt="Return a bulleted list of the fields ONLY."
        )

        self.logger.debug("Converting parsed form to JSON")
        result = parser.load_data(ev.application_form)[0]
        raw_json = self.llm.complete(
            f"This is a parsed form. Convert it into a JSON object containing only the list of fields to be filled in, in the form {{ fields: [...] }}. <form>{result.text}</form>. Return JSON ONLY, no markdown.")
        fields = json.loads(raw_json.text)["fields"]
        
        self.logger.log_step(f"Found {len(fields)} fields to fill")
        self.logger.debug(f"Fields: {fields}")

        await ctx.set("fields_to_fill", fields)
        return GenerateQuestionsEvent()

    @step
    async def generate_questions(self, ctx: Context, ev: GenerateQuestionsEvent) -> QueryEvent:
        self.logger.log_step("Generating initial questions...")

        fields = await ctx.get("fields_to_fill")
        self.logger.debug(f"Processing {len(fields)} fields")

        for field in fields:
            question = f"How would you answer this question about the candidate? <field>{field}</field>"
            
            ctx.send_event(QueryEvent(
                field=field,
                query=question
            ))

        await ctx.set("total_fields", len(fields))
        return

    @step
    async def ask_question(self, ctx: Context, ev: QueryEvent) -> ResponseEvent:
        self.logger.debug(f"Processing question for field: {ev.field}")
        response = self.query_engine.query(f"This is a question about the specific resume we have in our database: {ev.query}")
        self.logger.debug(f"Received response for field {ev.field}")
        return ResponseEvent(field=ev.field, response=response.response)

    @step
    async def fill_in_application(self, ctx: Context, ev: ResponseEvent) -> InputRequiredEvent:
        total_fields = await ctx.get("total_fields")
        self.logger.debug(f"Collecting responses for {total_fields} fields")

        responses = ctx.collect_events(ev, [ResponseEvent] * total_fields)
        if responses is None:
            self.logger.debug("Waiting for more responses")
            return None

        self.logger.log_step("All responses collected, generating final form")
        responseList = "\n".join("Field: " + r.field + "\n" + "Response: " + r.response for r in responses)

        result = self.llm.complete(f"""
            You are given a list of fields in an application form and responses to
            questions about those fields from a resume. Combine the two into a list of
            fields and succinct, factual answers to fill in those fields.

            <responses>
            {responseList}
            </responses>
        """)

        await ctx.set("filled_form", str(result))
        self.logger.log_step("Form filled, requesting human feedback")
        
        return InputRequiredEvent(
            prefix="How does this look? Give me any feedback you have on any of the answers.",
            result=result
        )

    @step
    async def get_feedback(self, ctx: Context, ev: HumanResponseEvent) -> FeedbackEvent | StopEvent:
        self.logger.log_step("Processing human feedback ...")
        self.logger.debug(f"Feedback content: {ev.response}")

        result = self.llm.complete(f"""
            You have received some human feedback on the form-filling task you've done.
            Does everything look good, or is there more work to be done?
            <feedback>
            {ev.response}
            </feedback>
            If everything is fine, respond with just the word 'OKAY'.
            If there's any other feedback, respond with just the word 'FEEDBACK'.
        """)

        verdict = result.text.strip()
        self.logger.log_step(f"Feedback verdict: {verdict}")

        if verdict == "OKAY":
            self.logger.log_step("Workflow completed successfully")
            return StopEvent(
                result=await ctx.get("filled_form")
            )
        else:
            self.logger.log_step("Additional iteration required based on feedback")
            return FeedbackEvent(feedback=ev.response)

    @step
    async def integrate_human_feedback(self, ctx: Context, ev: FeedbackEvent) -> InputRequiredEvent:
        self.logger.log_step("Integrating human feedback")
        filled_form = await ctx.get("filled_form")
        feedback = ev.feedback

        result = self.llm.complete(f"""
            You have received some human feedback on the form-filling task you've done.
            Please integrate the feedback into the form.
            <feedback>{filled_form}</feedback>
            <feedback>{feedback}</feedback>
            Return the updated form.
        """)
        await ctx.set("filled_form", str(result))
        return InputRequiredEvent(
            prefix="How does this look? Give me any feedback you have on any of the answers.",
            result=await ctx.get("filled_form")
        )
        

async def main():
    logger = get_logger("job_application")
    logger.log_step("Starting job application workflow")
    try:
        workflow = RAGWorkflow(timeout=600, verbose=False)
        handler = workflow.run(
            resume_file="./data/fake_resume.pdf",
            application_form="./data/rc126-10b.pdf"
        )

        async for event in handler.stream_events():
            if isinstance(event, InputRequiredEvent):
                logger.log_step("Human feedback required")
                print("We've filled in your form! Here are the results:\n")
                print(event.result)
                # now ask for input from the keyboard
                response = input(event.prefix)
                handler.ctx.send_event(
                    HumanResponseEvent(
                        response=response
                    )
                )

        response = await handler
        logger.log_step("Job application workflow completed")
        print("Agent complete! Here's your final result:")
        print(str(response))
    except Exception as e:
        logger.error(f"Error in workflow: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    asyncio.run(main())
