import asyncio
from llama_index.core.workflow import (
    StartEvent,
    StopEvent,
    Workflow,
    step,
    Context,
    Event
)
from llama_index.utils.workflow import draw_all_possible_flows

class StepAEvent(Event):
    query: str

class StepACompleteEvent(Event):
    result: str

class StepBEvent(Event):
    query: str

class StepBCompleteEvent(Event):
    result: str

class StepCEvent(Event):
    query: str

class StepCCompleteEvent(Event):
    result: str

class ConcurrentFlow(Workflow):
    @step
    async def start(
        self, ctx: Context, ev: StartEvent
    ) -> StepAEvent | StepBEvent | StepCEvent:
        ctx.send_event(StepAEvent(query="Query 1"))
        ctx.send_event(StepBEvent(query="Query 2"))
        ctx.send_event(StepCEvent(query="Query 3"))

    @step
    async def step_a(self, ctx: Context, ev: StepAEvent) -> StepACompleteEvent:
        print("Doing something A-ish")
        return StepACompleteEvent(result=ev.query)

    @step
    async def step_b(self, ctx: Context, ev: StepBEvent) -> StepBCompleteEvent:
        print("Doing something B-ish")
        return StepBCompleteEvent(result=ev.query)

    @step
    async def step_c(self, ctx: Context, ev: StepCEvent) -> StepCCompleteEvent:
        print("Doing something C-ish")
        return StepCCompleteEvent(result=ev.query)

    @step
    async def step_three(
        self,
        ctx: Context,
        ev: StepACompleteEvent | StepBCompleteEvent | StepCCompleteEvent,
    ) -> StopEvent:
        print("Received event ", ev.result)

        # wait until we receive 3 events
        events = ctx.collect_events(
            ev,
            [StepCCompleteEvent, StepACompleteEvent, StepBCompleteEvent],
        )
        if (events is None):
            return None

        # do something with all 3 results together
        print("All events received: ", events)
        return StopEvent(result="Done")
    
    # instantiate the workflow
basic_workflow = ConcurrentFlow(timeout=10, verbose=False)
draw_all_possible_flows(
        basic_workflow, 
        filename="workflows/basic_workflow.html"
    )

async def main():
    # run the workflow
    result = await basic_workflow.run()
    print(result)
    

if __name__ == "__main__":
    asyncio.run(main())