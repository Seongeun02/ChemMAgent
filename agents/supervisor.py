from agents.search_agent import SearchAgent
from agents.molecular_agent import MolecularAgent
from agents.reaction_agent import ReactionAgent
from agents.prompts import PLANNER_PROMPT, REPLANNER_PROMPT
from langchain import hub
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, START
from langgraph.graph import END

import operator
from typing import Annotated, List, Tuple, Union, Literal
from typing_extensions import TypedDict
from pydantic import BaseModel, Field

import json
import logging
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


class PlanExecute(TypedDict):
    input: str
    plan: List[str]
    past_steps: Annotated[List[Tuple], operator.add]
    response: str

class Plan(BaseModel):
    """Plan to follow in future"""
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    
class Response(BaseModel):
    """Response to user."""
    response: str

class Act(BaseModel):
    """Action to perform."""
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. "
        "If you need to further use tools to get the answer, use Plan."
    )

def should_end(state: PlanExecute):
    if "response" in state and state["response"]:
        return END
    else:
        return "supervisor"




class ChemAgent:
    def __init__(
        self,
        tool_list=['search', 'molecule', 'reaction'],
        supervisor_model="gpt-4-turbo-preview",
        planner_model="gpt-4o",
        temp=0,
        max_iterations=40,
        verbose=True,
    ):
        self.tool_list = tool_list
        self.tools = self.make_tools()
        
        self.supervisor_model = supervisor_model
        self.supervisor = self.create_supervisor()
    
        self.planner_model = planner_model
        self.temp = temp
        self.planner = self.create_planner()
        self.replanner = self.create_replanner()
        self.config = {"recursion_limit": max_iterations}
        self.app = self.build_graph()
    
    def make_tools(self):
        tools = []
        if 'search' in self.tool_list:
            tools += [SearchAgent()]
        if 'molecule' in self.tool_list:
            tools += [MolecularAgent()]
        if 'reaction' in self.tool_list:
            tools += [ReactionAgent()]
        if 'generation' in self.tool_list:
            pass
            # tools += [GenerativeAgent()]
        return tools
    
    def create_supervisor(self):
        prompt = hub.pull("ih/ih-react-agent-executor")
        prompt.pretty_print()
        llm = ChatOpenAI(model = self.supervisor_model)
        agent_executor = create_react_agent(llm, self.tools, state_modifier=prompt)
        return agent_executor
    
    def create_planner(self):
        planner_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    PLANNER_PROMPT,
                ),
                ("placeholder", "{messages}"),
            ]
        )

        planner_prompt = planner_prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        planner_prompt = planner_prompt.partial(tool_description=", ".join([tool.description for tool in self.tools]))
        planner = planner_prompt | ChatOpenAI(
            model=self.planner_model, temperature=self.temp
        ).with_structured_output(Plan)
        return planner

    def create_replanner(self):
        replanner_prompt = ChatPromptTemplate.from_template(
            REPLANNER_PROMPT
        )
        replanner_prompt = replanner_prompt.partial(tool_names=", ".join([tool.name for tool in self.tools]))
        replanner_prompt = replanner_prompt.partial(tool_description=", ".join([tool.description for tool in self.tools]))
        replanner = replanner_prompt | ChatOpenAI(
            model = self.planner_model, temperature=self.temp
        ).with_structured_output(Act)
        return replanner

    async def plan_step(self, state: PlanExecute):
        plan = await self.planner.ainvoke({"messages": [("user", state["input"])]})
        return {"plan": plan.steps}

    async def replan_step(self, state: PlanExecute):
        output = await self.replanner.ainvoke(state)
        if isinstance(output.action, Response):
            return {"response": output.action.response}
        else:
            return {"plan": output.action.steps}

    async def execute_step(self, state: PlanExecute):
        plan = state["plan"]
        plan_str = "\n".join(f"{i+1}. {step}" for i, step in enumerate(plan))
        task = plan[0]
        task_formatted = f"""For the following plan:
        {plan_str}\n\nYou are tasked with executing step {1}, {task}."""
        agent_response = await self.supervisor.ainvoke(
            {"messages": [("user", task_formatted)]}
        )
        return {
            "past_steps": [(task, agent_response["messages"][-1].content)],
        }
        

    def build_graph(self):
        workflow = StateGraph(PlanExecute)
        workflow.add_node("planner", self.plan_step)
        workflow.add_node("supervisor", self.execute_step)
        workflow.add_node("replan", self.replan_step)

        workflow.add_edge(START, "planner")
        workflow.add_edge("planner", "supervisor")
        workflow.add_edge("supervisor", "replan")

        workflow.add_conditional_edges(
            "replan",
            should_end,
            ["supervisor", END],
        )

        app = workflow.compile()
        return app

    async def run(self, inputs):
        async for event in self.app.astream({"input": inputs}, config=self.config):
            for k, v in event.items():
                if k != "__end__":
                    print(v)



