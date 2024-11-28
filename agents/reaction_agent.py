from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.output_parsers.openai_functions import JsonOutputFunctionsParser
import dotenv 
import getpass
import functools
import operator
from typing import Annotated, List, Tuple, Union
from utils import is_smiles, canonical_smiles
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_openai.chat_models import ChatOpenAI
import functools
from typing_extensions import TypedDict
from typing import Optional, Any
from tools.rxn4chem import RXNPredict, RXNRetrosynthesis, RXNPlanner
from langgraph.graph import StateGraph, START
from langgraph.graph import END
import os
import pandas as pd
import logging
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)

def is_path(path_str):
    """Check if a string is a valid path."""
    return os.path.exists(path_str)

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def build_model(model_type='gpt-4o'):
    model = ChatOpenAI(model=model_type)
    return model

def make_tools():
    env_vars = dotenv.dotenv_values(".env")  
    api_keys = {key: value for key, value in env_vars.items()}
    tools = {
        'ReactionPredict': RXNPredict(api_keys['RXN4CHEMISTRY_API_KEY']),
        'ReactionRetrosynthesis': RXNRetrosynthesis(api_keys['RXN4CHEMISTRY_API_KEY']),
        'ReactionPlanner': RXNPlanner(api_keys['RXN4CHEMISTRY_API_KEY'],
                                      api_keys['OPENAI_API_KEY'])
    }
    
    return tools

class ReactionTeamState(TypedDict):
    messages: Annotated[List[BaseMessage], operator.add]
    team_members: List[str]
    next: str
    

def create_reaction_agent(llm: ChatOpenAI, system_prompt, tools) -> str:
    _set_env("OPENAI_API_KEY")
    _set_env("RXN4CHEMISTRY_API_KEY")
    
    """LLM-based Router."""
    options = ["FINISH"] + tools
    function_def = {
        "name": "route",
        "description": "Select next role.",
        "parameters": {
            "title": "routeSchema",
            "type": "object",
            "properties": {
                "next": {
                    "title": "Next",
                    "anyOf": [
                        {"enum": options},
                    ],
                },
            },
            "required": ["next"],
        },
    }
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
            (
                "system",
            """
            Based on the above conversation, which option should be executed next?: {options}

            Extract the SMILES notation from the following query to execute a tool.
            Ensure that only the SMILES string is returned and nothing else.
            Provide detailed, step-by-step reasoning and ensure all conclusions are validated.
            For retrosynthesis, outline clear, actionable steps.
            Deliver your results in an educational, precise, and scientific tone.
            """,
            ),
        ]
    ).partial(options=str(options), team_members=", ".join(tools))
    return (
        prompt
        | llm.bind_functions(functions=[function_def], function_call="route")
        | JsonOutputFunctionsParser()
    )

def where_to_go(state):
    last_message = state['messages'][-1].content
    if is_smiles(last_message):
        print('route to supervisor')
        return "supervisor"
    elif is_path(last_message):
        df = pd.read_csv(last_message)
        if True in df['match'].tolist():
            print('route to planner')
            return "ReactionPlanner"
        else:
            print('route to retrosynthesis')
            return "ReactionRetrosynthesis"
    else:
        print('route to supervisor')
        return "supervisor"
    
    
def agent_node(state, tool, name):
    print('tool', tool)
    print('state', state)
    target = state['messages'][-1].content
    result = tool.invoke(target)

    return {"messages": [HumanMessage(content=result, name=name)]}


class ReactionAgent(BaseTool):
    name: str = "ReactionAgent"
    description: str = (
                        "A reaction agent equipped with tools for prediction of forward reaction prediction and retrosynthesis prediction."
                        "Provide SMILES representations as input to predict chemical reaction outcomes and plan retrosynthetic pathways.")
    model_type: str = "gpt-4o"
    reaction_agent: AgentExecutor = None
    reaction_chain: Optional[Any] = None

    def __init__(self, model_type='gpt-4o'):
        super().__init__()
        
        self.model_type = model_type
        system_prompt = "You are an expert organic chemist equipped with reaction related tools to solve problems. "
        llm = llm = ChatOpenAI(model=self.model_type)
        tool_name = ['ReactionPredict', 'ReactionRetrosynthesis']
        self.reaction_agent = create_reaction_agent(llm, system_prompt, tool_name)
        self.build_graph()
        
    
    def build_graph(self):
        reaction_graph = StateGraph(ReactionTeamState)
        
        tools = make_tools()
        reaction_graph.add_node("ReactionPredict", 
                                functools.partial(agent_node, tool= tools['ReactionPredict'], name="ReactionPredict"))
        reaction_graph.add_node("ReactionRetrosynthesis", 
                                functools.partial(agent_node, tool=tools['ReactionRetrosynthesis'], name="ReactionRetrosynthesis"))
        reaction_graph.add_node("ReactionPlanner",
                                functools.partial(agent_node, tool =tools['ReactionPlanner'], name="ReactionPlanner"))
        reaction_graph.add_node("supervisor", self.reaction_agent)

        reaction_graph.add_edge("ReactionPlanner", "supervisor")
        reaction_graph.add_edge("ReactionRetrosynthesis", "ReactionPredict")
        reaction_graph.add_conditional_edges(
            "supervisor",
            lambda x: print(f"Supervisor decided: {x['next']}") or x["next"],
            {"ReactionPlanner": "ReactionPlanner", 
            "ReactionRetrosynthesis": "ReactionRetrosynthesis", 
            "FINISH": END},
        )

        reaction_graph.add_conditional_edges(
            "ReactionPredict",
            lambda x: print(f"where_to_go result: {where_to_go(x)}") or where_to_go(x),
            { 
                "supervisor": "supervisor",
                "ReactionPlanner": "ReactionPlanner",
                "ReactionRetrosynthesis": "ReactionRetrosynthesis"
            }
        )

        reaction_graph.set_entry_point("supervisor")
        app = reaction_graph.compile()
        
        def enter_chain(message: str):

            results = {
                #"messages": [HumanMessage(content=msg) for msg in message],
                "messages": [HumanMessage(content=message)],
            }
            return results

        self.reaction_chain = enter_chain | app
        
    def _run(self, query: str) -> str:
        print("--Execute Reaction Agent--")
        s_list = []
        for s in self.reaction_chain.stream(
            query, {"recursion_limit": 20}
        ):
            if "__end__" not in s:
                s_list.append(s)
                print(s)
                print("---")
                
        print('end', s_list)
        return s_list[-1]['messages'].content

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()