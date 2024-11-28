from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
import dotenv 
import getpass
import os

from tools.search import *

def _set_env(var: str):
    if not os.environ.get(var):
        os.environ[var] = getpass.getpass(f"{var}: ")

def build_model(model_type='gpt-4o'):
    model = ChatOpenAI(model=model_type)
    return model

def make_tools():
    env_vars = dotenv.dotenv_values(".env")  
    api_keys = {key: value for key, value in env_vars.items()}
    tools = [
            Wikipedia(),
            WebSearch('tavily', api_keys['TAVILY_API_KEY'])
            ]
    
    return tools


def build_search_agent(model_type='gpt-4o'):
    _set_env("OPENAI_API_KEY")
    _set_env("TAVILY_API_KEY")
    model = build_model(model_type)
    tools = make_tools()
    agent_executor = create_react_agent(model, tools)
    
    return agent_executor


class SearchAgent(BaseTool):
    name: str = "SearchAgent"
    description: str = (
        "A search agent combining WebSearch and Wikipedia Search to efficiently retrieve information. "
        "Dynamically selects the appropriate tool to answer questions or explore topics. "
    )
    model_type: str = "gpt-4o"
    search_agent: AgentExecutor = None

    def __init__(self, model_type='gpt-4o'):
        super().__init__()
        self.model_type = model_type
        self.search_agent = build_search_agent(model_type)
        
    def _run(self, query: str) -> str:
        return self.search_agent.invoke(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()