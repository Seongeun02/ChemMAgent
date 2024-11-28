from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
import dotenv 
import getpass
import os

from tools.chemspace import *
from tools.safety import *
from tools.converters import *
from tools.rdkit import *

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
        Query2SMILES(api_keys['CHEMSPACE_API_KEY']),
        GetMoleculePrice(api_keys['CHEMSPACE_API_KEY']),
        Query2CAS(),
        SMILES2Name(),
        MolSimilarity(),
        SMILES2Weight(),
        FuncGroups(),
        ControlChemCheck(),
        SimilarControlChemCheck(),
    ]
    
    return tools


def build_molecular_agent(model_type='gpt-4o'):
    _set_env("OPENAI_API_KEY")
    _set_env("CHEMSPACE_API_KEY")
    model = build_model(model_type)
    tools = make_tools()
    agent_executor = create_react_agent(model, tools)
    
    return agent_executor


class MolecularAgent(BaseTool):
    name: str = "MolecularAgent"
    description: str = (
                        "A molecular agent equipped with tools for querying, analyzing, and processing chemical information."
                        "Handles tasks such as molecule similarity, functional group detection, property calculation, and controlled substance checks.")
    model_type: str = "gpt-4o"
    molecular_agent: AgentExecutor = None

    def __init__(self, model_type='gpt-4o'):
        super().__init__()
        self.model_type = model_type
        self.molecular_agent = build_molecular_agent(model_type)
        
    def _run(self, query: str) -> str:
        return self.molecular_agent.invoke(query)

    async def _arun(self, query: str) -> str:
        """Use the tool asynchronously."""
        raise NotImplementedError()