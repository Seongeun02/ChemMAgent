"""Wrapper for GT4SD functionalities."""
import os
import abc
from time import sleep
from typing import Optional

import subprocess
import pickle
import pandas as pd
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from utils import is_smiles

__all__ = ["conditional_generation", "property_prediction"]

class GT4SD(BaseTool):
    """Wrapper for RXN4Chem functionalities."""

    name: str = "GT4SD base class"
    description: str
    sleep_time: int = 5

    def __init__(self):
        """Init object."""
        super().__init__()

        try:
            run_script_in_env("~/miniforge3/envs/gt4sd", "env_tools/gt4sd/test_env.py")
        except Exception as e:
            print(f"An error occurred: {e}")

    @abc.abstractmethod
    def _run(self):  # type: ignore
        """Execute operation."""
        pass
    
    @abc.abstractmethod
    async def _arun(self):
        """Async run reaction prediction."""
        raise NotImplementedError("Async not implemented.")


class conditional_generation(GT4SD):
    """conditional generation"""

    name: str = "conditional_generation"
    description: str = (
        "Generate the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )

    def _run(self, args: dict) -> pd.DataFrame:
        """Run conditional generation"""
        # Check that input is smiles
        if not is_smiles(args['smiles']):
            return "Incorrect input."

        self.save_args(args)
        run_script_in_env("~/miniforge3/envs/gt4sd", 'env_tools/gt4sd/conditional_generation.py')
        results = self.get_results()
        return results

        
    def save_args(self, args):
        #if 'algorithm' not in list(args.keys()):
        #    args = self.default_config(args)
        
        with open('env_tools/gt4sd/args_cg.pkl', 'wb') as file:
            pickle.dump(args, file)

    
    def get_results(self) -> pd.DataFrame:
        results = pd.read_csv('env_tools/gt4sd/samples.csv')

        return results

    def default_config(self, args):
        args['algorithm'] = 'RegressionTransformer'
        args['algorithm_version'] = 'solubility'
        args['search'] = 'sample'
        args['temperature'] = 1.4
        args['tolerance'] = 20.0
        args['sampling_wrapper'] = {'fraction_to_mask': 0.2, 
                                    'property_goal': {'<esol>': 0.234}}
        return args
        
    async def _arun(self, cas_number):
        """Async run reaction prediction."""
        raise NotImplementedError("Async not implemented.")
    
    
class property_prediction(GT4SD):
    """property prediction"""

    name: str = "property_prediction"
    description: str = (
        "Predict the properties of a molecule. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )

    def _run(self, target_property: list, samples: pd.DataFrame) -> pd.DataFrame:
        """Run property prediction"""
        # Check that input is smiles
        validity = []
        for smi in samples['smiles']:
            validity.append(is_smiles(smi))
        samples['validity'] = validity

        for p in target_property:
            samples[p] = None
            
        self.save_samples(samples)
        run_script_in_env("~/miniforge3/envs/gt4sd", 'env_tools/gt4sd/property_prediction.py')
        results = self.get_results()
        return results
        
    def save_samples(self, samples):
        samples.to_csv('env_tools/gt4sd/samples.csv')

    def get_results(self) -> pd.DataFrame:
        results = pd.read_csv('env_tools/gt4sd/samples.csv')
        return results
        
    async def _arun(self, cas_number):
        """Async run reaction prediction."""
        raise NotImplementedError("Async not implemented.")
    
def run_script_in_env(env_path, script_path):
    subprocess.run(
        f"source ~/miniforge3/bin/activate && conda run -p {env_path} python {script_path}",
        shell=True,
        check=True
    )