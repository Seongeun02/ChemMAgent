"""Wrapper for RXN4Chem functionalities."""

import abc
import ast
import re
import os
import json
from time import sleep
from typing import Optional
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.tools import BaseTool
from rxn4chemistry import RXN4ChemistryWrapper  # type: ignore

from typing import Dict, List
from utils import is_smiles, canonical_smiles
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

__all__ = ["RXNPredict", "RXNRetrosynthesis", "RXNPlanner"]

def is_path(path_str):
    """Check if a string is a valid path."""
    return os.path.exists(path_str)

class RXN4Chem(BaseTool):
    """Wrapper for RXN4Chem functionalities."""

    name: str
    description: str
    rxn4chem_api_key: Optional[str] = None
    rxn4chem: RXN4ChemistryWrapper = None
    base_url: str = "https://rxn.res.ibm.com"
    sleep_time: int = 5

    def __init__(self, rxn4chem_api_key):
        """Init object."""
        super().__init__()

        self.rxn4chem_api_key = rxn4chem_api_key
        self.rxn4chem = RXN4ChemistryWrapper(
            api_key=self.rxn4chem_api_key, base_url=self.base_url
        )
        self.rxn4chem.project_id = "672d20466dd3f27aba673612"

    @abc.abstractmethod
    def _run(self, smiles: str):  # type: ignore
        """Execute operation."""
        pass

    @abc.abstractmethod
    async def _arun(self, smiles: str):
        """Async execute operation."""
        pass

    @staticmethod
    def retry(times: int, exceptions, sleep_time: int = 5):
        """
        Retry Decorator.

        Retries the wrapped function/method `times` times if the exceptions
        listed in ``exceptions`` are thrown
        :param times: The number of times to repeat the wrapped function/method
        :type times: Int
        :param Exceptions: Lists of exceptions that trigger a retry attempt
        :type Exceptions: Tuple of Exceptions
        """

        def decorator(func):
            def newfn(*args, **kwargs):
                attempt = 0
                while attempt < times:
                    try:
                        sleep(sleep_time)
                        return func(*args, **kwargs)
                    except exceptions:
                        print(
                            "Exception thrown when attempting to run %s, "
                            "attempt %d of %d" % (func, attempt, times)
                        )
                        attempt += 1
                return func(*args, **kwargs)

            return newfn

        return decorator


class RXNPredict(RXN4Chem):
    """Predict reaction."""

    name: str = "ReactionPredict"
    description: str = (
        "Predict the outcome of a chemical reaction. "
        "Takes as input the SMILES of the reactants separated by a dot '.', "
        "returns SMILES of the products."
    )

    def _run(self, reactants: str) -> str:
        """Run reaction prediction."""
        
        if is_smiles(reactants):
            ## forward prediction for single smiles
            prediction_id = self.predict_reaction(reactants)
            results = self.get_results(prediction_id)
            product = results["productMolecule"]["smiles"]
            return product
        elif is_path(reactants):
            ## validation for retrosynthesis
            df = pd.read_csv(reactants)
            prediction_id = self.predict_reaction_batch(df['Reactants'].tolist())
            result = self.get_results_batch(prediction_id)
            df = self.process_result(df, result)
            df.to_csv(reactants)
            return reactants
        else:
            return "Incorrect input."

    @RXN4Chem.retry(10, KeyError)
    def predict_reaction(self, reactants: str) -> str:
        """Make api request."""
        response = self.rxn4chem.predict_reaction(reactants)
        if "prediction_id" in response.keys():
            return response["task_id"]
        else:
            raise KeyError

    @RXN4Chem.retry(10, KeyError)
    def predict_reaction_batch(self, reactants: list) -> str:
        """Make api request."""
        response = self.rxn4chem.predict_reaction_batch(
                    precursors_list=reactants
                    )
        if "task_id" in response.keys():
            return response["task_id"]
        else:
            raise KeyError

    @RXN4Chem.retry(10, KeyError)
    def get_results(self, prediction_id: str) -> str:
        """Make api request."""
        results = self.rxn4chem.get_predict_reaction_results(prediction_id)
        if "payload" in results["response"].keys():
            return results["response"]["payload"]["attempts"][0]
        else:
            raise KeyError
        
    @RXN4Chem.retry(10, KeyError)
    def get_results_batch(self, prediction_id: str) -> dict:
        """Make api request."""
        results = self.rxn4chem.get_predict_reaction_batch_results(prediction_id)
        return results

    def process_result(self, df:pd.DataFrame, result:dict) -> pd.DataFrame:
        forward_confidence = []
        forward_product = []
        match = []
        for i,reaction_prediction in enumerate(result["predictions"]):
            try : 
                reaction_smiles = reaction_prediction['smiles']
                reactants, product = reaction_smiles.split(">>")
                forward_confidence.append(reaction_prediction['confidence'])
                forward_product.append(product)
                if canonical_smiles(df['Product'][i]) == canonical_smiles(product):
                    match.append(True)
                else:
                    match.append(False)
            except Exception as e:
                logging.warning(f"Error processing prediction {i}: {e}")
                forward_confidence.append(None)
                forward_product.append(None)
                match.append(None)

        df['forward_confidence'] = forward_confidence
        df['forward_product'] = forward_product
        df['match'] = match
        return df

    async def _arun(self, cas_number):
        """Async run reaction prediction."""
        raise NotImplementedError("Async not implemented.")


class RXNRetrosynthesis(RXN4Chem):
    """Predict retrosynthesis."""

    name: str = "ReactionRetrosynthesis"
    description: str = (
        "Obtain the synthetic route to a chemical compound. "
        "Takes as input the SMILES of the product, returns recipe."
    )

    def __init__(self, rxn4chem_api_key): 
        """Init object."""
        super().__init__(rxn4chem_api_key)

    def _run(self, target: str) -> dict:
        """Run retrosynthesis prediction."""
        # Check that input is smiles
        if not is_smiles(target):
            return "Incorrect input."

        prediction_id = self.predict_retrosynthesis(target)
        paths = self.get_paths(prediction_id)
        df = self.get_results(paths)
        dir_df = 'env_tools/rxn4chem/retrosynthesis_result.csv'
        df.to_csv(dir_df)
        # path_img = self.visualize_path(paths[0])
        #procedure = self.get_action_sequence(paths[0])
        return dir_df

    async def _arun(self, cas_number):
        """Async run retrosynthesis prediction."""
        raise NotImplementedError("Async not implemented.")

    @RXN4Chem.retry(10, KeyError)
    def predict_retrosynthesis(self, target: str) -> str:
        """Make api request."""
        response = self.rxn4chem.predict_automatic_retrosynthesis(
            product=target,
            fap=0.6,
            max_steps=3,
            nbeams=10,
            pruning_steps=2,
            ai_model="12class-tokens-2021-05-14",
        )
        if "prediction_id" in response.keys():
            return response["prediction_id"]
        raise KeyError

    @RXN4Chem.retry(20, KeyError)
    def get_paths(self, prediction_id: str) -> str:
        """Make api request."""
        results = self.rxn4chem.get_predict_automatic_retrosynthesis_results(
            prediction_id
        )
        if "retrosynthetic_paths" not in results.keys():
            raise KeyError
        paths = results["retrosynthetic_paths"]
        if paths is not None:
            if len(paths) > 0:
                with open("env_tools/rxn4chem/retrosynthetic_paths.json", "w") as json_file:
                    json.dump(paths, json_file, indent=4)
                return paths
        if results["status"] == "PROCESSING":
            sleep(self.sleep_time * 2)
            raise KeyError
        raise KeyError

    def collect_reactions_from_retrosynthesis(self, tree: Dict) -> pd.DataFrame:
        """
        Parse retrosynthesis tree and create a DataFrame with reaction SMARTS, reactants, products, and confidence levels.
        
        Args:
            tree (Dict): Retrosynthesis tree with nodes containing 'smiles' and optionally 'confidence'.
        
        Returns:
            pd.DataFrame: A DataFrame with SMARTS, reactants, products, and confidence levels for each reaction.
        """
        data = []
        def parse_tree(node):
            if 'children' in node and len(node['children']):
                reactants = '.'.join([child['smiles'] for child in node['children']])
                product = node['smiles']
                smarts = f"{reactants}>>{product}"
                confidence = node.get('confidence', None)
                seq_id = node['sequenceId']
                
                data.append({
                    "SMARTS": smarts,
                    "Reactants": reactants,
                    "Product": product,
                    "Confidence": confidence,
                    "SequenceId": seq_id
                })
                
                for child in node['children']:
                    parse_tree(child)
    
        parse_tree(tree)
        df = pd.DataFrame(data)
        return df

    def get_results(self, paths: list) -> pd.DataFrame:
        """get dataframe of smarts, reactants, product, and confidence of each results"""
        df = pd.DataFrame(columns=["SMARTS", "Reactants", "Product", "Confidence", "SequenceId"])
        for index, tree in enumerate(paths):
            reactions_df = self.collect_reactions_from_retrosynthesis(tree)
            reactions_df["Path_Index"] = index
            df = pd.concat([df, reactions_df], ignore_index=True)
        
        return df

        
        
class RXNPlanner(RXN4Chem):
    """Reaction Planner"""

    name: str = "ReactionPlanner"
    description: str = (
        "Obtain the synthetic route to a chemical compound. "
        "Takes as input the rxn4chem synthesis id, returns recipe."
    )
    openai_api_key: str = ""

    def __init__(self, rxn4chem_api_key, openai_api_key):
        """Init object."""
        super().__init__(rxn4chem_api_key)
        self.openai_api_key = openai_api_key

    def _run(self, df_dir: str) -> str:
        """Run retrosynthesis prediction."""
        # Check that input is smiles
        if not is_path(df_dir):
            return "Incorrect input."

        df = pd.read_csv(df_dir)
        idx = []
        match = df['match'].tolist()
        idx = [index for index, value in enumerate(match) if value]
        
        for i in idx:
            seq_id = df['SequenceId'][i]
            procedure = self.get_action_sequence(seq_id)
            if procedure != seq_id and procedure != 'Tool error':
                break
        
        return procedure

    async def _arun(self, cas_number):
        """Async run retrosynthesis prediction."""
        raise NotImplementedError("Async not implemented.")

    def get_action_sequence(self, seq_id):
        """Get sequence of actions."""
        response = self.synth_from_sequence(seq_id)
        if "synthesis_id" not in response.keys():
            return seq_id

        synthesis_id = response["synthesis_id"]
        nodeids = self.get_node_ids(synthesis_id)
        if nodeids is None:
            return "Tool error"

        # Attempt to get actions for each node + product information
        real_nodes = []
        actions_and_products = []
        for node in nodeids:
            node_resp = self.get_reaction_settings(
                synthesis_id=synthesis_id, node_id=node
            )
            if "actions" in node_resp.keys():
                real_nodes.append(node)
                actions_and_products.append(node_resp)

        json_actions = self._preproc_actions(actions_and_products)
        llm_sum = self._summary_gpt(json_actions)
        return llm_sum

    @RXN4Chem.retry(20, KeyError)
    def synth_from_sequence(self, sequence_id: str) -> str:
        """Make api request."""
        response = self.rxn4chem.create_synthesis_from_sequence(sequence_id=sequence_id)
        if "synthesis_id" in response.keys():
            return response
        raise KeyError

    @RXN4Chem.retry(20, KeyError)
    def get_node_ids(self, synthesis_id: str):
        """Make api request."""
        response = self.rxn4chem.get_node_ids(synthesis_id=synthesis_id)
        if isinstance(response, list):
            if len(response) > 0:
                return response
        return KeyError

    @RXN4Chem.retry(20, KeyError)
    def get_reaction_settings(self, synthesis_id: str, node_id: str):
        """Make api request."""
        response = self.rxn4chem.get_reaction_settings(
            synthesis_id=synthesis_id, node_id=node_id
        )
        if "actions" in response.keys():
            return response
        elif "response" in response.keys():
            if "error" in response["response"].keys():
                if response["response"]["error"] == "Too Many Requests":
                    sleep(self.sleep_time * 2)
                    raise KeyError
            return response
        raise KeyError

    def _preproc_actions(self, actions_and_products):
        """Preprocess actions."""
        json_actions = {"number_of_steps": len(actions_and_products)}

        for i, actn in enumerate(actions_and_products):
            json_actions[f"Step_{i}"] = {}
            json_actions[f"Step_{i}"]["actions"] = actn["actions"]
            json_actions[f"Step_{i}"]["product"] = actn["product"]

        # Clean actions to use less tokens: Remove False, None, ''
        clean_act_str = re.sub(
            r"\'[A-Za-z]+\': (None|False|\'\'),? ?", "", str(json_actions)
        )
        json_actions = ast.literal_eval(clean_act_str)

        return json_actions

    def _summary_gpt(self, json: dict) -> str:
        """Describe synthesis."""
        llm = ChatOpenAI(  # type: ignore
            temperature=0.05,
            model_name="gpt-3.5-turbo-16k",
            request_timeout=2000,
            max_tokens=2000,
            openai_api_key=self.openai_api_key,
        )
        prompt = (
            "Here is a chemical synthesis described as a json.\nYour task is "
            "to describe the synthesis, as if you were giving instructions for"
            "a recipe. Use only the substances, quantities, temperatures and "
            "in general any action mentioned in the json file. This is your "
            "only source of information, do not make up anything else. Also, "
            "add 15mL of DCM as a solvent in the first step. If you ever need "
            'to refer to the json file, refer to it as "(by) the tool". '
            "However avoid references to it. \nFor this task, give as many "
            f"details as possible.\n {str(json)}"
        )
        return llm([HumanMessage(content=prompt)]).content

    def visualize_path(self, path):
        """Visualize path."""
        from aizynthfinder import reactiontree  # type: ignore

        rxn_dict = self._path_to_dict(path)
        tree = reactiontree.ReactionTree.from_dict(rxn_dict)
        return tree.to_image()

    def _path_to_dict(self, path):
        """Convert path to dict."""
        if len(path["children"]) != 0:
            in_stock = False
            rxn_smi = path["smiles"] + ">>"
            for prec in path["children"]:
                rxn_smi += prec["smiles"] + "."
            rxn_smi = rxn_smi[:-1]

            children = [
                {
                    "type": "reaction",
                    "hide": False,
                    "smiles": rxn_smi,
                    "is_reaction": True,
                    "metadata": {},
                    "children": [self._path_to_dict(c) for c in path["children"]],
                }
            ]
        else:
            in_stock = True
            children = []

        return {
            "type": "mol",
            "route_metadata": {"created_at_iteration": 1, "is_solved": True},
            "hide": False,
            "smiles": path["smiles"],
            "is_chemical": True,
            "in_stock": in_stock,
            "children": children,
        }
        