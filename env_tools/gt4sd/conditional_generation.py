from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformer, RegressionTransformerMolecules
)
from gt4sd.algorithms.generation.moler import MoLeR, MoLeRDefaultGenerator
import pandas as pd
import rdkit.Chem as Chem
from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from paccmann_chemistry.utils import disable_rdkit_logging
import pickle

disable_rdkit_logging()

class ConditionalGeneration:
    
    def __init__(self, args, batch_size=32, num_samples=100):
        self.args = args
        self.batch_size = batch_size
        self.num_samples = num_samples
        
        if 'algorithm' not in list(self.args.keys()):
            self.regression_transformer_build()
        elif self.args['algorithm'] == 'RegressionTransformer':
            self.regression_transformer_build()
        elif self.args['algorithm'] == 'MoLeR':
            self.MoLeR_build()
    
    def regression_transformer_build(self):
        self.args = self.default_config(self.args)
        ## build regression transformer
        config = RegressionTransformerMolecules(
                 algorithm_version=self.args['algorithm_version'],
                 search="sample",
                 temperature=self.args['temperature'],
                 tolerance=self.args['tolerance'],
                 sampling_wrapper=self.args['sampling_wrapper']
        )
        self.algorithm = RegressionTransformer(configuration=config,
                                          target=self.args['smiles'])
        self.algorithm_name = 'RegressionTransformer'
        
    def MoLeR_build(self):
        ## build MoLeR Scaffold
        mol = Chem.MolFromSmiles(self.args['smiles'])
        scaff_smi = Chem.MolToSmiles(GetScaffoldForMol(mol))
        moler_config = MoLeRDefaultGenerator(scaffolds=scaff_smi)
        self.algorithm = MoLeR(configuration=moler_config)
        self.algorithm_name = 'MoLeR'
        
    def default_config(self, args):
        default = {'algorithm': 'RegressionTransformer',
                   'algorithm_version': 'solubility',
                   'search': 'sample',
                   'temperature': 1.4,
                   'tolerance': 20.0,
                   'sampling_wrapper':{'fraction_to_mask': 0.2,
                                       'property_goal': {'<esol>': 0.234}}}
        for key, value in default.items():
            if key not in list(args.keys()):
                args[key] = value
        return args

    def sample(self):
        cg_mol_df = pd.DataFrame()
        smiles = []
        print('sampling molecules')
        while len(smiles) < self.num_samples:
            smis = list(self.algorithm.sample(self.batch_size))
            smis = list(zip(*smis))[0]
            smiles.extend([s for s in smis if Chem.MolFromSmiles(s) and s not in smiles])
        cg_mol_df = pd.concat([
            cg_mol_df,
            pd.DataFrame({
                "algorithm": [self.algorithm_name] * len(smiles),
                "smiles": smiles
                })
            ], axis=0)
        
        return self.save_samples(cg_mol_df)
        
    def save_samples(self, sample_df):
        sample_df.to_csv('env_tools/gt4sd/samples.csv')
        return 'env_tools/gt4sd/samples.csv'
        

if __name__ == "__main__":
    with open('env_tools/gt4sd/args_cg.pkl', 'rb') as f:
        args = pickle.load(f)

    print(args)
    cg = ConditionalGeneration(args, batch_size=16, num_samples=32)
    output_path = cg.sample()
    print(f"Samples saved to {output_path}")
    







