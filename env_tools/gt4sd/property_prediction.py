from gt4sd.properties.core import PropertyPredictor
from gt4sd.properties.crystals import CRYSTALS_PROPERTY_PREDICTOR_FACTORY
from gt4sd.properties.molecules import MOLECULE_PROPERTY_PREDICTOR_FACTORY
from gt4sd.properties.proteins import PROTEIN_PROPERTY_PREDICTOR_FACTORY
from gt4sd.properties.scorer import (
    MoleculePropertyPredictorScorer,
    PropertyPredictorScorer,
    ProteinPropertyPredictorScorer,
)
from typing import Any, Dict, List
PROPERTY_PREDICTOR_FACTORY: Dict[str, Any] = {
    **CRYSTALS_PROPERTY_PREDICTOR_FACTORY,
    **MOLECULE_PROPERTY_PREDICTOR_FACTORY,
    **PROTEIN_PROPERTY_PREDICTOR_FACTORY,
}

AVAILABLE_PROPERTY_PREDICTORS = sorted(PROPERTY_PREDICTOR_FACTORY.keys())
AVAILABLE_PROPERTY_PREDICTORS

class PropertyPrediction:
    
    def __init__(self, samples):
        self.samples = samples
        self.property = samples.columns()
        
        if 'algorithm' not in list(self.args.keys()):
            self.MoLeR_build()
        elif self.args['algorithm'] == 'RegressionTransformer':
            self.regression_transformer_build()
        elif self.args['algorithm'] == 'MoLeR':
            self.MoLeR_build()
    
    def regression_transformer_build(self):
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
        ## build MoLeR
        scaff = GetScaffoldForMol(self.args['smiles'])
        moler_config = MoLeRDefaultGenerator(scaffolds=scaff)
        self.algorithm = MoLeR(configuration=moler_config)
        self.algorithm_name = 'MoLeR'
    
    def sample(self):
        cg_mol_df = pd.DataFrame()
        smiles = []
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
        sample_df.to_csv('gt4sd/samples.csv')
        return 'gt4sd/samples.csv'
        

if __name__ == "__main__":
    with open('gt4sd/args_cg.pkl', 'rb') as f:
        args = pickle.load(f)
    
    cg = ConditionalGeneration(args, batch_size=32, num_samples=100)
    output_path = cg.sample()
    print(f"Samples saved to {output_path}")
    







