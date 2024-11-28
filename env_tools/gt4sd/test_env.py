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

print('env set')

from rdkit.Chem.Scaffolds.MurckoScaffold import GetScaffoldForMol
from rdkit import Chem
from gt4sd.properties import PropertyPredictorRegistry

gentrl_ddr1_smi = 'C12C=CC=NN1C(C#CC1=C(C)C=CC3C(NC4=CC(C(F)(F)F)=CC=C4)=NOC1=3)=CN=2'
gentrl_ddr1_mol = Chem.MolFromSmiles(gentrl_ddr1_smi)
sol = PropertyPredictorRegistry.get_property_predictor("esol")
gentrl_ddr1_sol = sol(gentrl_ddr1_mol)
gentrl_ddr1_scaff = GetScaffoldForMol(gentrl_ddr1_mol)
gentrl_ddr1_scaff_smi = Chem.MolToSmiles(GetScaffoldForMol(gentrl_ddr1_mol))
print(sol)

from gt4sd.algorithms.generation.moler import MoLeR, MoLeRDefaultGenerator
from gt4sd.algorithms.conditional_generation.regression_transformer import (
    RegressionTransformer, RegressionTransformerMolecules
)

moler_config_scaff = MoLeRDefaultGenerator(scaffolds=gentrl_ddr1_scaff_smi)
moler_alg_scaff = MoLeR(configuration=moler_config_scaff)


print(list(moler_alg_scaff.sample(8)))