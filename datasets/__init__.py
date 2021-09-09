from .takekuchi.takekuchi import TakekuchiDataset
from .genea.genea import GeneaDataset

dataset_dict = {
    "takekuchi": TakekuchiDataset,
    "genea": GeneaDataset
}