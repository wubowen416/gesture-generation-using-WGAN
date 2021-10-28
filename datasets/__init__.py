from .takekuchi.takekuchi import TakekuchiDataset
from .takekuchi_ext.takekuchi_ext import Takekuchi_extDataset

dataset_dict = {
    "takekuchi": TakekuchiDataset,
    "takekuchi_ext": Takekuchi_extDataset
}