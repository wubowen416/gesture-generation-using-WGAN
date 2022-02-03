from .takekuchi.takekuchi import TakekuchiDataset
from .takekuchi_ext.takekuchi_ext import TakekuchiExtDataset

dataset_dict = {
    "takekuchi": TakekuchiDataset,
    "takekuchi_ext": TakekuchiExtDataset
}