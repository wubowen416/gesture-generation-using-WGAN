from .takekuchi.takekuchi import TakekuchiDataset
from .takekuchi_ext.takekuchi_ext import Takekuchi_extDataset
from .takekuchi_vel_amp.takekuchi_vel_amp import Takekuchi_vel_ampDataset
from .takekuchi_p.takekuchi_p import TakekuchiPrevDataset

dataset_dict = {
    "takekuchi": TakekuchiDataset,
    "takekuchi_ext": Takekuchi_extDataset,
    "takekuchi_vel_amp": Takekuchi_vel_ampDataset,
    "take_p": TakekuchiPrevDataset
}