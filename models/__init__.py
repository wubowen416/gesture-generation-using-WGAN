from .wgan.wgan import ConditionalWGAN
from .mdgan.mdgan import MultiDConditionalWGAN

model_dict = {
    "wgan": ConditionalWGAN,
    "mdgan": MultiDConditionalWGAN
}