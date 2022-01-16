from .wgan.wgan import ConditionalWGAN
from .dev.wgan import MultiDConditionalWGAN

model_dict = {
    "wgan": ConditionalWGAN,
    "mdgan": MultiDConditionalWGAN
}