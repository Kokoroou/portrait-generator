from torchvision import transforms
import torch

MAX_STRUCTURE_CODE = 6
MAX_COLOR_CODE = 6


def gen_color(gene=None, structure_code=0, color_code=0, info={}):
    """
    Generate new gene with different color
    :param gene:
    :param structure_code:
    :param color_code:
    :param info: Necessary information of previous gene
    :return: torch.Tensor
        New gene with different color
    """
    instyle = info["instyle"]
    exstyle = info["exstyle"]
    generator = info["generator"]
    encoder = info["encoder"]

    weight = [structure_code / float(MAX_STRUCTURE_CODE)] * 7 + [color_code / float(MAX_COLOR_CODE)] * 11
    # new_gene = generator([instyle], exstyle[0:1], interp_weights=weight,
    #                       z_plus_latent=True, truncation=0.7, truncation_latent=0, use_res=True)
    new_gene = generator(instyle, exstyle, interp_weights=weight,
                         z_plus_latent=True, truncation=0.7, truncation_latent=0, use_res=True)
    new_gene = torch.clamp(new_gene.detach(), -1, 1)[1]

    return new_gene
