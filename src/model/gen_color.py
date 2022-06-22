from torchvision import transforms
from model.dualstylegan import DualStyleGAN

generator = DualStyleGAN(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2, res_index=6)
generator.eval()


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

    weight = [structure_code / 5.0] * 7 + [color_code / 5.0] * 11
    # new_gene = generator([instyle], exstyle[0:1], interp_weights=weight,
    #                       z_plus_latent=True, truncation=0.7, truncation_latent=0, use_res=True)
    new_gene = generator(instyle, exstyle, interp_weights=weight,
                         z_plus_latent=True, truncation=0.7, truncation_latent=0, use_res=True)

    return new_gene
