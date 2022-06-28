from io import BytesIO
from math import ceil, sqrt

import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot
from torchvision.utils import make_grid

from src.global_var import *


def image_to_gene(image):
    """
    Convert from a image to a gene vector
    :param image: PIL.PngImagePlugin.PngImageFile
        An image used for generators
    :return: torch.Tensor
        A vector indicate a gene
    """
    gene = transforms(image).unsqueeze(dim=0)
    return gene


def gene_to_image(gene):
    """
    Convert from a gene vector to a face image
    :param gene: torch.Tensor
        A vector indicate a gene
    :return: PIL.PngImagePlugin.PngImageFile
        An image used for generators
    """
    gene = torchvision.utils.make_grid(torch.cat([gene], dim=0), 1, 1)
    gene = ((gene.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)

    image = Image.fromarray(gene)
    return image


def image_to_byte(image):
    """
    Convert from a gene vector to a face image
    :param image: PIL.PngImagePlugin.PngImageFile
        An image
    :return: byte
        A byte array for saving image
    """
    byte_image = BytesIO()
    image.save(byte_image, "PNG")
    return byte_image.getvalue()


def show_image(gene, show_separately=False, window_title="No Title"):
    """
    Show image converted from gene
    :param gene: A gene or a list of genes
    :param show_separately: Show image directly or by pyplot
    :param window_title: Title of window which show image
    :return: None
    """
    if show_separately:
        if isinstance(gene, list):  # Case 'gene' is a list of genes
            for g in gene:
                gene_to_image(g).show()
        else:
            gene_to_image(gene).show()
    else:
        # Change title of window
        pyplot.figure(num=window_title)

        if isinstance(gene, list):  # Case 'gene' is a list of genes
            gene_count = len(gene)
            n = ceil(sqrt(gene_count))  # Length of square to plot images

            # Make a grid of genes
            grid = make_grid(gene, n)
            # Turn off axis
            pyplot.axis("off")
            # Plot raw pixel data
            pyplot.imshow(grid.permute(1, 2, 0))

        else:
            # Turn off axis
            pyplot.axis("off")
            # Plot raw pixel data
            # pyplot.imshow(gene.permute(1, 2, 0))
            pyplot.imshow(gene)
        pyplot.show()
