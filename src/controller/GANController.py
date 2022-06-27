from io import BytesIO
from math import ceil, sqrt

import numpy as np
import matplotlib

# from matplotlib import pyplot
from torchvision.utils import make_grid

# from src.model.gen_style import *
from src.global_var import *
# import torchvision.transforms as transforms

from PIL import Image

# matplotlib.use('TkAgg')

import torch
import torchvision

STYLE = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']
MAX_WEIGHT = 5.0


def image_to_gene(image):
    """
    Convert from a image to a gene vector
    :param image: PIL.PngImagePlugin.PngImageFile
        An image used for generators
    :return: torch.Tensor
        A vector indicate a gene
    """
    # gene = PILToTensor()(image)

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
    # image = ToPILImage()(gene)

    gene = torchvision.utils.make_grid(torch.cat([gene], dim=0), 1, 1)
    gene = ((gene.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(np.uint8)

    image = Image.fromarray(gene)

    # print(type(image))
    # print(image.shape)

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

#
# def generate_random_style(gene, add_style_name=False):
#     """
#     Generate new image from gene with a random style by Dual Style GAN
#     :param gene: torch.Tensor
#         A vector indicate a gene
#     :param add_style_name: bool
#         Add name of used style to new image
#     :return: torch.Tensor
#         A vector indicate new gene
#     """
#     style = random.choice(STYLE)  # Style to generate new image
#
#     new_gene = gen_style(gene, style)  # Gene of image from gene + random style
#
#     if add_style_name:
#         # Modify info of text to add
#         text = style.capitalize()
#         font = ImageFont.truetype("Times New Roman", 50)
#         spacing = 4
#
#         image = gene_to_image(new_gene)  # Change to type PIL
#
#         # Call draw Method to add 2D graphics in an image
#         draw = ImageDraw.Draw(image)
#
#         # Get size to calculate
#         width, height = image.size
#         text_width, text_height = draw.textsize(text=text, font=font, spacing=spacing)
#
#         # Draw text into image
#         draw.text(((width - text_width) // 2, height - text_height), text=text, font=font, spacing=spacing)
#
#         new_gene = image_to_gene(image)  # Change back to type of gene
#     return new_gene


def generate_color(gene, structure_weight, color_weight):
    """
    Generate new image from gene with new color by Dual Style GAN
    :param gene: torch.Tensor
        A vector indicate a gene
    :param structure_weight: int
        Weights of structure codes
    :param color_weight: int
        Weights of color codes
    :return: torch.Tensor
        A vector indicate new gene
    """
    weight = [structure_weight / MAX_WEIGHT] * 7 + [color_weight / MAX_WEIGHT] * 11
    color = gene  # Gene of image from gene + random color
    return color


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


def transfer(img_arr):
    return ((img_arr.detach().numpy().transpose(1, 2, 0) + 1.0) * 127.5).astype(numpy.uint8)
