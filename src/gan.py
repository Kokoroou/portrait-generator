from PIL import Image
from matplotlib import pyplot
import numpy as np
from math import ceil, sqrt


class GAN(object):
    DIRECTORY = "../img/"
    NEW_GENE_COUNT = 16  # Number of new genes to generate in each step

    def __init__(self):
        """
        Generate image by Generative Adversarial Network
        """

    @staticmethod
    def image_to_gene(image):
        """
        Convert from a face image to a gene vector
        :param image: An image indicate a face
        :return: A vector indicate a gene
        """
        gene = np.asarray(image)
        return gene

    @staticmethod
    def gene_to_image(gene):
        """
        Convert from a gene vector to a face image
        :param gene: A vector indicate a gene
        :return: An image indicate a face
        """
        image = Image.fromarray(gene)
        return image

    def load_image(self, sub_dir, file_name):
        # Load image from file
        image = Image.open(self.DIRECTORY + "/" + sub_dir + "/" + file_name)

        # Convert to RGB, if needed
        image = image.convert('RGB')

        # Convert to array
        gene = self.image_to_gene(image)
        return gene

    def load_file(self):
        gene = None
        return gene


class DualStyleGAN(GAN):
    def __init__(self):
        """
        Generate a new portrait by DualStyleGAN
        """
        super().__init__()

    def show_image(self, gene, show_separately=False):
        """
        Show image converted from gene
        :param gene: A gene or a list of genes
        :param show_separately: Show image directly or by pyplot
        :return: None
        """
        if show_separately:
            if isinstance(gene, list):  # Case 'gene' is a list of genes
                for g in gene:
                    self.gene_to_image(g).show()
            else:
                self.gene_to_image(gene).show()
        else:
            if isinstance(gene, list):  # Case 'gene' is a list of genes
                gene_count = len(gene)
                n = ceil(sqrt(gene_count))  # Length of square to plot images

                # Plot a list of images
                for i in range(gene_count):
                    # Define subplot
                    pyplot.subplot(n, n, 1 + i)
                    # Turn off axis
                    pyplot.axis('off')
                    # Plot raw pixel data
                    pyplot.imshow(gene[i])
                pyplot.show()
            else:
                pyplot.axis("off")
                pyplot.imshow(gene)
                pyplot.show()

    def generate_style(self, gene):
        # Do something here...
        style = None  # Gene of image from gene + random style
        return style

    def generate_color(self, gene):
        # Do something here...
        color = None  # Gene of image from gene + random color
        return color

    def run(self):
        # Step 1: User import photo from their own library
        gene0 = self.load_file()
        self.show_image(gene0)

        # Step 2: Choose style
        styles = []
        for i in range(self.NEW_GENE_COUNT):
            styles.append(self.generate_style(gene0))
        self.show_image(styles)
        index = int(input("You choose style: "))
        gene1 = styles[index]
        self.show_image(gene1)

        # Step 3: Change color
        colors = []
        for i in range(self.NEW_GENE_COUNT):
            colors.append(self.generate_color(gene1))
        self.show_image(colors)
        index = int(input("You choose color: "))
        gene2 = colors[index]
        self.show_image(gene2)
