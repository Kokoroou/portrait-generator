from math import ceil, sqrt

from PIL import Image
from matplotlib import pyplot
from torchvision.transforms import PILToTensor, ToPILImage
from torchvision.utils import make_grid


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
        gene = PILToTensor()(image)
        return gene

    @staticmethod
    def gene_to_image(gene):
        """
        Convert from a gene vector to a face image
        :param gene: A vector indicate a gene
        :return: An image indicate a face
        """
        image = ToPILImage()(gene)
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
        FILE_NAME = '../img/test/girl.png'

        img = Image.open(FILE_NAME)  # Open file
        img.thumbnail((300, 300))  # Resize image and keep ratio

        gene = self.image_to_gene(img)
        return gene


class DualStyleGAN(GAN):
    def __init__(self):
        """
        Generate a new portrait by DualStyleGAN
        """
        super().__init__()

    def show_image(self, gene, show_separately=False, window_title="No Title"):
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
                    self.gene_to_image(g).show()
            else:
                self.gene_to_image(gene).show()
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
                pyplot.imshow(gene.permute(1, 2, 0))
            pyplot.show()

    def generate_style(self, gene):
        # Do something here...
        style = gene  # Gene of image from gene + random style
        return style

    def generate_color(self, gene):
        # Do something here...
        color = gene  # Gene of image from gene + random color
        return color

    def run(self):
        step = 1  # Variable to control current step
        gene0, gene1, gene2 = [], [], []
        styles = []
        colors = []

        while step:
            if step == 1:
                # Step 1: User import photo from their own library
                if isinstance(gene0, list):
                    gene0 = self.load_file()
                self.show_image(gene0, window_title="Photo imported")
                step += 1

            elif step == 2:
                # Step 2: Choose style
                if not styles:
                    for i in range(self.NEW_GENE_COUNT):
                        styles.append(self.generate_style(gene0))

                self.show_image(styles, window_title="New styles")

                index = int(input("You choose style: "))

                if index < 0 or index > self.NEW_GENE_COUNT:  # Do not go to next step
                    if index == -1:  # Go back to previous step
                        print("Go back to previous step!")
                        step -= 1
                    elif index == self.NEW_GENE_COUNT + 1:  # Refresh this step
                        print("Refresh styles!")
                        styles = []
                    elif index < -1 or index > self.NEW_GENE_COUNT + 1:  # Invalid number
                        print("Invalid style!")
                else:  # Continue to next step
                    step += 1
                    if index == 0:  # Use current style and continue
                        print("Do not change style!")
                        gene1 = gene0
                    else:  # Choose one of given new styles
                        gene1 = styles[index - 1]
                        self.show_image(gene1, window_title="Style chosen")

            elif step == 3:
                # Step 3: Change color
                if not colors:
                    for i in range(self.NEW_GENE_COUNT):
                        colors.append(self.generate_color(gene1))
                self.show_image(colors, window_title="New colors")
                index = int(input("You choose color: "))

                if index < 0 or index > self.NEW_GENE_COUNT:  # Do not go to next step
                    if index == -1:  # Go back to previous step
                        print("Go back to previous step!")
                        step -= 1
                    elif index == self.NEW_GENE_COUNT + 1:  # Refresh this step
                        print("Refresh colors!")
                        colors = []
                    elif index < -1 or index > self.NEW_GENE_COUNT + 1:  # Invalid number
                        print("Invalid color!")
                else:  # Continue to next step
                    step += 1
                    if index == 0:  # Use current style and continue
                        print("Do not change color!")
                        gene2 = gene1
                    else:  # Choose one of given new styles
                        gene2 = colors[index - 1]
                        self.show_image(gene2, window_title="Final image")

            elif step == 4:
                # Step 4: Save final image
                user_choice = input("Do you want to save image (Y/N): ")
                if user_choice == "Y" or user_choice == "y":
                    file_name = input("File name: ")
                    self.gene_to_image(gene2).save(file_name + ".png")
                step = 0
