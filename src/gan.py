class GAN(object):
    NEW_GENE_COUNT = 15

    def __init__(self):
        """
        Generate picture by Generative Adversarial Network
        """
        self.gene = None

    @staticmethod
    def picture_to_gene(picture):
        """
        Mapping from a picture to a gene vector
        :param picture: A file of picture (TODO: Change this description when using with real data)
        :return: A vector indicate a gene
        """
        # Do something here...
        gene = None
        return gene

    @staticmethod
    def gene_to_picture(gene):
        """
        Mapping from a gene vector to a picture
        :param gene: A vector indicate a gene
        :return: A file of picture (TODO: Change this description when using with real data)
        """
        # Do something here...
        picture = None
        return picture

    def get_input(self):
        # Do something here...
        picture = None
        self.gene = self.picture_to_gene(picture)
        return self.gene


class DualStyleGAN(GAN):
    def __init__(self):
        """
        Generate a new portrait by DualStyleGAN
        """
        super().__init__()
        self.style = None

    def show_picture(self, gene):
        # Return the picture with the given gene
        # Do something here...
        # Run function gene_to_picture(gene) and show with plt or another library
        pass

    def show_style(self):
        # Randomly show some of picture in library
        # Do something here...
        style = None  # Gene of the photo which is chosen as style
        user_choice = False
        if user_choice:
            self.style = style

        self.show_picture(style)
        return style

    def show_color(self):
        # Show picture of self.gene with random color
        # Do something here...
        color = None  # Gene of picture from self.gene with random color
        user_choice = False
        if user_choice:
            self.gene = color

        self.show_picture(color)
        return color

    def generate(self):
        # Generate new gene with chosen style
        # Do something here...

        self.gene = None
        return self.gene

    def run(self):
        # Step 1: User import photo from their own library
        gene0 = self.get_input()
        self.show_picture(gene0)

        # Step 2: Choose style
        for i in range(self.NEW_GENE_COUNT):
            self.show_style()
        gene1 = self.generate()
        self.show_picture(gene1)

        # Step 3: Change color
        for i in range(self.NEW_GENE_COUNT):
            self.show_color()
        gene2 = self.gene
        self.show_picture(gene2)
