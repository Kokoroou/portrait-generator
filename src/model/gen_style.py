import os
import random
from argparse import Namespace

from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from model.sampler.icp import ICPTrainer
from src.controller.GANController import *
from src.global_var import *

styles = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']


class StyleGenerator(object):
    # MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'checkpoint')  # Stand-alone
    MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), 'checkpoint')  # Streamlit

    def __init__(self):
        self.style = ""
        self.src_image_tensor = None
        self.encoder = None
        self.generator = None
        self.gene = None
        self.weight = None

    def load_encoder(self):
        model_path = os.path.join(self.MODEL_DIR, 'encoder.pt')
        ckpt = torch.load(model_path, map_location='cpu')
        opts = ckpt['opts']
        opts['checkpoint_path'] = model_path
        opts = Namespace(**opts)
        opts.device = device
        encoder = pSp(opts)
        encoder.eval()
        self.encoder = encoder.to(device)

    def load_dual_style_gan(self):
        if not self.style:
            raise Exception("Style for generator isn't added!")

        generator = DualStyleGAN(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2, res_index=6)
        generator.eval()
        ckpt = torch.load(os.path.join(self.MODEL_DIR, self.style, 'generator.pt'),
                          map_location=lambda storage, loc: storage)
        generator.load_state_dict(ckpt["g_ema"])
        self.generator = generator.to(device)

    def load_sampler_network(self):
        if not self.style:
            raise Exception("Style for generator isn't added!")

        icptc = ICPTrainer(np.empty([0, 512 * 11]), 128)
        icpts = ICPTrainer(np.empty([0, 512 * 7]), 128)
        ckpt = torch.load(os.path.join(self.MODEL_DIR, self.style, 'sampler.pt'),
                          map_location=lambda storage, loc: storage)
        icptc.icp.netT.load_state_dict(ckpt['color'])
        icpts.icp.netT.load_state_dict(ckpt['structure'])
        icptc.icp.netT = icptc.icp.netT.to(device)
        icpts.icp.netT = icpts.icp.netT.to(device)
        self.icptc = icptc
        self.icpts = icpts

    def modified_generator(self, style_index=-1, src_image_index=-1, structure_rate=0.0, color_rate=0.0):
        # Save style name
        if style_index == -1:
            style_index = random.randrange(0, len(styles))
        elif style_index < -1 or style_index >= len(styles):
            raise Exception(f"Style with index {style_index} doesn't exist!")

        # Check if style is changed
        if self.style != styles[style_index]:
            self.style = styles[style_index]

            # Modify generator and sampler network
            self.load_dual_style_gan()
            self.load_sampler_network()

        if os.path.exists(os.path.join(self.MODEL_DIR, self.style, 'refined_exstyle_code.npy')):
            exstyle_name = 'refined_exstyle_code.npy'
        else:
            exstyle_name = 'exstyle_code.npy'

        # Save tensor of source image
        src_images = np.load(os.path.join(self.MODEL_DIR, self.style, exstyle_name), allow_pickle=True).item()
        src_image_names = list(src_images.keys())
        if src_image_index == -1:
            src_image_index = random.randrange(0, len(src_image_names))
        elif src_image_index < -1 or src_image_index >= len(src_image_names):
            raise (Exception(f"Source image with index {src_image_index} doesn't exist!"))
        self.src_image_name = src_image_names[src_image_index]
        self.src_image_tensor = src_images[self.src_image_name]

        # Save weight
        self.weight = [structure_rate] * 7 + [color_rate] * 11

    def add_gene(self, gene):
        self.gene = gene

    def generate(self):
        if not self.encoder:
            self.load_encoder()
        if not self.generator:
            self.load_dual_style_gan()
        if not isinstance(self.gene, torch.Tensor):
            raise Exception("Gene for generator isn't added")

        # Print current state to screen
        print(f"Generate new portrait with style {self.style}:\n"
              f"--Source style image: {self.src_image_name}\n")

        with torch.no_grad():
            img_rec, instyle = self.encoder(self.gene, randomize_noise=False, return_latents=True,
                                            z_plus_latent=True, return_z_plus_latent=True, resize=False)
            img_rec = torch.clamp(img_rec.detach(), -1, 1)

            latent = torch.tensor(self.src_image_tensor).repeat(2, 1, 1).to(device)
            # latent[0] for both color and structure transfer and latent[1] for only structure transfer
            latent[1, 7:18] = instyle[0, 7:18]
            exstyle = self.generator.generator.style(
                latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
                latent.shape)

            img_gen, _ = self.generator([instyle.repeat(2, 1, 1)], exstyle, z_plus_latent=True,
                                        truncation=0.7, truncation_latent=0, use_res=True,
                                        interp_weights=[0.6] * 7 + [1] * 11)
            img_gen = torch.clamp(img_gen.detach(), -1, 1)
            # deactivate color-related layers by setting w_c = 0
            img_gen2, _ = self.generator([instyle], exstyle[0:1], z_plus_latent=True,
                                         truncation=0.7, truncation_latent=0, use_res=True,
                                         interp_weights=[0.6] * 7 + [0] * 11)
            img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)

        new_gene = img_gen[1]

        return new_gene


if __name__ == "__main__":
    # Generate a sample gene for everyone to understand
    FILE_NAME = '../../img/test/girl.png'

    img = Image.open(FILE_NAME)  # Open file
    img.thumbnail((1024, 1024))
    gene = image_to_gene(img)
    # print(gene.size())

    generator = StyleGenerator()
    generator.modified_generator()
    generator.add_gene(gene)

    next_gene = generator.generate()
    # print(new_gene.size())

    image = gene_to_image(next_gene)
    image.show()
    # show_image(image)
