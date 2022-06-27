import random
from model.dualstylegan import DualStyleGAN
from src.controller.GANController import *
import torch
from PIL import Image
from src.global_var import *
from argparse import Namespace
from model.encoder.psp import pSp
from model.sampler.icp import ICPTrainer

styles = ['cartoon', 'caricature', 'anime', 'arcane', 'comic', 'pixar', 'slamdunk']


def gen_style(gene, style_id=0):
    info = {}

    # Select style type
    if style_id == -1:
        style_id = random.randint(0, len(styles)-1)
    style = styles[style_id]

    print("Create new image with style", style)

    # MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.getcwd())), 'checkpoint')  # Stand-alone
    MODEL_DIR = os.path.join(os.path.dirname(os.getcwd()), 'checkpoint')  # Streamlit
    weight = [0.75] * 7 + [1] * 11

    if os.path.exists(os.path.join(MODEL_DIR, style, 'refined_exstyle_code.npy')):
        exstyle_name = 'refined_exstyle_code.npy'
    else:
        exstyle_name = 'exstyle_code.npy'

    # Load DualStyleGAN
    generator = DualStyleGAN(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2, res_index=6)
    generator.eval()
    ckpt = torch.load(os.path.join(MODEL_DIR, style, 'generator.pt'), map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)

    # Load encoder
    model_path = os.path.join(MODEL_DIR, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder = encoder.to(device)

    # # Load extrinsic style code
    # exstyles = np.load(os.path.join(MODEL_DIR, style, exstyle_name), allow_pickle=True).item()

    # Load sampler network
    icptc = ICPTrainer(np.empty([0, 512 * 11]), 128)
    icpts = ICPTrainer(np.empty([0, 512 * 7]), 128)
    ckpt = torch.load(os.path.join(MODEL_DIR, style, 'sampler.pt'), map_location=lambda storage, loc: storage)
    icptc.icp.netT.load_state_dict(ckpt['color'])
    icpts.icp.netT.load_state_dict(ckpt['structure'])
    icptc.icp.netT = icptc.icp.netT.to(device)
    icpts.icp.netT = icpts.icp.netT.to(device)

    # Source image for exstyle
    src_images = np.load(os.path.join(MODEL_DIR, style, exstyle_name), allow_pickle=True).item()
    src_image_names = list(src_images.keys())
    image_index = random.randint(0, len(src_image_names))
    src_image_name = src_image_names[image_index]

    # # Add encoder
    # encoder = pSp(opts)  # Most time-consuming
    # encoder.eval()
    # encoder.to(device)

    # # Add generator
    # generator = DualStyleGAN(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2, res_index=6)
    # generator.eval()

    # # Add instyle, exstyle
    # latent = torch.tensor(src_images[src_image_name]).to(device)
    # instyle = list(torch.randn(6, 512).to(device))
    # exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
    #     latent.shape)

    with torch.no_grad():
        img_rec, instyle = encoder(gene, randomize_noise=False, return_latents=True,
                                   z_plus_latent=True, return_z_plus_latent=True, resize=False)
        img_rec = torch.clamp(img_rec.detach(), -1, 1)

        latent = torch.tensor(src_images[src_image_name]).repeat(2, 1, 1).to(device)
        # latent[0] for both color and structrue transfer and latent[1] for only structrue transfer
        latent[1, 7:18] = instyle[0, 7:18]
        exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
            latent.shape)

        img_gen, _ = generator([instyle.repeat(2, 1, 1)], exstyle, z_plus_latent=True,
                               truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.6] * 7 + [1] * 11)
        img_gen = torch.clamp(img_gen.detach(), -1, 1)
        # deactivate color-related layers by setting w_c = 0
        img_gen2, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                truncation=0.7, truncation_latent=0, use_res=True, interp_weights=[0.6] * 7 + [0] * 11)
        img_gen2 = torch.clamp(img_gen2.detach(), -1, 1)

    # Add all info to pass to next steps
    info["generator"] = generator
    info["encoder"] = encoder
    info["instyle"] = [instyle.repeat(2, 1, 1)]
    info["exstyle"] = exstyle

    new_gene = img_gen[1]

    return new_gene, info

if __name__ == "__main__":
    # Generate a sample gene for everyone to understand
    FILE_NAME = '../../img/test/girl.png'

    img = Image.open(FILE_NAME)  # Open file
    img.thumbnail((1024, 1024))
    gene = image_to_gene(img)
    # print(gene.size())

    new_gene, _ = gen_style(gene)
    # print(new_gene.size())

    image = gene_to_image(new_gene)
    image.show()
    # show_image(image)
