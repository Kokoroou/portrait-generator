import argparse
import os
import random
import tkinter
from argparse import Namespace

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms

from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from util import save_image, load_image, visualize

import time

from src.controller.GANController import *

matplotlib.use('TkAgg')


def stop_watch(recent_job, time_before, time_after):
    print(f"Time to do {recent_job} is {time_after - time_before}s")


def get_info(var_name, var):
    print(f"Type of {var_name} is {type(var)}")
    print(f"{var_name} = {var}")
    print()


class CLIOptions:
    """
    | Config options for write in Command Line Interface (CLI)
    """
    def __init__(self, src):
        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        self.parser.add_argument("--content", type=str, default=src, help="path of the content image")
        self.parser.add_argument("--style", type=str, default='cartoon', help="target style type")
        self.parser.add_argument("--style_id", type=int, default=51, help="the id of the style image")
        self.parser.add_argument("--truncation", type=float, default=0.75,
                                 help="truncation for intrinsic style code (content)")
        self.parser.add_argument("--weight", type=float, nargs=18, default=[0.75] * 7 + [1] * 11,
                                 help="weight of the extrinsic style")
        self.parser.add_argument("--name", type=str, default='anime_transfer',
                                 help="filename to save the generated images")
        self.parser.add_argument("--preserve_color", action="store_true",
                                 help="preserve the color of the content image")
        self.parser.add_argument("--model_path", type=str, default='.././checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt',
                                 help="name of the saved Dual Style GAN")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")

        self.option = None

    def parse(self):
        # Save all options' value in an Namespace
        self.option = self.parser.parse_args()

        if self.option.exstyle_name is None:
            # Use refined version of extrinsic style if exist
            if os.path.exists(os.path.join(self.option.model_path, self.option.style, 'refined_exstyle_code.npy')):
                self.option.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.option.exstyle_name = 'exstyle_code.npy'

        # args = vars
        # print('Load options')
        # for name, value in sorted(args.items()):
        #    print('%s: %s' % (str(name), str(value)))

        return self.option


def run_alignment(args):
    """
    Recognize and align a face
    :param args: Namespace
        Options configured in CLIOptions
    :return: PIL.PngImagePlugin.PngImageFile
        An aligned face
    """
    # Import model to recognize face
    import dlib
    from model.encoder.align_all_parallel import align_face

    model_name = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')

    # Check if pre trained model is downloaded
    if not os.path.exists(model_name):
        import wget
        import bz2

        # Download dlib's pre trained model for detect 68 points in human face
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', model_name + '.bz2')

        # Unzip the downloaded file
        zipfile = bz2.BZ2File(model_name + '.bz2')

        # Write data to new file
        data = zipfile.read()
        open(model_name, 'wb').write(data)

    # Align face image
    predictor = dlib.shape_predictor(model_name)
    aligned_image = align_face(filepath=args.content, predictor=predictor)

    return aligned_image


def gen_color(src):
    device = "cpu"

    parser = CLIOptions(src)
    args = parser.parse()

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    generator = DualStyleGAN(size=1024, style_dim=512, n_mlp=8, channel_multiplier=2, res_index=6)
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name),
                      map_location=lambda storage, loc: storage)

    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)

    model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')  # Second time-consuming
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)  # Most time-consuming

    encoder.eval()
    encoder.to(device)

    print(encoder)

    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()

    #print('Load models successfully!')

    with torch.no_grad():
        # Load content image
        if args.align_face:
            image = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
            image = F.adaptive_avg_pool2d(image, 1024)
        else:
            image = load_image(args.content).to(device)

        # reconstructed content image and its intrinsic style code
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(image, 256), randomize_noise=False, return_latents=True,
                                   z_plus_latent=True, return_z_plus_latent=True, resize=False)  # Most time-consuming

        img_rec = torch.clamp(img_rec.detach(), -1, 1)

        style_name = list(exstyles.keys())[args.style_id]
        latent = torch.tensor(exstyles[style_name]).to(device)


        if args.preserve_color:
            latent[:, 7:18] = instyle[:, 7:18]
        # extrinsic style code
        exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
            latent.shape)

    results = []
    # for i in range(4):  # change weights of structure codes
    #     for j in range(4):  # change weights of color codes
    i, j = 3, 4
    w = [i / 5.0] * 7 + [j / 5.0] * 11
    # w = [random.random()] * 7 + [random.random()]*11
    img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                           truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)  # Most time-consuming


    img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 128), -1, 1)

    # print(type(img_gen))
    print(img_gen.size())

    results += [img_gen]
    return results

if __name__ == "__main__":
    results = gen_color('../img/test/girl.png')
    for i in range(len(results)):
        results[i] = transfer(results[i])
    show_image(results)

    # vis = torchvision.utils.make_grid(torch.cat(results, dim=0), 4, 1)
    # plt.figure(figsize=(10, 10), dpi=120)
    # visualize(vis.cpu())
    # plt.show()

    index = int(input("You choose color: "))

    results = [results[index]]
    # vis = torchvision.utils.make_grid(torch.cat(results, dim=0), 1, 1)
    # plt.figure(figsize=(10, 10), dpi=40)
    # visualize(vis.cpu())
    # plt.show()

    show_image(transfer(results))
