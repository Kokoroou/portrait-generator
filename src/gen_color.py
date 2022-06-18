import os
import random

import numpy as np
import torch
from util import save_image, load_image, visualize
import argparse
from argparse import Namespace
from torchvision import transforms
from torch.nn import functional as F
import torchvision
from model.dualstylegan import DualStyleGAN
from model.encoder.psp import pSp
from PIL import Image
import matplotlib.pyplot as plt
import tkinter
import matplotlib
matplotlib.use('TkAgg')

class TestOptions():
    def __init__(self, src):
        self.parser = argparse.ArgumentParser(description="Exemplar-Based Style Transfer")
        self.parser.add_argument("--content", type=str, default=src,
                                 help="path of the content image")
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
        self.parser.add_argument("--model_path", type=str, default='./checkpoint/', help="path of the saved models")
        self.parser.add_argument("--model_name", type=str, default='generator.pt',
                                 help="name of the saved dualstylegan")
        self.parser.add_argument("--output_path", type=str, default='./output/', help="path of the output images")
        self.parser.add_argument("--data_path", type=str, default='./data/', help="path of dataset")
        self.parser.add_argument("--align_face", action="store_true", help="apply face alignment to the content image")
        self.parser.add_argument("--exstyle_name", type=str, default=None, help="name of the extrinsic style codes")

    def parse(self):
        self.opt = self.parser.parse_args()
        if self.opt.exstyle_name is None:
            if os.path.exists(os.path.join(self.opt.model_path, self.opt.style, 'refined_exstyle_code.npy')):
                self.opt.exstyle_name = 'refined_exstyle_code.npy'
            else:
                self.opt.exstyle_name = 'exstyle_code.npy'
        args = vars(self.opt)
        #print('Load options')
        #for name, value in sorted(args.items()):
        #    print('%s: %s' % (str(name), str(value)))
        return self.opt


def run_alignment(args):
    import dlib
    from model.encoder.align_all_parallel import align_face
    model_name = os.path.join(args.model_path, 'shape_predictor_68_face_landmarks.dat')
    if not os.path.exists(model_name):
        import wget, bz2
        wget.download('http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2', model_name + '.bz2')
        zipfile = bz2.BZ2File(model_name + '.bz2')
        data = zipfile.read()
        open(model_name, 'wb').write(data)
    predictor = dlib.shape_predictor(model_name)
    aligned_image = align_face(filepath=args.content, predictor=predictor)
    return aligned_image


def gen_color(src):
    device = "cpu"

    parser = TestOptions(src)
    args = parser.parse()
    #print('*' * 98)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    generator = DualStyleGAN(1024, 512, 8, 2, res_index=6)
    generator.eval()

    ckpt = torch.load(os.path.join(args.model_path, args.style, args.model_name),
                      map_location=lambda storage, loc: storage)
    generator.load_state_dict(ckpt["g_ema"])
    generator = generator.to(device)

    model_path = os.path.join(args.model_path, 'encoder.pt')
    ckpt = torch.load(model_path, map_location='cpu')
    opts = ckpt['opts']
    opts['checkpoint_path'] = model_path
    opts = Namespace(**opts)
    opts.device = device
    encoder = pSp(opts)
    encoder.eval()
    encoder.to(device)

    exstyles = np.load(os.path.join(args.model_path, args.style, args.exstyle_name), allow_pickle='TRUE').item()

    #print('Load models successfully!')

    with torch.no_grad():
        # load content image
        if args.align_face:
            I = transform(run_alignment(args)).unsqueeze(dim=0).to(device)
            I = F.adaptive_avg_pool2d(I, 1024)
        else:
            I = load_image(args.content).to(device)

        # reconstructed content image and its intrinsic style code
        img_rec, instyle = encoder(F.adaptive_avg_pool2d(I, 256), randomize_noise=False, return_latents=True,
                                   z_plus_latent=True, return_z_plus_latent=True, resize=False)
        img_rec = torch.clamp(img_rec.detach(), -1, 1)

        stylename = list(exstyles.keys())[args.style_id]
        latent = torch.tensor(exstyles[stylename]).to(device)
        if args.preserve_color:
            latent[:, 7:18] = instyle[:, 7:18]
        # extrinsic styte code
        exstyle = generator.generator.style(latent.reshape(latent.shape[0] * latent.shape[1], latent.shape[2])).reshape(
            latent.shape)

    results = []
    for i in range(4):  # change weights of structure codes
        for j in range(4):  # change weights of color codes
            w = [i / 5.0] * 7 + [j / 5.0] * 11
            #w = [random.random()] * 7 + [random.random()]*11
            img_gen, _ = generator([instyle], exstyle[0:1], z_plus_latent=True,
                                   truncation=0.7, truncation_latent=0, use_res=True, interp_weights=w)
            img_gen = torch.clamp(F.adaptive_avg_pool2d(img_gen.detach(), 128), -1, 1)
            results += [img_gen]

    return results

if __name__ == "__main__":
    results = gen_color('../img/test/frontal face.png')
    vis = torchvision.utils.make_grid(torch.cat(results, dim=0), 4, 1)
    plt.figure(figsize=(10, 10), dpi=120)
    visualize(vis.cpu())
    plt.show()

    index = int(input("You choose color: "))

    results=[results[index]]
    vis = torchvision.utils.make_grid(torch.cat(results, dim=0), 1, 1)
    plt.figure(figsize=(10, 10), dpi=40)
    visualize(vis.cpu())
    plt.show()
