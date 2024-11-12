from torch.utils.data import dataset
from tqdm import tqdm
import network
import utils
import os
import random
import argparse
import numpy as np

from torch.utils import data
from datasets import VOCSegmentation, Cityscapes, cityscapes
from torchvision import transforms as T
from metrics import StreamSegMetrics

import torch
import torch.nn as nn

from PIL import Image
import matplotlib
import matplotlib.pyplot as plt
from glob import glob

from scipy.ndimage import binary_dilation


def get_argparser():
    parser = argparse.ArgumentParser()

    # Datset Options
    parser.add_argument("--input", type=str,  default=os.path.join("..", "segmentation-result"),
                        help="path to a single image or image directory")
    parser.add_argument("--dataset", type=str, default='cityscapes',
                        choices=['voc', 'cityscapes'], help='Name of training set')

    # Deeplab Options
    available_models = sorted(name for name in network.modeling.__dict__ if name.islower() and \
                              not (name.startswith("__") or name.startswith('_')) and callable(
                              network.modeling.__dict__[name])
                              )

    parser.add_argument("--model", type=str, default='deeplabv3plus_mobilenet',
                        choices=available_models, help='model name')
    parser.add_argument("--separable_conv", action='store_true', default=False,
                        help="apply separable conv to decoder and aspp")
    parser.add_argument("--output_stride", type=int, default=16, choices=[8, 16])

    # Train Options
    parser.add_argument("--save_val_results_to", default=os.path.join("..", "segmentation-result"),
                        help="save segmentation results to the specified dir")

    parser.add_argument("--crop_val", action='store_true', default=False,
                        help='crop validation (default: False)')
    parser.add_argument("--val_batch_size", type=int, default=4,
                        help='batch size for validation (default: 4)')
    parser.add_argument("--crop_size", type=int, default=513)

    
    parser.add_argument("--ckpt", default="checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth", type=str,
                        help="resume from checkpoint")
    parser.add_argument("--gpu_id", type=str, default='0',
                        help="GPU ID")
    return parser

def get_prediction_indices(model, img, device, dilation_size=5):
    # Move model to evaluation mode and disable gradient computation
    model.eval()
    with torch.no_grad():
        # Move image to the specified device
        img = img.to(device)
        
        # Get prediction logits and convert to predicted class IDs
        pred = model(img).max(1)[1].cpu().numpy()[0]  # HW format

        # Apply dilation to the predicted regions to increase the class boundaries
        dilated_indices_dict = {}
        for class_id in [13, 14, 15]:  # Define target classes here
            # Create a binary mask for the target class
            mask = (pred == class_id).astype(int)
            
            # Apply dilation to the mask to increase the boundary size
            dilated_mask = binary_dilation(mask, structure=np.ones((dilation_size, dilation_size)))  # Dilation
            
            # Get the indices of the dilated class region
            dilated_indices_dict[f"indices_{class_id}"] = np.argwhere(dilated_mask)

        return dilated_indices_dict

def main():
    opts = get_argparser().parse_args()
    if opts.dataset.lower() == 'voc':
        opts.num_classes = 21
        decode_fn = VOCSegmentation.decode_target
    elif opts.dataset.lower() == 'cityscapes':
        opts.num_classes = 19
        decode_fn = Cityscapes.decode_target

    os.environ['CUDA_VISIBLE_DEVICES'] = opts.gpu_id
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    # Load image files
    image_files = []
    if os.path.isdir(opts.input):
        for ext in ['png', 'jpeg', 'jpg', 'JPEG']:
            files = glob(os.path.join(opts.input, '**/*.%s' % (ext)), recursive=True)
            if len(files) > 0:
                image_files.extend(files)
    elif os.path.isfile(opts.input):
        image_files.append(opts.input)

    # Set up model
    model = network.modeling.__dict__[opts.model](num_classes=opts.num_classes, output_stride=opts.output_stride)
    if opts.separable_conv and 'plus' in opts.model:
        network.convert_to_separable_conv(model.classifier)
    utils.set_bn_momentum(model.backbone, momentum=0.01)

    if opts.ckpt is not None and os.path.isfile(opts.ckpt):
        checkpoint = torch.load(opts.ckpt, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["model_state"])
        model = nn.DataParallel(model)
        model.to(device)
        print("Resume model from %s" % opts.ckpt)
        del checkpoint
    else:
        print("[!] Retrain")
        model = nn.DataParallel(model)
        model.to(device)

    # Transform setup
    if opts.crop_val:
        transform = T.Compose([
            T.Resize(opts.crop_size),
            T.CenterCrop(opts.crop_size),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])
    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
        ])

    if opts.save_val_results_to:
        os.makedirs(opts.save_val_results_to, exist_ok=True)

    with torch.no_grad():
        for img_path in tqdm(image_files):
            ext = os.path.basename(img_path).split('.')[-1]
            img_name = os.path.basename(img_path)[:-len(ext) - 1]
            img = Image.open(img_path).convert('RGB')
            img = transform(img).unsqueeze(0)  # To tensor of NCHW format

            # # Get prediction indices using the helper function
            # indices_dict = get_prediction_indices(model, img, device)
            # print(indices_dict)
        
            dilated_indices_dict = get_prediction_indices(model, img, device, dilation_size=5)
            print(dilated_indices_dict)

            # Save prediction indices
            if opts.save_val_results_to:
                np.save(os.path.join(opts.save_val_results_to, img_name + '_indices.npy'), dilated_indices_dict)

            # # Colorize predictions and save
            # pred = model(img).max(1)[1].cpu().numpy()[0]  # HW format
            # colorized_preds = decode_fn(pred).astype('uint8')
            # colorized_preds = Image.fromarray(colorized_preds)
            # if opts.save_val_results_to:
            #     colorized_preds.save(os.path.join(opts.save_val_results_to, img_name + '.png'))

if __name__ == '__main__':
    main()
