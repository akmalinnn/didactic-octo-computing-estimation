import sys
sys.path.append('core')

import argparse
import os
import cv2
import glob
import numpy as np
import torch
from PIL import Image

from raft import RAFT
from utils import flow_viz
from utils.utils import InputPadder



DEVICE = 'cuda'

def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)


def viz(img, flo):
    img = img[0].permute(1,2,0).cpu().numpy()
    flo = flo[0].permute(1,2,0).cpu().numpy()
    
    # map flow to rgb image
    flo = flow_viz.flow_to_image(flo)
    img_flo = np.concatenate([img, flo], axis=0)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    cv2.waitKey()


def demo(args):
    argsModel = "models/raft-kitti.pth"
    model = torch.nn.DataParallel(RAFT(args))
    model.load_state_dict(torch.load(argsModel, weights_only=True))

    model = model.module
    model.to(DEVICE)
    model.eval()

    with torch.no_grad():
        images = glob.glob(os.path.join(args.path, '*.png')) + \
                 glob.glob(os.path.join(args.path, '*.jpg'))
        
        images = sorted(images)
        
        for imfile1, imfile2 in zip(images[:-1], images[1:]):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            flow_low, flow_up = model(image1, image2, iters=20, test_mode=True)
            
            # Extract base filename for the current image pair
            base_name1 = os.path.basename(imfile1)
            frame_identifier = base_name1.split('.')[0]
            
            # Construct the indices filename based on the first image's identifier
            indices_path = os.path.join(args.path, f'{frame_identifier}_indices.npy')
            
            if os.path.exists(indices_path):
                # Load the dictionary from .npy file with allow_pickle=True
                indices_dict = np.load(indices_path, allow_pickle=True).item()
            else:
                print(f"No indices file found for {frame_identifier} at {indices_path}.")
                continue  # Skip this frame if no indices are found

            # Create an empty tensor to hold the filtered flow
            flow_up_filtered = torch.zeros((2, image1.shape[2], image1.shape[3]), device=DEVICE)

            # Fill in the filtered flow using the loaded indices for each class
            #set to zero just filtere flow up
            for key, indices in indices_dict.items():
                flow_up_filtered[:, indices[:, 0], indices[:, 1]] = flow_up[0, :, indices[:, 0], indices[:, 1]] = 0



            viz(image1, flow_up)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default= "models/raft-kitti.pth",help="restore checkpoint")
    parser.add_argument('--path',  default=os.path.join("..", "segmentation-result"), help="dataset for evaluation")
    parser.add_argument('--small', action='store_true', help='use small model')
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    args = parser.parse_args()

    demo(args)
