import argparse
import coremltools as ct
from timm import create_model
import torch

import warnings
warnings.filterwarnings('ignore')

from models import mobilevig
from util import *


def parse():
    parser = argparse.ArgumentParser(description='Convert from PyTorch to CoreML')
    parser.add_argument('--model', metavar='ARCH', default='mobilevig_ti')
    parser.add_argument('--ckpt', default='', type=str, metavar='PATH',
                    help='Initialize model from this checkpoint (default: none)')
    parser.add_argument("--resolution", default=224, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse()
    model = create_model(args.model)
    try:
        model.load_state_dict(torch.load(args.ckpt, map_location='cpu')['model'])
        print('load success, model is initialized with pretrained checkpoint')
    except:
        print('model initialized without pretrained checkpoint')

    model.eval()
    dummy_input = torch.randn(1, 3, args.resolution, args.resolution)

    with torch.no_grad():
        profile = Profiler(model)
        MACs, params = profile(dummy_input)
        print(sum(MACs) / 1e9, 'GMACs')
        print(sum(params) / 1e6, 'M parameters')

    traced_model = torch.jit.trace(model, dummy_input)
    model = ct.convert(
        traced_model,
        inputs=[ct.ImageType(
            name="x_1", 
            shape=dummy_input.shape,
            scale=1.0/(255.0*0.226),
            bias=[-0.485/0.226, -0.456 / 0.226, -0.406 / 0.226])]
    )

    model.save(args.model + ".mlmodel")
    print('exported coreml model')
