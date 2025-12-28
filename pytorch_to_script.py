import argparse
import torch
from pytorch_model import DeepFillV2InPaint

parser = argparse.ArgumentParser(description='Convert a checkpoint to a torchscript model')
parser.add_argument("--out", type=str,
                    default="torchscript/deepfillv2_places2_script.pt", help="path for the output file")
parser.add_argument("--checkpoint", type=str,
                    default="pretrained/states_pt_places2.pth", help="path to the checkpoint file")


def main():

    args = parser.parse_args()
    model = DeepFillV2InPaint(checkpoint_path=args.checkpoint)
    model.eval()

    scripted = torch.jit.trace(model,(torch.ones((1,3,400,400),dtype=torch.float32),torch.ones((1,1,400,400),dtype=torch.float32)))

    torch.jit.save(scripted,args.out)
    print(f"Saved output file at: {args.out}")


if __name__ == '__main__':
    main()
