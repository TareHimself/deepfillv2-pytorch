import argparse
from PIL import Image
import torch
import torchvision.transforms as T
from pytorch_model import DeepFillV2InPaint

parser = argparse.ArgumentParser(description='Test inpainting')
parser.add_argument("--image", type=str,
                    default="examples/inpaint/case1.png", help="path to the image file")
parser.add_argument("--mask", type=str,
                    default="examples/inpaint/case1_mask.png", help="path to the mask file")
parser.add_argument("--out", type=str,
                    default="examples/inpaint/case1_out_test_torchscript.png", help="path for the output file")
parser.add_argument("--model", type=str,
                    default="torchscript/deepfillv2_places2_script.pt", help="the torchscript model")


def main():

    args = parser.parse_args()

    use_cuda_if_available = True
    device = torch.device('cuda' if torch.cuda.is_available()
                          and use_cuda_if_available else 'cpu')
    
    model = torch.jit.load(args.model,map_location=device)

    # load image and mask
    image = Image.open(args.image)
    mask = Image.open(args.mask)

    # prepare input
    image = T.ToTensor()(image)
    mask = T.ToTensor()(mask)

    _, h, w = image.shape
    grid = 8

    image = image[:3, :h//grid*grid, :w//grid*grid].unsqueeze(0)
    mask = mask[0:1, :h//grid*grid, :w//grid*grid].unsqueeze(0)

    print(f"Shape of image: {image.shape}")

    with torch.inference_mode():
        result = model(image,mask)

    image_inpainted = result * 255

    # save inpainted image
    img_out = (image_inpainted[0].permute(1, 2, 0))
    img_out = img_out.to(device='cpu', dtype=torch.uint8)
    img_out = Image.fromarray(img_out.numpy())
    img_out.save(args.out)

    print(f"Saved output file at: {args.out}")


if __name__ == '__main__':
    main()
