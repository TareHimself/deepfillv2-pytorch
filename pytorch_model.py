from typing import Optional
import torch
import torch.nn as nn

class DeepFillV2InPaint(nn.Module):
    def __init__(self,checkpoint_path: Optional[str] = None):
        super().__init__()
        from model.networks import Generator
        self.generator = Generator(cnum_in=5, cnum=48, return_flow=False)
        if checkpoint_path is not None:
            generator_state_dict = torch.load(checkpoint_path,map_location=torch.device("cpu"))['G']
            self.generator.load_state_dict(generator_state_dict, strict=True)

    def forward(self,images: torch.Tensor,masks: torch.Tensor):
        """
        Args:
            images (Tensor): rgba input of shape [batch, channels, H, W] in range [0,1]
            masks (Tensor): mask of shape [batch, 1, H, W] in range [0,1]
        """
        batch_count,_,h,w = images.shape
        images = images * 2 - 1.
        masks = (masks > 0.5).to(dtype=torch.float32)
        images_masked = images * (1.-masks)
        ones_x = torch.ones((batch_count,1,h,w),dtype=images.dtype,device=images.device)
        x = torch.cat([images_masked,ones_x,ones_x*masks],dim=1)
        _, x_stage2 = self.generator(x, masks)
        inpainted = images * (1.-masks) + x_stage2 * masks
        return (inpainted + 1) * 0.5# scale back to [0,1]
