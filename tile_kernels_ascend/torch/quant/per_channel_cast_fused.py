from typing import Optional, Union
import torch
from tile_kernels_ascend.torch.quant.types import QuantTensor
from tile_kernels_ascend.torch.quant.cast import cast


def per_channel_cast_fused(
    x: Union[torch.Tensor, QuantTensor],
    num_per_tokens: int,
    num_per_channels: Optional[int],
    round_sf: bool,
    pos_to_token: Optional[torch.Tensor],
) -> QuantTensor:
    is_fused_cast_back = isinstance(x, tuple)
    if pos_to_token is not None:
        x_data = x[0] if is_fused_cast_back else x
        x_gathered = x_data[pos_to_token.clamp(min=0)]
        valid_mask = (pos_to_token >= 0).unsqueeze(1)
        x_gathered = torch.where(
            valid_mask, x_gathered.to(torch.float32),
            torch.zeros_like(x_gathered, dtype=torch.float32),
        ).to(x_data.dtype)
        if is_fused_cast_back:
            x_sf = x[1]
            x_sf_gathered = x_sf[pos_to_token.clamp(min=0)]
            x_sf_gathered = torch.where(valid_mask, x_sf_gathered, torch.zeros_like(x_sf_gathered))
            x = (x_gathered, x_sf_gathered)
        else:
            x = x_gathered
    x_block_size = (1, num_per_channels) if is_fused_cast_back else None
    return cast(x, 'e4m3', block_size=(num_per_tokens, 1), x_block_size=x_block_size, round_sf=round_sf)
