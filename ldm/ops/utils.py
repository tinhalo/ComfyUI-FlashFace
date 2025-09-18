import importlib.util
import math
from typing import Optional, Union, Tuple

import torch
import torch.cuda.amp as amp
import torch.nn.functional as F


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
    flash_dtyp: torch.dtype = torch.float16
) -> torch.Tensor:
    """
    Compute scaled dot product attention using the most efficient implementation available.
    
    If PyTorch 2.0+ is available, uses the native scaled_dot_product_attention.
    If flash-attention is available and conditions are met, uses flash attention.
    Otherwise, falls back to manual implementation.
    
    Args:
        q: Query tensor [B, L1, N, C1]
        k: Key tensor [B, L2, N, C1]
        v: Value tensor [B, L2, N, C2]
        attn_mask: Optional attention mask [B, ..., L1, L2]
        dropout_p: Dropout probability
        is_causal: Whether to use causal attention masking
        flash_dtyp: Data type to use for flash attention
        
    Returns:
        Output tensor [B, L1, N, C2]
    """
    # Check if we can use PyTorch 2.0+ native implementation
    pytorch_version = torch.__version__.split('.')
    use_native = int(pytorch_version[0]) >= 2
    
    # Process parameters
    b, l1, l2 = len(q), q.size(1), k.size(1)
    
    # Initialize output variable
    output = None
    
    # Option 1: Use flash attention if available and conditions are met
    if importlib.util.find_spec('flash_attn') is not None and \
            q.device.type == 'cuda' and q.size(-1) <= 256 and attn_mask is None:
        from flash_attn import flash_attn_func

        def half(x):
            return x.to(flash_dtyp) if x.dtype not in (torch.float16, torch.bfloat16) else x

        # flash attention
        with amp.autocast():
            output = flash_attn_func(
                q=half(q),
                k=half(k),
                v=half(v),
                dropout_p=dropout_p,
                causal=is_causal
            )

        # convert the data type back
        if output is not None and output.dtype != q.dtype:
            output = output.to(q.dtype)
    
    # Option 2: Use PyTorch 2.0+ native implementation
    elif use_native:
        # process mask
        if attn_mask is not None and is_causal:
            attn_mask = attn_mask.view(b, -1, l1, l2).tril()
            is_causal = False

        # Use PyTorch's native implementation - more efficient in PyTorch 2.8.0
        output = F.scaled_dot_product_attention(
            query=q.transpose(1, 2).contiguous(),
            key=k.transpose(1, 2).contiguous(),
            value=v.transpose(1, 2).contiguous(),
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal
        ).transpose(1, 2).contiguous()
    
    # Option 3: Manual implementation for older PyTorch versions
    else:
        attn = torch.einsum('binc,bjnc->bnij', q, k) / math.sqrt(q.size(-1))

        # apply mask
        if attn_mask is not None:
            attn_mask = attn_mask.view(b, -1, l1, l2)
            if attn_mask.dtype == torch.bool:
                attn = attn.masked_fill(attn_mask == 0, float('-inf'))
            else:
                attn = attn + attn_mask

        # causal mask
        if is_causal:
            attn = attn.masked_fill(
                torch.tril(attn.new_ones(1, 1, l1, l2).float()).type_as(attn) == 0,
                float('-inf')
            )

        # gather context
        attn = F.softmax(attn.float(), dim=-1).type_as(attn)
        if dropout_p > 0:
            attn = F.dropout(attn, p=dropout_p)
            
        output = torch.einsum('bnij,bjnc->binc', attn, v)
    
    # Ensure output is contiguous
    if output is not None:
        return output.contiguous()
    
    # Fallback in case no method worked
    raise RuntimeError("No valid attention implementation was executed")
