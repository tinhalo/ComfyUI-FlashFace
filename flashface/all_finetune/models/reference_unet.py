import copy

import torch
import torch.nn as nn

from ldm.models.unet import (AttentionBlock, MultiHeadAttention, UNet,
                             sinusoidal_embedding)

# Constants for mode values
MODE_WRITE = 'w'  # Write mode - store reference features
MODE_PROCESS_REF = 'pr'  # Process reference mode - use stored references
MODE_NO_REF = 'nr'  # No reference mode

# Constants for dimensions
CONV_IN_CHANNELS = 5
CONV_OUT_CHANNELS = 320
CONV_KERNEL_SIZE = 3


class RefStableUNet(UNet):
    """Reference-based UNet model for stable diffusion.
    
    This extends the basic UNet to incorporate reference images for
    guided generation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.iter_index = 0
        # Use device from kwargs or default to detecting available device
        self.device = kwargs.get('device', torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        
    def replace_input_conv(self):
        """Replace input convolution to handle additional channels.
        
        Preserves weights for the first 4 channels and initializes the 5th.
        """
        ori_conv2d = self.encoder[0]
        device = next(self.parameters()).device  # Get device from existing parameters

        self.ori_encoder[0] = nn.Conv2d(CONV_IN_CHANNELS, CONV_OUT_CHANNELS, CONV_KERNEL_SIZE, padding=1).to(device)
        for _, p in self.ori_encoder[0].named_parameters():
            p.requires_grad = True

        # Initialize new conv weights
        self.ori_encoder[0].weight.data.fill_(0)
        self.ori_encoder[0].bias.data.fill_(0)
        # Copy old weights for the first 4 channels
        self.ori_encoder[0].weight.data[:, :4, :, :].copy_(
            ori_conv2d.weight.data)
        self.ori_encoder[0].bias.data.copy_(ori_conv2d.bias.data)

    def init_refnet(self, copy_model=False, enable_encoder=False):
        """Initialize reference network components.
        
        Sets up the reference encoders, decoders, and middle blocks.
        
        Args:
            copy_model: Whether to make deep copies of model components
            enable_encoder: Whether to enable encoder references
        """
        self.share_cache = dict(mode=MODE_WRITE, style_fidelity=1)
        for _, p in self.named_parameters():
            p.requires_grad = True

        # Create deep copies with memory management considerations
        try:
            self.ref_encoder = copy.deepcopy(self.encoder)
            self.ref_decoder = copy.deepcopy(self.decoder)
            self.ref_middle = copy.deepcopy(self.middle)
        except (RuntimeError, NotImplementedError) as e:
            # Fall back to shallow copy if memory issues occur
            print(f"Warning: Deep copy failed with {e}. Using references instead.")
            self.ref_encoder = self.encoder
            self.ref_decoder = self.decoder
            self.ref_middle = self.middle
            
        # Store original components
        self.ori_encoder = self.encoder
        self.ori_decoder = self.decoder
        self.ori_middle = self.middle

        # Set up share cache and keys for all modules
        self._setup_module_cache(self.ref_middle, 'middle')
        self._setup_module_cache(self.ori_middle, 'middle')
        self._setup_module_cache(self.ref_encoder, 'encoder')
        self._setup_module_cache(self.ref_decoder, 'decoder')
        self._setup_module_cache(self.ori_encoder, 'encoder')
        self._setup_module_cache(self.ori_decoder, 'decoder')

        all_ref_related_modules = [
            self.ref_decoder, self.ref_middle, self.ori_decoder,
            self.ori_middle
        ]

        # Apply custom forward function to attention blocks
        for coder in all_ref_related_modules:
            self._apply_attn_forward_patch(coder)

        all_ori_related_modules = [self.ori_decoder, self.ori_middle]

        # Set up self-attention copies for original modules
        for coder in all_ori_related_modules:
            self._setup_self_attn_first(coder)
            
    def _setup_module_cache(self, module, prefix):
        """Set up caching for modules.
        
        Args:
            module: The module to set up caching for
            prefix: Prefix for the cache keys
        """
        for n, m in module.named_modules():
            m.share_cache = self.share_cache
            m.key = f'{prefix}_{n}'

    def _apply_attn_forward_patch(self, module):
        """Apply attention forward patch to module.
        
        Args:
            module: The module to patch
        """
        for _, m in module.named_modules():
            if isinstance(m, AttentionBlock):
                self._patch_attention_forward(m)
                
    def _patch_attention_forward(self, attention_block):
        """Apply a patched forward method to an attention block.
        
        Args:
            attention_block: The attention block to patch
        """
        # Create a class to hold the original methods to avoid closure issues
        class OriginalMethods:
            self_attn = attention_block.self_attn
            cross_attn = attention_block.cross_attn
            ffn = attention_block.ffn
            norm1 = attention_block.norm1
            norm2 = attention_block.norm2
            norm3 = attention_block.norm3
        
        orig = OriginalMethods()
        
        # Define a patched forward method
        def patched_forward(self_mod, x, context=None, mask=None, **kwargs):
            return self._attention_forward_impl(
                self_mod, x, context, mask, 
                orig.self_attn, orig.cross_attn, orig.ffn, 
                orig.norm1, orig.norm2, orig.norm3, **kwargs
            )
        
        # Bind the method to the module
        attention_block.forward = patched_forward.__get__(attention_block)

    def _attention_forward_impl(self, module, x, context, mask,
                               orig_self_attn, orig_cross_attn, orig_ffn,
                               orig_norm1, orig_norm2, orig_norm3, **kwargs):
        """Implementation of attention forward logic.
        
        Args:
            module: The attention module
            x: Input tensor
            context: Context tensor
            mask: Attention mask
            orig_self_attn: Original self attention module
            orig_cross_attn: Original cross attention module
            orig_ffn: Original feed-forward network
            orig_norm1: Original norm1 layer
            orig_norm2: Original norm2 layer
            orig_norm3: Original norm3 layer
            
        Returns:
            Processed tensor
        """
        # Handle disabled self-attention case
        if module.disable_self_attn:
            norm1_out = orig_norm1(x)
            attn_output = orig_self_attn(norm1_out, context)
            x = x + attn_output
            return self._apply_cross_attention_and_ffn(
                x, context, mask, orig_cross_attn, orig_ffn, orig_norm2, orig_norm3
            )
        
        # Process normal self-attention
        norm1_out = orig_norm1(x)
        cache_key = module.key
        current_mode = module.share_cache['mode']
        
        # Handle different modes
        if current_mode == MODE_PROCESS_REF:
            x = self._process_reference_mode(
                module, x, norm1_out, cache_key, orig_self_attn
            )
        elif current_mode == MODE_WRITE:
            # Store reference in cache and apply self-attention
            module.share_cache[cache_key] = norm1_out
            attn_output = orig_self_attn(norm1_out)
            x = x + attn_output
        
        # Apply cross-attention and FFN
        return self._apply_cross_attention_and_ffn(
            x, context, mask, orig_cross_attn, orig_ffn, orig_norm2, orig_norm3
        )
    
    def _process_reference_mode(self, module, x, norm1_out, cache_key, orig_self_attn):
        """Process reference mode for attention.
        
        Args:
            module: The attention module
            x: Input tensor
            norm1_out: Output from norm1
            cache_key: Cache key for retrieval
            orig_self_attn: Original self attention module
            
        Returns:
            Processed tensor
        """
        # Apply self attention first
        attn_output = orig_self_attn(norm1_out)
        x = x + attn_output
        
        # Check if we have diffusion conditions
        if 'num_diff_condition' not in module.share_cache:
            return x
            
        num_diff_condition = module.share_cache['num_diff_condition']
        b, _, c = norm1_out.shape  # Use _ instead of l for unused variable
        
        # Get reference from cache
        ref = module.share_cache.get(cache_key)
        if ref is None:
            return x
            
        # Process reference
        ref_expanded = ref.unsqueeze(0).repeat_interleave(b, dim=0)
        ref_reshaped = ref_expanded.reshape(b, -1, c)
        ctx = torch.cat([norm1_out, ref_reshaped], dim=1)
        
        # Get similarity value, ensuring it's a float
        similarity_value = float(module.share_cache.get('similarity', 1.0))
        
        # Process based on diffusion condition count
        if num_diff_condition == 2:
            return self._process_two_conditions(
                module, x, norm1_out, ctx, similarity_value
            )
        elif num_diff_condition == 3:
            return self._process_three_conditions(
                module, x, norm1_out, ctx, b, similarity_value
            )
        
        return x
    
    def _process_two_conditions(self, module, x, norm1_out, ctx, similarity_value):
        """Process for two diffusion conditions.
        
        Args:
            module: The attention module
            x: Input tensor
            norm1_out: Output from norm1
            ctx: Context tensor
            similarity_value: Similarity value
            
        Returns:
            Processed tensor
        """
        ref_attn = module.self_attn.self_attn_first(norm1_out, ctx)
        self_attn = module.self_attn.self_attn_first(norm1_out)
        return x + ref_attn * similarity_value + (1.0 - similarity_value) * self_attn
    
    def _process_three_conditions(self, module, x, norm1_out, ctx, batch_size, similarity_value):
        """Process for three diffusion conditions.
        
        Args:
            module: The attention module
            x: Input tensor
            norm1_out: Output from norm1
            ctx: Context tensor
            batch_size: Batch size
            similarity_value: Similarity value
            
        Returns:
            Processed tensor
        """
        # Create tensor on same device as inputs
        similarity_tensor = torch.tensor(
            [similarity_value, 0.0, 0.0], device=x.device
        )
        num_samples = batch_size // 3
        sim_expanded = similarity_tensor.unsqueeze(1).repeat_interleave(
            num_samples, dim=1
        ).flatten()
        sim_final = sim_expanded.unsqueeze(1).unsqueeze(2)
        
        ref_attn = module.self_attn.self_attn_first(norm1_out, ctx)
        self_attn = module.self_attn.self_attn_first(norm1_out)
        return x + ref_attn * sim_final + (1.0 - sim_final) * self_attn
    
    def _apply_cross_attention_and_ffn(self, x, context, mask, 
                                      cross_attn, ffn, norm2, norm3):
        """Apply cross-attention and feed-forward network.
        
        Args:
            x: Input tensor
            context: Context tensor
            mask: Attention mask
            cross_attn: Cross attention module
            ffn: Feed-forward network
            norm2: Norm2 layer
            norm3: Norm3 layer
            
        Returns:
            Processed tensor
        """
        norm2_out = norm2(x)
        x = x + cross_attn(norm2_out, context, mask)
        
        norm3_out = norm3(x)
        x = x + ffn(norm3_out)
        return x

    def _setup_self_attn_first(self, module):
        """Set up self-attention first for module.
        
        Args:
            module: The module to set up
        """
        for n, m in module.named_modules():
            if isinstance(m, MultiHeadAttention) and n.endswith('self_attn'):
                m.self_attn_first = copy.deepcopy(m)
                for p in m.self_attn_first.parameters():
                    p.requires_grad = True

    def switch_mode(self, mode=MODE_WRITE):
        """Switch between different operating modes.
        
        Args:
            mode: One of MODE_WRITE, MODE_PROCESS_REF, or MODE_NO_REF
        
        Raises:
            ValueError: If mode is not recognized
        """
        if mode == MODE_WRITE:
            self.share_cache['mode'] = MODE_WRITE
            self.encoder = self.ref_encoder
            self.decoder = self.ref_decoder
            self.middle = self.ref_middle
        elif mode == MODE_PROCESS_REF:
            self.share_cache['mode'] = MODE_PROCESS_REF
            self.encoder = self.ori_encoder
            self.decoder = self.ori_decoder
            self.middle = self.ori_middle
        elif mode == MODE_NO_REF:
            self.share_cache['mode'] = MODE_NO_REF
            self.encoder = self.ori_encoder
            self.decoder = self.ori_decoder
            self.middle = self.ori_middle
        else:
            raise ValueError(f'Unknown mode {mode}. Expected one of: {MODE_WRITE}, {MODE_PROCESS_REF}, {MODE_NO_REF}')

    def forward(self,
                x,
                t,
                y=None,
                context=None,
                mask=None,
                caching=None,
                style_fidelity=0.5):
        """Forward pass with reference handling.
        
        Args:
            x: Input tensor
            t: Time step
            y: Class embedding (optional)
            context: Context for cross-attention
            mask: Attention mask
            caching: Attention cache
            style_fidelity: Style fidelity factor
            
        Returns:
            Output tensor
        """
        # Validate inputs
        if context is None:
            raise ValueError("Context cannot be None")
            
        num_sample = t.shape[0]
        num_diff_condition = context.shape[0] // num_sample
        
        # Validate diffusion conditions
        if not (num_diff_condition == 2 or num_diff_condition == 3):
            raise ValueError(f"Number of diffusion conditions must be 2 or 3, got {num_diff_condition}")
            
        t = t.repeat_interleave(num_diff_condition, dim=0)
        # embeddings
        self.share_cache['num_diff_condition'] = num_diff_condition
        num_refs = self.share_cache.get('num_pairs', 1)

        similarity = self.share_cache.get('similarity', 1.0)

        e = self.time_embedding(sinusoidal_embedding(t, self.dim))

        self.switch_mode(MODE_WRITE)
        ref_x = self.share_cache['ref']
        ref_context = self.share_cache['ref_context']

        ref_e = e[:1].repeat_interleave(num_refs, dim=0)

        # Store similarity as string to avoid type issues with the dictionary
        self.share_cache['similarity'] = str(float(similarity))

        # encoder-decoder for reference
        args = (ref_e, ref_context, mask, caching, style_fidelity)
        ref_x, *xs = self.encode(ref_x, *args)
        self.decode(ref_x, *args, *xs)

        # Process with reference
        self.switch_mode(MODE_PROCESS_REF)
        args = (e, context, mask, caching, style_fidelity)
        
        # Get masks and handle potential non-tensor values
        masks = self.share_cache.get('masks')
        if not isinstance(masks, torch.Tensor):
            raise ValueError("Expected 'masks' in share_cache to be a tensor")
            
        masks = masks.repeat_interleave(num_diff_condition, dim=0)
        # Use device of x instead of hardcoded cuda
        cuda_mask = masks.to(x.device)[:, None]
        x = torch.cat([x, cuda_mask], dim=1)
        x, *xs = self.encode(x, *args)
        x = self.decode(x, *args, *xs)

        return x


def sd_v1_ref_unet(pretrained=False,
                   version='sd-v1-5_ema',
                   device='cpu',
                   enable_encoder=False,
                   **kwargs):
    """UNet of Stable Diffusion 1.x (1.1~1.5)."""
    # sanity check
    assert version in ('sd-v1-1_ema', 'sd-v1-1_nonema', 'sd-v1-2_ema',
                       'sd-v1-2_nonema', 'sd-v1-3_ema', 'sd-v1-3_nonema',
                       'sd-v1-4_ema', 'sd-v1-4_nonema', 'sd-v1-5_ema',
                       'sd-v1-5_nonema', 'sd-v1-5-inpainting_nonema')

    # dedue dimension
    in_dim = 4
    if 'inpainting' in version:
        in_dim = 9

    # init model
    cfg = dict(in_dim=in_dim,
               dim=320,
               y_dim=None,
               context_dim=768,
               out_dim=4,
               dim_mult=[1, 2, 4, 4],
               num_heads=8,
               head_dim=None,
               num_res_blocks=2,
               num_attn_blocks=1,
               attn_scales=[1 / 4, 1 / 2, 1],
               dropout=0.0)
    cfg.update(**kwargs)
    model = RefStableUNet(**cfg).to(device)
    model.init_refnet(copy_model=True, enable_encoder=enable_encoder)

    return model
