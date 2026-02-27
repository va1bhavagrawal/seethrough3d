import os
import json
import torch
from PIL import Image
import numpy as np
from typing import Optional, List, Tuple
from transformers import CLIPTokenizer, T5TokenizerFast

import sys 
sys.path.append(os.path.join(os.getcwd(), "..")) 
from train.src.pipeline import FluxPipeline
from train.src.transformer_flux import FluxTransformer2DModel
from train.src.lora_helper import set_single_lora, set_multi_lora, unset_lora
from train.src.jsonl_datasets import make_train_dataset, collate_fn
import config


class InferenceArgs:
    """Arguments configuration for inference dataset loading"""
    def __init__(self, jsonl_path: str, pretrained_model_name: str):
        # Basic paths
        self.current_train_data_dir = jsonl_path
        self.inference_embeds_dir = "" # dummy value 
        self.pretrained_model_name_or_path = pretrained_model_name
        
        # Column configurations
        self.subject_column = None  # Set to None since we're using spatial
        self.spatial_column = "cv"
        self.target_column = "target"
        self.caption_column = "PLACEHOLDER_prompts"
        
        # Size configurations
        self.cond_size = 512
        self.noise_size = 512
        
        # Other required parameters
        self.revision = None
        self.variant = None
        self.max_sequence_length = 512


class InferenceEngine:
    """
    Handles model loading and inference for the Gradio interface.
    Pre-loads the base model and dynamically loads LoRA weights based on checkpoint selection.
    """
    
    def __init__(self, base_model_path: str = config.PRETRAINED_MODEL_NAME_OR_PATH, device: str = "cuda"):
        """
        Initialize the inference engine with base model.
        
        Args:
            base_model_path: Path to the base FLUX model
            device: Device to run inference on (default: "cuda")
        """
        self.device = device
        self.base_model_path = base_model_path
        self.current_lora_path = None
        
        print(f"Loading base model from {base_model_path}...")
        
        # Load pipeline and transformer
        self.pipe = FluxPipeline.from_pretrained(
            base_model_path, 
            torch_dtype=torch.bfloat16, 
            device=device
        )
        
        transformer = FluxTransformer2DModel.from_pretrained(
            base_model_path, 
            subfolder="transformer",
            torch_dtype=torch.bfloat16, 
            device=device
        )
        
        self.pipe.transformer = transformer
        self.pipe.to(device)
        
        # Load tokenizers (same as in train.py and infer.ipynb)
        print("Loading tokenizers...")
        self.tokenizer_one = CLIPTokenizer.from_pretrained(
            base_model_path,
            subfolder="tokenizer",
            revision=None,
        )
        self.tokenizer_two = T5TokenizerFast.from_pretrained(
            base_model_path,
            subfolder="tokenizer_2", 
            revision=None,
        )
        self.tokenizers = [self.tokenizer_one, self.tokenizer_two]
        
        print("Base model and tokenizers loaded successfully!")
    
    def load_lora(self, checkpoint_name: str, lora_weights: List[float] = [1.0]):
        """
        Load LoRA weights for a specific checkpoint.
        
        Args:
            checkpoint_name: Name of the checkpoint (e.g., "checkpoint_1")
            lora_weights: Weights for the LoRA adaptation
        """
        # Construct LoRA path
        lora_path = os.path.join(config.LORA_WEIGHTS_ROOT, checkpoint_name, "lora.safetensors")

        print(f"\n\nGOT THE FOLLOWING LORA PATH: {lora_path}\n\n") 
        
        # Check if path exists
        if not os.path.exists(lora_path):
            raise FileNotFoundError(f"LoRA checkpoint not found at: {lora_path}")
        
        # Only reload if it's a different checkpoint
        if self.current_lora_path != lora_path:
            print(f"Loading LoRA weights from {lora_path}...")
            set_single_lora(
                self.pipe.transformer, 
                lora_path, 
                lora_weights=lora_weights,
                cond_size=512
            )
            self.current_lora_path = lora_path
            print(f"LoRA weights loaded successfully!")
        else:
            print(f"LoRA already loaded for {checkpoint_name}")
    
    def clear_cache(self):
        """Clear attention processor cache"""
        for name, attn_processor in self.pipe.transformer.attn_processors.items():
            if hasattr(attn_processor, 'bank_kv'):
                attn_processor.bank_kv.clear()
    
    def tensor_to_image_list(self, tensor):
        """Convert normalized tensor to PIL Image list"""
        if tensor is None:
            return []
        
        images = []
        for img_tensor in tensor:
            # Denormalize from [-1, 1] to [0, 1]
            img = (img_tensor.cpu().permute(1, 2, 0) * 0.5 + 0.5).clamp(0, 1).numpy()
            # Convert to [0, 255] uint8
            img = (img * 255.0).astype(np.uint8)
            images.append(Image.fromarray(img))
        
        return images
    
    def run_inference(
        self,
        jsonl_path: str,
        checkpoint_name: str,
        height: int = 512,
        width: int = 512,
        seed: int = 42,
        guidance_scale: float = 3.5,
        num_inference_steps: int = 25,
        max_sequence_length: int = 512
    ) -> Tuple[bool, Optional[Image.Image], str]:
        """
        Run inference using data from JSONL file.
        Uses the same data loading pipeline as training (make_train_dataset).
        
        Args:
            jsonl_path: Path to the JSONL file containing inference data
            checkpoint_name: Name of checkpoint to use
            height: Output image height
            width: Output image width
            seed: Random seed for generation
            guidance_scale: Guidance scale for diffusion
            num_inference_steps: Number of denoising steps
            max_sequence_length: Maximum sequence length for text encoding
            
        Returns:
            Tuple of (success: bool, image: PIL.Image or None, message: str)
        """
        try:
            # Load LoRA for selected checkpoint
            self.load_lora(checkpoint_name)
            
            # Check if JSONL file exists
            if not os.path.exists(jsonl_path):
                return False, None, f"JSONL file not found at: {jsonl_path}"
            
            # Create inference arguments
            inference_args = InferenceArgs(
                jsonl_path=jsonl_path,
                pretrained_model_name=self.base_model_path
            )
            
            # Create dataset using the same pipeline as training
            print("Creating inference dataset...")
            inference_dataset = make_train_dataset(inference_args, self.tokenizers, accelerator=None, noise_size=512)
            
            # Create dataloader with batch_size=1
            inference_dataloader = torch.utils.data.DataLoader(
                inference_dataset,
                batch_size=1,
                shuffle=False,
                collate_fn=collate_fn,
                num_workers=0,
            )
            
            # Get the first (and only) batch
            batch = next(iter(inference_dataloader))
            
            # Extract data from batch
            caption = batch["prompts"][0] if isinstance(batch["prompts"], list) else batch["prompts"]
            call_ids = batch["call_ids"]
            
            print(f"\n{'='*60}")
            print(f"Running inference with:")
            print(f"  Checkpoint: {checkpoint_name}")
            print(f"  Prompt: {caption}")
            print(f"  Call IDs: {call_ids}")
            print(f"  Height: {height}, Width: {width}")
            print(f"  Seed: {seed}, Steps: {num_inference_steps}")
            print(f"  Guidance Scale: {guidance_scale}")
            print(f"{'='*60}\n")
            
            # Convert spatial condition tensors to PIL Images
            spatial_imgs = self.tensor_to_image_list(batch["cond_pixel_values"])
            
            # Prepare cuboids segmentation masks
            cuboids_segmasks = batch.get("cuboids_segmasks", None)
            
            # Prepare joint attention kwargs
            joint_attention_kwargs = {
                "call_ids": call_ids,
                "cuboids_segmasks": cuboids_segmasks,
            }
            
            print(f"Spatial images: {len(spatial_imgs)}")
            print(f"{len(cuboids_segmasks) = }, {cuboids_segmasks[0].shape = }")
            # print(f"Cuboids segmasks shape: {cuboids_segmasks.shape if cuboids_segmasks is not None else 'None'}")
            cuboids_segmasks = torch.stack(cuboids_segmasks, dim=0) if cuboids_segmasks is not None else None 
            
            # Run inference
            image = self.pipe(
                prompt=caption,
                height=int(height),
                width=int(width),
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                max_sequence_length=max_sequence_length,
                generator=torch.Generator("cpu").manual_seed(seed),
                subject_images=[],  # No subject images for spatial conditioning
                spatial_images=spatial_imgs,
                cond_size=512,
                **joint_attention_kwargs
            ).images[0]
            
            # Clear cache
            self.clear_cache()
            torch.cuda.empty_cache()
            
            success_msg = f"✅ Successfully generated image using {checkpoint_name}"
            print(f"\n{success_msg}\n")
            
            return True, image, success_msg
            
        except Exception as e:
            error_msg = f"❌ Inference failed: {str(e)}"
            print(f"\n{error_msg}\n")
            import traceback
            traceback.print_exc()
            return False, None, error_msg


# Global inference engine instance
_inference_engine: Optional[InferenceEngine] = None


def initialize_inference_engine(base_model_path: str = config.PRETRAINED_MODEL_NAME_OR_PATH):
    """
    Initialize the global inference engine.
    Should be called once when the Gradio demo starts.
    """
    global _inference_engine
    
    if _inference_engine is None:
        print("\n" + "="*60)
        print("INITIALIZING INFERENCE ENGINE")
        print("="*60 + "\n")
        
        _inference_engine = InferenceEngine(base_model_path=base_model_path)
        
        print("\n" + "="*60)
        print("INFERENCE ENGINE READY")
        print("="*60 + "\n")
    
    return _inference_engine


def get_inference_engine() -> InferenceEngine:
    """
    Get the global inference engine instance.
    Raises an error if not initialized.
    """
    global _inference_engine
    
    if _inference_engine is None:
        raise RuntimeError(
            "Inference engine not initialized. "
            "Call initialize_inference_engine() first."
        )
    
    return _inference_engine


def run_inference_from_gradio(
    checkpoint_name: str,
    height: int = 512,
    width: int = 512,
    seed: int = 42,
    guidance_scale: float = 3.5,
    num_inference_steps: int = 25,
    jsonl_path: str = os.path.join(config.GRADIO_FILES_DIR, "cuboids.jsonl")
) -> Tuple[bool, Optional[Image.Image], str]:
    """
    Wrapper function to run inference from Gradio interface.
    
    Args:
        checkpoint_name: Name of checkpoint to use (from dropdown)
        height: Output image height
        width: Output image width
        seed: Random seed
        guidance_scale: Guidance scale
        num_inference_steps: Number of denoising steps
        jsonl_path: Path to JSONL file with inference data
        
    Returns:
        Tuple of (success, generated_image, status_message)
    """
    engine = get_inference_engine()
    
    return engine.run_inference(
        jsonl_path=jsonl_path,
        checkpoint_name=checkpoint_name,
        height=height,
        width=width,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_inference_steps
    )