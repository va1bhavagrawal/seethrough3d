import argparse
import copy
import logging
import random 
import math
import os
import shutil
import gc 
from contextlib import nullcontext
from pathlib import Path
import re
from safetensors.torch import save_file

from PIL import Image
import numpy as np
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed

from tqdm.auto import tqdm
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast

import diffusers

from diffusers import (
	AutoencoderKL,
	FlowMatchEulerDiscreteScheduler
)
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
	cast_training_params,
	compute_density_for_timestep_sampling,
	compute_loss_weighting_for_sd3,
)
import os.path as osp 
from diffusers.utils.torch_utils import is_compiled_module
from diffusers.utils import (
	check_min_version,
	is_wandb_available,
	convert_unet_state_dict_to_peft
)

from src.lora_helper import *
from src.pipeline import FluxPipeline, resize_position_encoding, prepare_latent_subject_ids
from src.layers import MultiDoubleStreamBlockLoraProcessor, MultiSingleStreamBlockLoraProcessor
from src.transformer_flux import FluxTransformer2DModel
from src.jsonl_datasets import make_train_dataset, collate_fn

if is_wandb_available():
	import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.31.0.dev0")

logger = get_logger(__name__)

import matplotlib.pyplot as plt
import torch


def load_text_encoders(args, class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision, variant=args.variant
    )
    text_encoder_two = class_two.from_pretrained(
        args.pretrained_model_name_or_path, subfolder="text_encoder_2", revision=args.revision, variant=args.variant
    )
    return text_encoder_one, text_encoder_two


def _encode_prompt_with_t5(
	text_encoder,
	tokenizer,
	max_sequence_length=512,
	prompt=None,
	num_images_per_prompt=1,
	device=None,
	text_input_ids=None,
):
	prompt = [prompt] if isinstance(prompt, str) else prompt
	batch_size = len(prompt)

	if tokenizer is not None:
		text_inputs = tokenizer(
			prompt,
			padding="max_length",
			max_length=max_sequence_length,
			truncation=True,
			return_length=False,
			return_overflowing_tokens=False,
			return_tensors="pt",
		)
		text_input_ids = text_inputs.input_ids
	else:
		if text_input_ids is None:
			raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

	prompt_embeds = text_encoder(text_input_ids.to(device))[0]

	if hasattr(text_encoder, "module"):
		dtype = text_encoder.module.dtype
	else:
		dtype = text_encoder.dtype
	prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

	_, seq_len, _ = prompt_embeds.shape

	# duplicate text embeddings and attention mask for each generation per prompt, using mps friendly method
	prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
	prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

	return prompt_embeds


def _encode_prompt_with_clip(
	text_encoder,
	tokenizer,
	prompt: str,
	device=None,
	text_input_ids=None,
	num_images_per_prompt: int = 1,
):
	prompt = [prompt] if isinstance(prompt, str) else prompt
	batch_size = len(prompt)

	if tokenizer is not None:
		text_inputs = tokenizer(
			prompt,
			padding="max_length",
			max_length=77,
			truncation=True,
			return_overflowing_tokens=False,
			return_length=False,
			return_tensors="pt",
		)

		text_input_ids = text_inputs.input_ids
	else:
		if text_input_ids is None:
			raise ValueError("text_input_ids must be provided when the tokenizer is not specified")

	prompt_embeds = text_encoder(text_input_ids.to(device), output_hidden_states=False)

	if hasattr(text_encoder, "module"):
		dtype = text_encoder.module.dtype
	else:
		dtype = text_encoder.dtype
	# Use pooled output of CLIPTextModel
	prompt_embeds = prompt_embeds.pooler_output
	prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

	# duplicate text embeddings for each generation per prompt, using mps friendly method
	prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
	prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)

	return prompt_embeds


def encode_prompt(
	text_encoders,
	tokenizers,
	prompt: str,
	max_sequence_length,
	device=None,
	num_images_per_prompt: int = 1,
	text_input_ids_list=None,
):
	prompt = [prompt] if isinstance(prompt, str) else prompt

	if hasattr(text_encoders[0], "module"):
		dtype = text_encoders[0].module.dtype
	else:
		dtype = text_encoders[0].dtype

	pooled_prompt_embeds = _encode_prompt_with_clip(
		text_encoder=text_encoders[0],
		tokenizer=tokenizers[0],
		prompt=prompt,
		device=device if device is not None else text_encoders[0].device,
		num_images_per_prompt=num_images_per_prompt,
		text_input_ids=text_input_ids_list[0] if text_input_ids_list else None,
	)

	prompt_embeds = _encode_prompt_with_t5(
		text_encoder=text_encoders[1],
		tokenizer=tokenizers[1],
		max_sequence_length=max_sequence_length,
		prompt=prompt,
		num_images_per_prompt=num_images_per_prompt,
		device=device if device is not None else text_encoders[1].device,
		text_input_ids=text_input_ids_list[1] if text_input_ids_list else None,
	)

	text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)

	return prompt_embeds, pooled_prompt_embeds, text_ids


def visualize_training_data(batch, vae, model_input, noisy_model_input, cond_input, args, global_step, accelerator):
	"""
	Visualize training data including all entities from the batch.
	
	Args:
		batch: Training batch containing data
		vae: VAE model for decoding latents
		model_input: Clean latents before adding noise
		noisy_model_input: Noisy latents passed to transformer
		cond_input: Spatial condition latents (may be None)
		args: Training arguments
		global_step: Current training step
		accelerator: Accelerator instance
	"""
	
	# Check availability of conditions
	has_spatial_condition = batch["cond_pixel_values"] is not None
	has_cuboids_segmasks = "cuboids_segmasks" in batch and batch["cuboids_segmasks"] is not None
	has_cuboids_segmasks_bev = "cuboids_segmasks_bev" in batch and batch["cuboids_segmasks_bev"] is not None
	
	# Initialize variables
	spatial_img = None
	
	with torch.no_grad():
		# Get VAE config for proper decoding
		vae_config_shift_factor = vae.config.shift_factor
		vae_config_scaling_factor = vae.config.scaling_factor
		vae_dtype = vae.dtype 
		vae = vae.to(torch.float32) 
		
		# Decode spatial condition if available
		if has_spatial_condition:
			cond_for_decode = (cond_input / vae_config_scaling_factor) + vae_config_shift_factor
			spatial_decoded = vae.decode(cond_for_decode.float()).sample
			spatial_decoded = (spatial_decoded / 2 + 0.5).clamp(0, 1)  # Normalize to [0,1]
			spatial_img = spatial_decoded[0].float().cpu().permute(1, 2, 0).numpy()
		
		# Decode clean model input
		clean_for_decode = (model_input / vae_config_scaling_factor) + vae_config_shift_factor
		clean_decoded = vae.decode(clean_for_decode.float()).sample
		clean_decoded = (clean_decoded / 2 + 0.5).clamp(0, 1)
		
		# Decode noisy model input
		noisy_for_decode = (noisy_model_input / vae_config_scaling_factor) + vae_config_shift_factor
		noisy_decoded = vae.decode(noisy_for_decode.float()).sample
		noisy_decoded = (noisy_decoded / 2 + 0.5).clamp(0, 1)
		
		# Convert to CPU and numpy for visualization (take first batch item)
		clean_img = clean_decoded[0].float().cpu().permute(1, 2, 0).numpy()
		noisy_img = noisy_decoded[0].float().cpu().permute(1, 2, 0).numpy()
		
		# Get text prompt and other info
		text_prompt = batch["prompts"][0] if isinstance(batch["prompts"], list) else batch["prompts"]
		call_id = batch["call_ids"][0] if batch["call_ids"] is not None else "N/A"
		
		# Create figure with more subplots to accommodate all entities including BEV
		fig, axes = plt.subplots(4, 3, figsize=(18, 24))
		# fig.suptitle(f'Training Data Visualization - Step {global_step}', fontsize=16)
		
		# Spatial condition (0,0)
		if has_spatial_condition and spatial_img is not None:
			axes[0, 0].imshow(spatial_img)
			axes[0, 0].set_title('Spatial Condition')
		else:
			axes[0, 0].text(0.5, 0.5, 'NOT AVAILABLE', 
						   horizontalalignment='center', verticalalignment='center',
						   transform=axes[0, 0].transAxes, fontsize=14, fontweight='bold')
			axes[0, 0].set_title('Spatial Condition')
		axes[0, 0].axis('off')
		
		# Clean model input (0,2)
		axes[0, 2].imshow(clean_img)
		axes[0, 2].set_title('Clean Model Input')
		axes[0, 2].axis('off')
		
		# Noisy model input (1,0)
		axes[1, 0].imshow(noisy_img)
		axes[1, 0].set_title('Noisy Model Input')
		axes[1, 0].axis('off')
		
		# Cuboids segmentation masks with legend (1,1 and 1,2)
		if has_cuboids_segmasks:
			segmask = batch["cuboids_segmasks"][0].float().cpu().numpy()  # Shape: (n_subjects, h, w)
			n_subjects, h, w = segmask.shape
			
			# Only use first 4 subjects for visualization
			n_subjects_to_show = min(4, n_subjects)
			
			# Create colored segmentation visualization
			np.random.seed(42)  # For consistent colors
			colors = np.random.rand(n_subjects_to_show + 1, 3)  # +1 for background
			colors[0] = [0, 0, 0]  # Background is black
			
			# Create 2x2 grid of individual subject masks
			grid_h, grid_w = 2, 2
			combined_mask = np.zeros((h * grid_h, w * grid_w, 3))
			
			for idx in range(n_subjects_to_show):
				row = idx // grid_w
				col = idx % grid_w
				
				# Create binary mask for this subject
				subject_mask = np.zeros((h, w, 3))
				mask = segmask[idx] > 0.5  # Binary threshold
				subject_mask[mask] = colors[idx + 1]
				
				# Place in grid
				combined_mask[row*h:(row+1)*h, col*w:(col+1)*w] = subject_mask
			
			axes[1, 1].imshow(combined_mask)
			axes[1, 1].set_title('Cuboids Segmentation (2x2 Grid)')
			axes[1, 1].axis('off')
			
			# Create legend in the next subplot (1,2) - only for first 4 subjects
			axes[1, 2].set_xlim(0, 1)
			axes[1, 2].set_ylim(0, 1)
			
			# Add legend entries
			legend_y_positions = np.linspace(0.9, 0.1, n_subjects_to_show + 1)
			axes[1, 2].text(0.1, legend_y_positions[0], f"Background", 
						   color=colors[0], fontsize=12, fontweight='bold')
			
			for subject_idx in range(n_subjects_to_show):
				axes[1, 2].text(0.1, legend_y_positions[subject_idx + 1], 
							   f"Subject {subject_idx}", 
							   color=colors[subject_idx + 1], fontsize=12, fontweight='bold')
			
			axes[1, 2].set_title('Segmentation Legend (First 4)')
			axes[1, 2].axis('off')
		else:
			axes[1, 1].text(0.5, 0.5, 'NOT AVAILABLE', 
						   horizontalalignment='center', verticalalignment='center',
						   transform=axes[1, 1].transAxes, fontsize=14, fontweight='bold')
			axes[1, 1].set_title('Cuboids Segmentation')
			axes[1, 1].axis('off')
			
			axes[1, 2].text(0.5, 0.5, 'NOT AVAILABLE', 
						   horizontalalignment='center', verticalalignment='center',
						   transform=axes[1, 2].transAxes, fontsize=14, fontweight='bold')
			axes[1, 2].set_title('Segmentation Legend')
			axes[1, 2].axis('off')
		
		# BEV Cuboids segmentation masks with legend (2,0 and 2,1)
		if has_cuboids_segmasks_bev:
			segmask_bev = batch["cuboids_segmasks_bev"][0].float().cpu().numpy()  # Shape: (n_subjects, h, w)
			n_subjects_bev, h_bev, w_bev = segmask_bev.shape
			
			# Create colored segmentation visualization for BEV (use different seed for different colors)
			np.random.seed(123)  # Different seed for BEV colors
			colors_bev = np.random.rand(n_subjects_bev + 1, 3)  # +1 for background
			colors_bev[0] = [0, 0, 0]  # Background is black
			
			# Create RGB image from BEV segmentation
			colored_segmask_bev = np.zeros((h_bev, w_bev, 3))
			for subject_idx in range(n_subjects_bev):
				mask_bev = segmask_bev[subject_idx] > 0.5  # Binary threshold
				colored_segmask_bev[mask_bev] = colors_bev[subject_idx + 1]
			
			axes[2, 0].imshow(colored_segmask_bev)
			axes[2, 0].set_title('BEV Cuboids Segmentation')
			axes[2, 0].axis('off')
			
			# Create BEV legend in the next subplot (2,1)
			axes[2, 1].set_xlim(0, 1)
			axes[2, 1].set_ylim(0, 1)
			
			# Add BEV legend entries
			legend_y_positions_bev = np.linspace(0.9, 0.1, n_subjects_bev + 1)
			axes[2, 1].text(0.1, legend_y_positions_bev[0], f"Background", 
						   color=colors_bev[0], fontsize=12, fontweight='bold')
			
			for subject_idx in range(n_subjects_bev):
				axes[2, 1].text(0.1, legend_y_positions_bev[subject_idx + 1], 
							   f"Subject {subject_idx}", 
							   color=colors_bev[subject_idx + 1], fontsize=12, fontweight='bold')
			
			axes[2, 1].set_title('BEV Segmentation Legend')
			axes[2, 1].axis('off')
		else:
			axes[2, 0].text(0.5, 0.5, 'NOT AVAILABLE', 
					   horizontalalignment='center', verticalalignment='center',
						   transform=axes[2, 0].transAxes, fontsize=14, fontweight='bold')
			axes[2, 0].set_title('BEV Cuboids Segmentation')
			axes[2, 0].axis('off')
		
			axes[2, 1].text(0.5, 0.5, 'NOT AVAILABLE', 
						   horizontalalignment='center', verticalalignment='center',
						   transform=axes[2, 1].transAxes, fontsize=14, fontweight='bold')
			axes[2, 1].set_title('BEV Segmentation Legend')
			axes[2, 1].axis('off')
		
		# Text prompt and call ID (2,2)
		axes[2, 2].text(0.5, 0.5, f'Text Prompt:\n\n"{text_prompt}"\n\nCall ID: {call_id}', 
					   horizontalalignment='center', verticalalignment='center',
					   transform=axes[2, 2].transAxes, fontsize=12, wrap=True)
		axes[2, 2].set_title('Text Prompt & Call ID')
		axes[2, 2].axis('off')
		
		# Pixel values info (3,0)
		pixel_info = f'Pixel Values Shape: {batch["pixel_values"].shape}\n'
		if has_spatial_condition:
			pixel_info += f'Spatial Shape: {batch["cond_pixel_values"].shape}\n'
		if has_cuboids_segmasks:
			pixel_info += f'Cuboids Segmasks: {len(batch["cuboids_segmasks"])}\n'
		if has_cuboids_segmasks_bev:
			pixel_info += f'BEV Segmasks: {len(batch["cuboids_segmasks_bev"])}'
		
		axes[3, 0].text(0.5, 0.5, pixel_info, 
					   horizontalalignment='center', verticalalignment='center',
					   transform=axes[3, 0].transAxes, fontsize=10, fontfamily='monospace')
		axes[3, 0].set_title('Tensor Shapes')
		axes[3, 0].axis('off')
		
		# Training info (3,1)
		training_info = f'Global Step: {global_step}\nConditions:\nSpatial: {"✓" if has_spatial_condition else "✗"}\nSubject: {"fuck you"}\nSegmasks: {"✓" if has_cuboids_segmasks else "✗"}\nBEV Segmasks: {"✓" if has_cuboids_segmasks_bev else "✗"}'
		axes[3, 1].text(0.5, 0.5, training_info, 
					   horizontalalignment='center', verticalalignment='center',
					   transform=axes[3, 1].transAxes, fontsize=12, fontfamily='monospace')
		axes[3, 1].set_title('Training Info')
		axes[3, 1].axis('off')
		
		# Additional info (3,2) - can be used for any extra debugging info
		axes[3, 2].text(0.5, 0.5, 'Additional Info\n(Reserved)', 
					   horizontalalignment='center', verticalalignment='center',
					   transform=axes[3, 2].transAxes, fontsize=12, fontfamily='monospace')
		axes[3, 2].set_title('Reserved')
		axes[3, 2].axis('off')
		
		plt.tight_layout()
		
		# Save the visualization
		save_dir = os.path.join(args.output_dir, "visualizations")
		os.makedirs(save_dir, exist_ok=True)
		save_path = os.path.join(save_dir, f"training_vis_step_{global_step}.png")
		plt.savefig(save_path, dpi=150, bbox_inches='tight')
		plt.close()
		
		logger.info(f"Training visualization saved to {save_path}")

		vae = vae.to(vae_dtype)

def import_model_class_from_model_name_or_path(
		pretrained_model_name_or_path: str, revision: str, subfolder: str = "text_encoder"
):
	text_encoder_config = PretrainedConfig.from_pretrained(
		pretrained_model_name_or_path, subfolder=subfolder, revision=revision
	)
	model_class = text_encoder_config.architectures[0]
	if model_class == "CLIPTextModel":
		from transformers import CLIPTextModel

		return CLIPTextModel
	elif model_class == "T5EncoderModel":
		from transformers import T5EncoderModel

		return T5EncoderModel
	else:
		raise ValueError(f"{model_class} is not supported.")


def parse_args(input_args=None):
	parser = argparse.ArgumentParser(description="Simple example of a training script.")
	parser.add_argument("--lora_num", type=int, default=2, help="number of the lora.")
	parser.add_argument("--cond_size", type=int, default=512, help="size of the condition data.")
	parser.add_argument("--debug", type=int, default=0, help="whether to enter debug mode -- visualizations, gradient checks, etc.")
	parser.add_argument("--mode",type=str,default=None,help="The mode of the controller. Choose between ['depth', 'pose', 'canny'].")
	parser.add_argument("--run_name",type=str,required=True,help="the name of the wandb run")
	parser.add_argument(
		"--train_data_dir",
		type=str,
		default="",
		help=(
			"A folder containing the training data. Folder contents must follow the structure described in"
			" https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
			" must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
		),
	)
	parser.add_argument(
		"--inference_embeds_dir",
		type=str,
		default=None, 
		help=(
			"the captions for images"
		),
	)
	parser.add_argument(
		"--pretrained_model_name_or_path",
		type=str,
		default="",
		required=False,
		help="Path to pretrained model or model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--revision",
		type=str,
		default=None,
		required=False,
		help="Revision of pretrained model identifier from huggingface.co/models.",
	)
	parser.add_argument(
		"--variant",
		type=str,
		default=None,
		help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
	)
	parser.add_argument(
		"--spatial_column",
		type=str,
		default="None",
		help="The column of the dataset containing the canny image. By "
			 "default, the standard Image Dataset maps out 'file_name' "
			 "to 'image'.",
	)
	parser.add_argument(
		"--target_column",
		type=str,
		default="image",
		help="The column of the dataset containing the target image. By "
			 "default, the standard Image Dataset maps out 'file_name' "
			 "to 'image'.",
	)
	parser.add_argument(
		"--caption_column",
		type=str,
		default="caption_left,caption_right",
		help="The column of the dataset containing the instance prompt for each image",
	)
	parser.add_argument("--repeats", type=int, default=1, help="How many times to repeat the training data.")
	parser.add_argument(
		"--max_sequence_length",
		type=int,
		default=512,
		help="Maximum sequence length to use with with the T5 text encoder",
	)
	parser.add_argument(
		"--ranks",
		type=int,
		nargs="+",
		default=[128],
		help=("The dimension of the LoRA update matrices."),
	)
	parser.add_argument(
		"--network_alphas",
		type=int,
		nargs="+",
		default=[128],
		help=("The dimension of the LoRA update matrices."),
	)
	parser.add_argument(
		"--output_dir",
		type=str,
		required=True,
		help="The output directory where the model predictions and checkpoints will be written.",
	)
	parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
	parser.add_argument(
		"--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
	)
	parser.add_argument("--num_train_epochs", type=int, default=50)
	parser.add_argument(
		"--max_train_steps",
		type=int,
		default=None,
		help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
	)
	parser.add_argument(
		"--checkpointing_steps",
		type=int,
		default=1000,
		help=(
			"Save a checkpoint of the training state every X updates. These checkpoints can be used both as final"
			" checkpoints in case they are better than the last checkpoint, and are also suitable for resuming"
			" training using `--resume_from_checkpoint`."
		),
	)
	parser.add_argument(
		"--resume_from_checkpoint",
		type=str,
		default=None,
		help=(
			"Whether training should be resumed from a previous checkpoint. Use a path saved by"
			' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
		),
	)
	parser.add_argument(
		"--gradient_accumulation_steps",
		type=int,
		default=1,
		help="Number of updates steps to accumulate before performing a backward/update pass.",
	)
	parser.add_argument(
		"--gradient_checkpointing",
		action="store_true",
		help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
	)
	parser.add_argument(
		"--learning_rate",
		type=float,
		default=1e-4,
		help="Initial learning rate (after the potential warmup period) to use.",
	)

	parser.add_argument(
		"--guidance_scale",
		type=float,
		default=1,
		help="the FLUX.1 dev variant is a guidance distilled model",
	)
	parser.add_argument(
		"--scale_lr",
		action="store_true",
		default=False,
		help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
	)
	parser.add_argument(
		"--lr_scheduler",
		type=str,
		default="constant",
		help=(
			'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
			' "constant", "constant_with_warmup"]'
		),
	)
	parser.add_argument(
		"--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
	)
	parser.add_argument(
		"--lr_num_cycles",
		type=int,
		default=1,
		help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
	)
	parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
	parser.add_argument(
		"--dataloader_num_workers",
		type=int,
		default=2,
		help=(
			"Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
		),
	)
	parser.add_argument(
		"--weighting_scheme",
		type=str,
		default="none",
		choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
		help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
	)
	parser.add_argument(
		"--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
	)
	parser.add_argument(
		"--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
	)
	parser.add_argument(
		"--mode_scale",
		type=float,
		default=1.29,
		help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
	)
	parser.add_argument(
		"--optimizer",
		type=str,
		default="AdamW",
		help=('The optimizer type to use. Choose between ["AdamW", "prodigy"]'),
	)

	parser.add_argument(
		"--use_8bit_adam",
		action="store_true",
		help="Whether or not to use 8-bit Adam from bitsandbytes. Ignored if optimizer is not set to AdamW",
	)

	parser.add_argument(
		"--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam and Prodigy optimizers."
	)
	parser.add_argument(
		"--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam and Prodigy optimizers."
	)
	parser.add_argument(
		"--prodigy_beta3",
		type=float,
		default=None,
		help="coefficients for computing the Prodigy stepsize using running averages. If set to None, "
			 "uses the value of square root of beta2. Ignored if optimizer is adamW",
	)
	parser.add_argument("--prodigy_decouple", type=bool, default=True, help="Use AdamW style decoupled weight decay")
	parser.add_argument("--adam_weight_decay", type=float, default=1e-04, help="Weight decay to use for unet params")
	parser.add_argument(
		"--adam_weight_decay_text_encoder", type=float, default=1e-03, help="Weight decay to use for text_encoder"
	)

	parser.add_argument(
		"--adam_epsilon",
		type=float,
		default=1e-08,
		help="Epsilon value for the Adam optimizer and Prodigy optimizers.",
	)

	parser.add_argument(
		"--prodigy_use_bias_correction",
		type=bool,
		default=True,
		help="Turn on Adam's bias correction. True by default. Ignored if optimizer is adamW",
	)
	parser.add_argument(
		"--prodigy_safeguard_warmup",
		type=bool,
		default=True,
		help="Remove lr from the denominator of D estimate to avoid issues during warm-up stage. True by default. "
			 "Ignored if optimizer is adamW",
	)
	parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
	parser.add_argument(
		"--logging_dir",
		type=str,
		default="logs",
		help=(
			"[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
			" *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
		),
	)
	parser.add_argument(
		"--report_to",
		type=str,
		default="tensorboard",
		help=(
			'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
			' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
		),
	)
	parser.add_argument(
		"--mixed_precision",
		type=str,
		default="bf16",
		choices=["no", "fp16", "bf16"],
		help=(
			"Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
			" 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
			" flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
		),
	)
	parser.add_argument(
		"--upcast_before_saving",
		action="store_true",
		default=False,
		help=(
			"Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
			"Defaults to precision dtype used for training to save memory"
		),
	)

	if input_args is not None:
		args = parser.parse_args(input_args)
	else:
		args = parser.parse_args()
	return args


def main(args):
	if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
		# due to pytorch#99272, MPS does not yet support bfloat16.
		raise ValueError(
			"Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
		)

	if args.resume_from_checkpoint is not None: 
		assert osp.exists(args.resume_from_checkpoint), f"Make sure that the `resume_from_checkpoint` {args.resume_from_checkpoint} exists." 
		args.pretrained_lora_path = osp.join(args.resume_from_checkpoint, f"lora.safetensors") 
		assert osp.exists(args.pretrained_lora_path), f"Make sure that the `pretrained_lora_path` {args.pretrained_lora_path} exists." 
	else: 
		args.pretrained_lora_path = None 

	args.output_dir = osp.join(args.output_dir, args.run_name) 
	args.logging_dir = osp.join(args.output_dir, args.logging_dir) 
	os.makedirs(args.output_dir, exist_ok=True)
	os.makedirs(args.logging_dir, exist_ok=True)
	logging_dir = Path(args.output_dir, args.logging_dir)

	if args.spatial_column == "None":
		args.spatial_column = None

	accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
	# kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
	accelerator = Accelerator(
		gradient_accumulation_steps=args.gradient_accumulation_steps,
		mixed_precision=args.mixed_precision,
		log_with=args.report_to, 
		project_config=accelerator_project_config,
		# kwargs_handlers=[kwargs],
	)

	def save_model_hook(models, weights, output_dir):
		pass 

	def load_model_hook(models, input_dir):
		pass 

	# Disable AMP for MPS.
	if torch.backends.mps.is_available():
		accelerator.native_amp = False

	if args.report_to == "wandb":
		if not is_wandb_available():
			raise ImportError("Make sure to install wandb if you want to use it for logging during training.")

	# Make one log on every process with the configuration for debugging.
	logging.basicConfig(
		format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
		datefmt="%m/%d/%Y %H:%M:%S",
		level=logging.INFO,
	)
	logger.info(accelerator.state, main_process_only=False)
	if accelerator.is_local_main_process:
		transformers.utils.logging.set_verbosity_warning()
		diffusers.utils.logging.set_verbosity_info()
	else:
		transformers.utils.logging.set_verbosity_error()
		diffusers.utils.logging.set_verbosity_error()

	# If passed along, set the training seed now.
	if args.seed is not None:
		set_seed(args.seed)

	# Handle the repository creation
	if accelerator.is_main_process:
		if args.output_dir is not None:
			os.makedirs(args.output_dir, exist_ok=True)

	# Load the tokenizers
	tokenizer_one = CLIPTokenizer.from_pretrained(
		args.pretrained_model_name_or_path,
		subfolder="tokenizer",
		revision=args.revision,
	)
	tokenizer_two = T5TokenizerFast.from_pretrained(
		args.pretrained_model_name_or_path,
		subfolder="tokenizer_2",
		revision=args.revision,
	)

	# Load scheduler and models
	noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="scheduler"
	)
	noise_scheduler_copy = copy.deepcopy(noise_scheduler)
	gc.collect() 
	torch.cuda.empty_cache() 

	text_encoder_cls_one = import_model_class_from_model_name_or_path(
		args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder"
	)
	text_encoder_cls_two = import_model_class_from_model_name_or_path(
		args.pretrained_model_name_or_path, args.revision, subfolder="text_encoder_2"
	)
	if args.inference_embeds_dir is None: 
		text_encoder_one, text_encoder_two = load_text_encoders(args, text_encoder_cls_one, text_encoder_cls_two)
	else: 
		assert osp.exists(args.inference_embeds_dir), f"Make sure that the `inference_embeds_dir` {args.inference_embeds_dir} exists." 
	vae = AutoencoderKL.from_pretrained(
		args.pretrained_model_name_or_path,
		subfolder="vae",
		revision=args.revision,
		variant=args.variant,
	)
	transformer = FluxTransformer2DModel.from_pretrained(
		args.pretrained_model_name_or_path, subfolder="transformer", revision=args.revision, variant=args.variant
	)

	# We only train the additional adapter LoRA layers
	transformer.requires_grad_(True)
	vae.requires_grad_(False)
	if args.inference_embeds_dir is None: 
		text_encoder_one.requires_grad_(False)
		text_encoder_two.requires_grad_(False)

	# For mixed precision training we cast all non-trainable weights (vae, text_encoder and transformer) to half-precision
	# as these weights are only used for inference, keeping weights in full precision is not required.
	weight_dtype = torch.float32
	if accelerator.mixed_precision == "fp16":
		weight_dtype = torch.float16
	elif accelerator.mixed_precision == "bf16":
		weight_dtype = torch.bfloat16

	if torch.backends.mps.is_available() and weight_dtype == torch.bfloat16:
		# due to pytorch#99272, MPS does not yet support bfloat16.
		raise ValueError(
			"Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
		)

	vae.to(accelerator.device, dtype=weight_dtype)
	transformer.to(accelerator.device, dtype=weight_dtype)
	if args.inference_embeds_dir is None: 
		text_encoder_one.to(accelerator.device, dtype=torch.float32)
		text_encoder_two.to(accelerator.device, dtype=torch.float32)

	if args.gradient_checkpointing:
		transformer.enable_gradient_checkpointing()

	#### lora_layers ####
	if args.pretrained_lora_path is not None:
		lora_path = args.pretrained_lora_path
		checkpoint = load_checkpoint(lora_path)
		lora_attn_procs = {}
		double_blocks_idx = list(range(19))
		single_blocks_idx = list(range(38))
		number = 1
		for name, attn_processor in transformer.attn_processors.items():
			match = re.search(r'\.(\d+)\.', name)
			if match:
				layer_index = int(match.group(1))
			
			if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
				lora_state_dicts = {}
				for key, value in checkpoint.items():
					# Match based on the layer index in the key (assuming the key contains layer index)
					if re.search(r'\.(\d+)\.', key):
						checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
						if checkpoint_layer_index == layer_index and key.startswith("transformer_blocks"):
							lora_state_dicts[key] = value
				
				print("setting LoRA Processor for", name)
				lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
					dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
				)
				
				# Load the weights from the checkpoint dictionary into the corresponding layers
				for n in range(number):
					lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
					lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
					lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
					lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
					lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
					lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
					lora_attn_procs[name].proj_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.down.weight', None)
					lora_attn_procs[name].proj_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.proj_loras.{n}.up.weight', None)
				
			elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
				
				lora_state_dicts = {}
				for key, value in checkpoint.items():
					# Match based on the layer index in the key (assuming the key contains layer index)
					if re.search(r'\.(\d+)\.', key):
						checkpoint_layer_index = int(re.search(r'\.(\d+)\.', key).group(1))
						if checkpoint_layer_index == layer_index and key.startswith("single_transformer_blocks"):
							lora_state_dicts[key] = value
				
				print("setting LoRA Processor for", name)        
				lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
					dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
				)
				
				# Load the weights from the checkpoint dictionary into the corresponding layers
				for n in range(number):
					lora_attn_procs[name].q_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.down.weight', None)
					lora_attn_procs[name].q_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.q_loras.{n}.up.weight', None)
					lora_attn_procs[name].k_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.down.weight', None)
					lora_attn_procs[name].k_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.k_loras.{n}.up.weight', None)
					lora_attn_procs[name].v_loras[n].down.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.down.weight', None)
					lora_attn_procs[name].v_loras[n].up.weight.data = lora_state_dicts.get(f'{name}.v_loras.{n}.up.weight', None)
			else:
				lora_attn_procs[name] = FluxAttnProcessor2_0()
	else:
		lora_attn_procs = {}
		double_blocks_idx = list(range(19))
		single_blocks_idx = list(range(38))
		for name, attn_processor in transformer.attn_processors.items():
			match = re.search(r'\.(\d+)\.', name)
			if match:
				layer_index = int(match.group(1))
			if name.startswith("transformer_blocks") and layer_index in double_blocks_idx:
				lora_state_dicts = {}
				print("setting LoRA Processor for", name)
				lora_attn_procs[name] = MultiDoubleStreamBlockLoraProcessor(
					dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
				)

			elif name.startswith("single_transformer_blocks") and layer_index in single_blocks_idx:
				print("setting LoRA Processor for", name)
				lora_attn_procs[name] = MultiSingleStreamBlockLoraProcessor(
					dim=3072, ranks=args.ranks, network_alphas=args.network_alphas, lora_weights=[1 for _ in range(args.lora_num)], device=accelerator.device, dtype=weight_dtype, cond_width=args.cond_size, cond_height=args.cond_size, n_loras=args.lora_num
				)

			else:
				lora_attn_procs[name] = attn_processor        
	######################
	transformer.set_attn_processor(lora_attn_procs)
	transformer.train()
	for n, param in transformer.named_parameters():
		if '_lora' not in n:
			param.requires_grad = False
	print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'M parameters')

	def unwrap_model(model):
		model = accelerator.unwrap_model(model)
		model = model._orig_mod if is_compiled_module(model) else model
		return model

	# Potentially load in the weights and states from a previous save
	if args.resume_from_checkpoint:
		foldername = osp.basename(args.resume_from_checkpoint) 
		first_epoch = epoch = int(foldername.split("-")[1].split("__")[0])  
		initial_global_step = global_step = int(foldername.split("-")[-1]) 
	else:
		initial_global_step = 0
		global_step = 0
		first_epoch = 0

	if args.scale_lr:
		args.learning_rate = (
				args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
		)

	# Make sure the trainable params are in float32.
	if args.mixed_precision == "fp16":
		models = [transformer]
		# only upcast trainable parameters (LoRA) into fp32
		cast_training_params(models, dtype=torch.float32)

	# Optimization parameters
	params_to_optimize = [p for p in transformer.parameters() if p.requires_grad]
	transformer_parameters_with_lr = {"params": params_to_optimize, "lr": args.learning_rate}
	print(sum([p.numel() for p in transformer.parameters() if p.requires_grad]) / 1000000, 'parameters')

	optimizer_class = torch.optim.AdamW
	optimizer = optimizer_class(
		[transformer_parameters_with_lr],
		betas=(args.adam_beta1, args.adam_beta2),
		weight_decay=args.adam_weight_decay,
		eps=args.adam_epsilon,
	)

	tokenizers = [tokenizer_one, tokenizer_two]

	# now, we will define a dataset for each epoch to make it easier to save the state   
	shuffled_jsonls = os.listdir(osp.dirname(args.train_data_dir)) 
	base_jsonl_name = osp.basename(args.train_data_dir).replace(".jsonl", "") 
	shuffled_jsonls = sorted([_ for _ in shuffled_jsonls if _.endswith('.jsonl') and "shuffled" in _ and base_jsonl_name in _])   
	shuffled_jsonls = [osp.join(osp.dirname(args.train_data_dir), _) for _ in shuffled_jsonls] 
	print(f"{shuffled_jsonls = }")
	assert len(shuffled_jsonls) > 0, f"Make sure that there are shuffled jsonl files in {osp.dirname(args.train_data_dir)}" 
	train_dataloaders = [] 
	for epoch in range(args.num_train_epochs): # prepare dataloader for each epoch, irrespective of the resume state  
		shuffled_idx = epoch % len(shuffled_jsonls) 
		train_data_file = shuffled_jsonls[shuffled_idx] 
		assert osp.exists(train_data_file), f"Make sure that the train data jsonl file {train_data_file} exists." 
		args.current_train_data_dir = train_data_file 
		train_dataset = make_train_dataset(args, tokenizers, accelerator) 
		train_dataloader = torch.utils.data.DataLoader(
			train_dataset,
			batch_size=args.train_batch_size,
			shuffle=False, # yayy!! reproducible experiments!
			collate_fn=collate_fn,
			num_workers=args.dataloader_num_workers,
		)
		train_dataloaders.append(train_dataloader) 

	vae_config_shift_factor = vae.config.shift_factor
	vae_config_scaling_factor = vae.config.scaling_factor

	# Scheduler and math around the number of training steps.
	overrode_max_train_steps = False
	num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)  
	if args.max_train_steps is None:
		args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
		overrode_max_train_steps = True

	lr_scheduler = get_scheduler(
		args.lr_scheduler,
		optimizer=optimizer,
		num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
		num_training_steps=args.max_train_steps * accelerator.num_processes,
		num_cycles=args.lr_num_cycles,
		power=args.lr_power,
	)


	accelerator.register_save_state_pre_hook(save_model_hook)
	accelerator.register_load_state_pre_hook(load_model_hook)
	optimizer, lr_scheduler = accelerator.prepare(
		optimizer, lr_scheduler 
	)

	print(f"before preparation, {len(train_dataloaders[0]) = }") 

	prepared_train_dataloaders = [] 
	for train_dataloader in train_dataloaders: 
		prepared_train_dataloaders.append(accelerator.prepare(train_dataloader)) 
	train_dataloaders = prepared_train_dataloaders 

	print(f"after preparation, {len(train_dataloaders[0]) = }") 

	if args.pretrained_lora_path is not None: 
		accelerator.load_state(osp.dirname(args.pretrained_lora_path))  

	# Explicitly move optimizer states to accelerator.device
	for state in optimizer.state.values():
		for k, v in state.items():
			if isinstance(v, torch.Tensor):
				state[k] = v.to(accelerator.device)

	transformer = accelerator.prepare(transformer) 

	# We need to recalculate our total training steps as the size of the training dataloader may have changed.
	num_update_steps_per_epoch = math.ceil(len(train_dataloaders[0]) / args.gradient_accumulation_steps)
	if overrode_max_train_steps:
		args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
	# Afterwards we recalculate our number of training epochs
	args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

	# We need to initialize the trackers we use, and also store our configuration.

	if accelerator.is_main_process:  
		accelerator.init_trackers(args.run_name) 


	# Train!
	total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

	logger.info("***** Running training *****")
	logger.info(f"  Num examples = {len(train_dataset)}")
	logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
	logger.info(f"  Num Epochs = {args.num_train_epochs}")
	logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
	logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
	logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
	logger.info(f"  Total optimization steps = {args.max_train_steps}")

	progress_bar = tqdm(
		range(0, args.max_train_steps),
		initial=initial_global_step,
		desc="Steps",
		# Only show the progress bar once on each machine.
		disable=not accelerator.is_local_main_process,
	)

	def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
		sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
		schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
		timesteps = timesteps.to(accelerator.device)
		step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

		sigma = sigmas[step_indices].flatten()
		while len(sigma.shape) < n_dim:
			sigma = sigma.unsqueeze(-1)
		return sigma
	
	# some fixed parameters 
	vae_scale_factor = 16
	height_cond = 2 * (args.cond_size // vae_scale_factor)
	width_cond = 2 * (args.cond_size // vae_scale_factor)        
	offset = 64

	num_training_visualizations = 10   

	skip_steps = initial_global_step - first_epoch * num_update_steps_per_epoch   
	print(f"{skip_steps = }")
	for epoch in range(first_epoch, args.num_train_epochs):
		transformer.train()
		train_dataloader = train_dataloaders[epoch] # use a new dataloader for each epoch  
		if epoch == first_epoch and skip_steps > 0:
			logger.info(f"Skipping {skip_steps} batches in epoch {epoch} due to resuming from checkpoint")
			# dataloader_iterator = skip_first_batches_manual(train_dataloader, skip_steps)
			dataloader_iterator = accelerator.skip_first_batches(train_dataloader, skip_steps) 
			# Convert back to enumerate format
			enumerated_dataloader = enumerate(dataloader_iterator, start=skip_steps)
		else: 
			enumerated_dataloader = enumerate(train_dataloader) 
		for step, batch in enumerated_dataloader: 
			progress_bar.set_description(f"epoch {epoch}, dataset_ids: {batch['index']}") 
			models_to_accumulate = [transformer]
			with accelerator.accumulate(models_to_accumulate):
				
				if args.inference_embeds_dir is None: 
					print(f"encoding {batch['prompts'] = }")
					# prompt_embeds, pooled_prompt_embeds, text_ids = text_encoding_pipeline.encode_prompt(
					# 	prompt=batch["prompts"], 
					# 	prompt_2=batch["prompts"], 
					# )
					# prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
					# 	text_encoders=[text_encoder_one, text_encoder_two],
					# 	tokenizers=[tokenizer_one, tokenizer_two], 
					# 	prompt=batch["prompts"], 
					# 	max_sequence_length=512, 
					# 	device=accelerator.device, 
					# )
					for i, prompt in enumerate(batch["prompts"]): 
						prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
							text_encoders=[text_encoder_one, text_encoder_two],
							tokenizers=[tokenizer_one, tokenizer_two], 
							prompt=prompt, 
							max_sequence_length=512, 
							device=accelerator.device, 
						)
						print(f"{prompt_embeds.shape = }, {pooled_prompt_embeds.shape = }, {text_ids.shape = }") 
						# checking if the cached embeddings match 
						inference_embeds_dir = "/archive/vaibhav.agrawal/a-bev-of-the-latents/inference_embeds_datasetv7_superhard"
						cached_prompt_path = osp.join(inference_embeds_dir, f"{'_'.join(prompt.lower().split())}.pth") 
						assert osp.exists(cached_prompt_path), f"Make sure that the cached prompt embedding {cached_prompt_path} exists." 
						cached_prompt_embeds = torch.load(cached_prompt_path, map_location="cpu") 
						assert torch.allclose(cached_prompt_embeds["prompt_embeds"].cpu().float(), prompt_embeds.cpu().float(), atol=1e-3), f"Cached prompt embeds for prompt {prompt} do not match the computed prompt embeds. Make sure that the cached prompt embeds are correct., {torch.mean(torch.abs(cached_prompt_embeds['prompt_embeds'].cpu().float() - prompt_embeds.cpu().float())) = }, {torch.mean(torch.abs(cached_prompt_embeds['prompt_embeds'].cpu().float())) = }"  
						assert torch.allclose(cached_prompt_embeds["pooled_prompt_embeds"].cpu().float(), pooled_prompt_embeds.cpu().float(), atol=1e-3), f"Cached pooled prompt embeds for prompt {prompt} do not match the computed pooled prompt embeds. Make sure that the cached pooled prompt embeds are correct., {torch.mean(torch.abs(cached_prompt_embeds['pooled_prompt_embeds'].cpu().float() - pooled_prompt_embeds.cpu().float())) = }" 
						print(f"fucking passed the test!")
				else: 
					assert "prompt_embeds" in batch and "pooled_prompt_embeds" in batch, "Make sure that the dataloader returns `prompt_embeds` and `pooled_prompt_embeds` when `inference_embeds_dir` is not None." 
					prompt_embeds = batch["prompt_embeds"] 
					pooled_prompt_embeds = batch["pooled_prompt_embeds"] 
					text_ids = torch.zeros((batch["prompt_embeds"].shape[1], 3)) 
					prompt_embeds = prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
					pooled_prompt_embeds = pooled_prompt_embeds.to(dtype=vae.dtype, device=accelerator.device)
					text_ids = text_ids.to(dtype=vae.dtype, device=accelerator.device)


				pixel_values = batch["pixel_values"].to(dtype=vae.dtype)
				height_ = 2 * (int(pixel_values.shape[-2]) // vae_scale_factor)
				width_ = 2 * (int(pixel_values.shape[-1]) // vae_scale_factor)

				model_input = vae.encode(pixel_values).latent_dist.sample()
				model_input = (model_input - vae_config_shift_factor) * vae_config_scaling_factor
				model_input = model_input.to(dtype=weight_dtype)

				latent_image_ids, cond_latent_image_ids = resize_position_encoding(
					model_input.shape[0],
					height_,
					width_,
					height_cond,
					width_cond,
					accelerator.device,
					weight_dtype,
				)

				# Sample noise that we'll add to the latents
				noise = torch.randn_like(model_input)
				bsz = model_input.shape[0]

				# Sample a random timestep for each image
				# for weighting schemes where we sample timesteps non-uniformly
				u = compute_density_for_timestep_sampling(
					weighting_scheme=args.weighting_scheme,
					batch_size=bsz,
					logit_mean=args.logit_mean,
					logit_std=args.logit_std,
					mode_scale=args.mode_scale,
				)
				indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
				timesteps = noise_scheduler_copy.timesteps[indices].to(device=model_input.device)

				# Add noise according to flow matching.
				# zt = (1 - texp) * x + texp * z1
				sigmas = get_sigmas(timesteps, n_dim=model_input.ndim, dtype=model_input.dtype)
				noisy_model_input = (1.0 - sigmas) * model_input + sigmas * noise

				packed_noisy_model_input = FluxPipeline._pack_latents(
					noisy_model_input,
					batch_size=model_input.shape[0],
					num_channels_latents=model_input.shape[1],
					height=model_input.shape[2],
					width=model_input.shape[3],
				)
				
				latent_image_ids_to_concat = [latent_image_ids]
				packed_cond_model_input_to_concat = []
				
				if args.spatial_column is not None:
					# in case the condition is spatial 
					cond_pixel_values = batch["cond_pixel_values"].to(dtype=vae.dtype)             
					cond_input = vae.encode(cond_pixel_values).latent_dist.sample()
					cond_input = (cond_input - vae_config_shift_factor) * vae_config_scaling_factor
					cond_input = cond_input.to(dtype=weight_dtype)
					# number of conditions in the concatenated condition image 
					cond_number = cond_pixel_values.shape[-2] // args.cond_size
					cond_latent_image_ids = torch.concat([cond_latent_image_ids for _ in range(cond_number)], dim=-2)
					latent_image_ids_to_concat.append(cond_latent_image_ids)

					packed_cond_model_input = FluxPipeline._pack_latents(
						cond_input,
						batch_size=cond_input.shape[0],
						num_channels_latents=cond_input.shape[1],
						height=cond_input.shape[2],
						width=cond_input.shape[3],
					)
					packed_cond_model_input_to_concat.append(packed_cond_model_input)
				else: 
					cond_input = None 
					
				latent_image_ids = torch.concat(latent_image_ids_to_concat, dim=-2)
				cond_packed_noisy_model_input = torch.concat(packed_cond_model_input_to_concat, dim=-2)

				# handle guidance
				if accelerator.unwrap_model(transformer).config.guidance_embeds:
					guidance = torch.tensor([args.guidance_scale], device=accelerator.device)
					guidance = guidance.expand(model_input.shape[0])
				else:
					guidance = None

				# Visualize training data before transformer forward pass
				if accelerator.is_main_process and args.debug and num_training_visualizations > 0 and global_step % 5 == 0: 
					visualize_training_data(
						batch=batch,
						vae=vae, 
						model_input=model_input,
						noisy_model_input=noisy_model_input,
						cond_input=cond_input,
						args=args,
						global_step=global_step,
						accelerator=accelerator
					)
					num_training_visualizations -= 1 

				# Predict the noise residual
				model_pred = transformer(
					hidden_states=packed_noisy_model_input,
					# YiYi notes: divide it by 1000 for now because we scale it by 1000 in the transforme rmodel (we should not keep it but I want to keep the inputs same for the model for testing)
					cond_hidden_states=cond_packed_noisy_model_input,
					timestep=timesteps / 1000,
					guidance=guidance,
					pooled_projections=pooled_prompt_embeds,
					encoder_hidden_states=prompt_embeds,
					txt_ids=text_ids,
					img_ids=latent_image_ids,
					return_dict=False,
					call_ids=batch["call_ids"], 
					cuboids_segmasks=batch["cuboids_segmasks"], 
				)[0]

				model_pred = FluxPipeline._unpack_latents(
					model_pred,
					height=int(pixel_values.shape[-2]),
					width=int(pixel_values.shape[-1]),
					vae_scale_factor=vae_scale_factor,
				)

				# these weighting schemes use a uniform timestep sampling
				# and instead post-weight the loss
				weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

				# flow matching loss
				target = noise - model_input

				# Compute regular loss.
				loss = torch.mean(
					(weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
					1,
				)

				loss = loss.mean()
				accelerator.backward(loss)
				if accelerator.sync_gradients:
					params_to_clip = (transformer.parameters())
					accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)

				optimizer.step()
				lr_scheduler.step()
				optimizer.zero_grad()

			# Checks if the accelerator has performed an optimization step behind the scenes
			if accelerator.sync_gradients:
				progress_bar.update(1)
				global_step += 1

				if accelerator.is_main_process:
					if global_step % args.checkpointing_steps == 0:
						# _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
						save_path = os.path.join(args.output_dir, f"epoch-{epoch}__checkpoint-{global_step}")
						os.makedirs(save_path, exist_ok=True)
						unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict()
						lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
						save_file(
							lora_state_dict,
							os.path.join(save_path, "lora.safetensors")
						)
						accelerator.save_state(save_path) 
						os.remove(osp.join(save_path, "model.safetensors")) 
						logger.info(f"Saved state to {save_path}")

			logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
			progress_bar.set_postfix(**logs)
			accelerator.log(logs, step=global_step)

		save_path = os.path.join(args.output_dir, f"epoch-{epoch}__checkpoint-{global_step}")
		os.makedirs(save_path, exist_ok=True)
		unwrapped_model_state = accelerator.unwrap_model(transformer).state_dict()
		lora_state_dict = {k:unwrapped_model_state[k] for k in unwrapped_model_state.keys() if '_lora' in k}
		save_file(
			lora_state_dict,
			os.path.join(save_path, "lora.safetensors")
		)
		accelerator.save_state(save_path) 
		os.remove(osp.join(save_path, "model.safetensors")) 
		logger.info(f"Saved state to {save_path}")
		accelerator.wait_for_everyone()
	accelerator.end_training()


if __name__ == "__main__":
	args = parse_args()
	main(args)