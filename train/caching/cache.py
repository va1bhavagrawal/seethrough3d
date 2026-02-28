import os
import os.path as osp
import json
from PIL import Image 
from tqdm import tqdm 
import torch 
import random 
from train import * 

pretrained_path = "black-forest-labs/FLUX.1-dev" 
OUTPUT_DIR = "/archive/vaibhav.agrawal/a-bev-of-the-latents/inference_embeds" 
DATASET_JSONL = "/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv7_superhard/cuboids__234subjects.jsonl"


def load_text_encoders(class_one, class_two):
    text_encoder_one = class_one.from_pretrained(
        pretrained_path, subfolder="text_encoder", revision=None, variant=None
    )
    text_encoder_two = class_two.from_pretrained(
        pretrained_path, subfolder="text_encoder_2", revision=None, variant=None
    )
    return text_encoder_one, text_encoder_two


def load_prompts_from_jsonl(DATASET_JSONL):
	"""
	Load unique prompts from JSONL file with subjects filled in.
	
	Args:
		DATASET_JSONL: Path to the JSONL file
	
	Returns:
		List of unique prompts with subjects filled in
	"""
	unique_prompts = set()
	
	print(f"Reading prompts from {DATASET_JSONL}...")
	with open(DATASET_JSONL, 'r') as f:
		for line in f:
			entry = json.loads(line)
			
			# Get the placeholder prompt and subjects
			placeholder_prompt = entry.get("PLACEHOLDER_prompts", "")
			subjects = entry.get("subjects", [])
			
			if not placeholder_prompt or not subjects:
				continue
			
			# Create placeholder text from subjects
			placeholder_text = ""
			for subject_idx, subject in enumerate(subjects):
				if subject_idx == 0:
					placeholder_text = placeholder_text + f"{subject}"
				else:
					placeholder_text = placeholder_text + f" and {subject}"
			
			# Replace PLACEHOLDER with actual subjects
			filled_prompt = placeholder_prompt.replace("PLACEHOLDER", placeholder_text)
			unique_prompts.add(filled_prompt)
	
	prompts_list = list(unique_prompts)
	print(f"Found {len(prompts_list)} unique prompts after filling subjects.")
	
	return prompts_list


if __name__ == "__main__": 
	# import correct text encoder classes
	with torch.no_grad(): 
		accelerator = Accelerator()
		text_encoder_cls_one = import_model_class_from_model_name_or_path(
			pretrained_path, revision=None,  
		)
		text_encoder_cls_two = import_model_class_from_model_name_or_path(
			pretrained_path, subfolder="text_encoder_2", revision=None, 
		)

		text_encoder_one, text_encoder_two = load_text_encoders(
			text_encoder_cls_one, text_encoder_cls_two 
		)
		text_encoder_one = text_encoder_one.to(accelerator.device) 
		text_encoder_two = text_encoder_two.to(accelerator.device) 

		tokenizer_one = CLIPTokenizer.from_pretrained(
			pretrained_path, 
			subfolder="tokenizer",
			revision=None, 
		)
		tokenizer_two = T5TokenizerFast.from_pretrained(
			pretrained_path, 
			subfolder="tokenizer_2",
			revision=None, 
		)

		
		# Load unique prompts from JSONL
		all_prompts = load_prompts_from_jsonl(DATASET_JSONL)
		
		# Add empty string and space for negative embeds
		all_prompts.extend(["", " "])
		
		print(f"Total prompts to cache (including negative embeds): {len(all_prompts)}")

		random.seed() 
		random.shuffle(all_prompts) # if this is run on multiple processes, then random shuffling will reduce the chances of multiple processes trying to cache the same prompt at the same time  
		for prompt in all_prompts: 
			if prompt == "": 
				latents_path = osp.join(OUTPUT_DIR, "negative_embeds.pth") 
			elif prompt == " ": 
				latents_path = osp.join(OUTPUT_DIR, "space_prompt.pth") 
			else: 
				latents_path = osp.join(OUTPUT_DIR, f"{'_'.join(prompt.split())}.pth")   
			latents_path = latents_path.replace("____", "__") 
			if osp.exists(latents_path): 
				print(f"Embeds for {prompt = } already cached at {latents_path}, skipping...") 
				continue 
			print(f"doing {prompt = }")
			prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
				text_encoders=[text_encoder_one, text_encoder_two],
				tokenizers=[tokenizer_one, tokenizer_two], 
				prompt=[prompt],
				max_sequence_length=512, 
				device=accelerator.device
			) 
			assert torch.allclose(text_ids, torch.zeros_like(text_ids)), f"{text_ids = }" 

			os.makedirs(osp.dirname(latents_path), exist_ok=True) 
			embeds = {
				"prompt": prompt, 
				"prompt_embeds": prompt_embeds, 
				"pooled_prompt_embeds": pooled_prompt_embeds, 
			}
			torch.save(embeds, latents_path)         
		accelerator.wait_for_everyone()
		print(f"\nCaching complete! Embeddings saved to {OUTPUT_DIR}")