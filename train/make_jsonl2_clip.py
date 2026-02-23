import os
import os.path as osp
import json
from tqdm import tqdm 
import torch
import pickle
from group_subjects_tabletop import groups, groups_prompts
from train import * 

PRETRAINED_MODEL_NAME_OR_PATH = "black-forest-labs/FLUX.1-dev" 

def load_clip_evaluation_results(eval_dir, subjects_comb, img_idx, img_name):
	"""
	Load the CLIP similarity results for a specific image.
	
	Args:
		eval_dir: Base directory containing CLIP evaluation results
		subjects_comb: Subject combination directory name
		img_idx: Image index directory name
		img_name: Name of the image file
	
	Returns:
		Minimum CLIP similarity score, or None if file not found
	"""
	# Construct the path to the pkl file
	pkl_filename = img_name.replace(".jpg", ".pkl")
	pkl_path = osp.join(eval_dir, subjects_comb, img_idx, pkl_filename)
	
	if not osp.exists(pkl_path):
		return None
	
	try:
		with open(pkl_path, 'rb') as f:
			data = pickle.load(f)
		
		similarities = data.get('similarities', [])
		
		# Filter out any potential zero scores for non-existent subjects
		valid_similarities = [s for s in similarities if s > 0.0]
		
		if valid_similarities:
			return min(valid_similarities)
		else:
			return 0.0
			
	except Exception as e:
		print(f"Warning: Could not process file {pkl_path}. Error: {e}")
		return None


def get_call_ids_from_placeholder_prompt_flux(prompt: str, tokenizer_three, subjects, subjects_embeds: list, debug: bool):  
	assert prompt.find("<placeholder>") != -1, "Prompt must contain <placeholder> to get call ids" 

	# the placeholder token ID for all the tokenizers 
	placeholder_token_three = tokenizer_three.encode("<placeholder>", return_tensors="pt")[0][:-1].item()  
	prompt_tokens_three = tokenizer_three.encode(prompt, return_tensors="pt")[0].tolist()  

	placeholder_token_locations_three = [i for i, w in enumerate(prompt_tokens_three) if w == placeholder_token_three] 
	prompt = prompt.replace("<placeholder> ", "") 


	call_ids = [] 
	for subject_idx, (subject, subject_embed) in enumerate(zip(subjects, subjects_embeds)):  
		subject_prompt_ids_t5 = subject_embed["input_ids_t5"][:-1] # T5 has SOT token only  
		num_t5_tokens_subject = len(subject_prompt_ids_t5) 

		t5_call_ids_subject = [i + placeholder_token_locations_three[subject_idx] - 2 * subject_idx - 1 for i in range(num_t5_tokens_subject)] 
		call_ids.append(t5_call_ids_subject) 

		prompt_wo_placeholder = prompt.replace("<placeholder> ", "") 
		t5_call_strs = tokenizer_three.batch_decode(tokenizer_three.encode(prompt_wo_placeholder, return_tensors="pt")[0].tolist())  
		t5_call_strs = [t5_call_strs[i] for i in t5_call_ids_subject] 
		if debug: 
			print(f"{prompt = }, t5 CALL strs for {subject} = {t5_call_strs}") 

	return call_ids  


def generate_cuboids_jsonl(data_dir, output_path, subject_names_embeds_flux, tokenizer_one, tokenizer_two, 
						   clip_eval_dir=None, min_clip_similarity=0.26):
	"""
	Generate a JSONL file for cuboids dataset similar to pose.jsonl format.
	
	Args:
		data_dir: Path to the images directory (same as BlenderFLUXSyntheticDataset data_dir)
		output_path: Path where the cuboids.jsonl file should be saved
		clip_eval_dir: Directory containing CLIP evaluation results (optional)
		min_clip_similarity: Minimum CLIP similarity threshold for depth_flux images (default: 0.26)
	"""
	
	# Create inverse groups mapping
	inverse_groups = {}
	for category in groups:
		for subject in groups[category]:
			assert subject not in inverse_groups
			inverse_groups[subject] = category
	
	jsonl_entries = []
	filtered_count = 0
	total_depth_flux = 0

	imgs_dir = osp.join(data_dir, "main_imgs") 
	cuboids_dir = osp.join(data_dir, "cuboids_monochrome") 
	
	# Iterate over the dataset structure (same as BlenderFLUXSyntheticDataset)
	subjects_combs = os.listdir(imgs_dir) 
	import random 
	random.shuffle(subjects_combs) 
	for subjects_comb in tqdm(subjects_combs):
		if len(subjects_comb.split("__")) > 4: 
			continue 
		if subjects_comb.startswith("_"): 
			continue 
		subjects_ = subjects_comb.split("__")
		subjects = [" ".join(subject_.split("_")) for subject_ in subjects_]
		if "bed" in subjects: 
			continue 
		
		subjects_groups = [inverse_groups[subject] for subject in subjects]
		PROMPTS = groups_prompts[subjects_groups[-2]]
		
		subjects_comb_dir = osp.join(imgs_dir, subjects_comb)

		assert clip_eval_dir is not None 

		for img_idx in os.listdir(subjects_comb_dir):
			if not osp.isdir(osp.join(subjects_comb_dir, img_idx)): 
				continue 

			img_idx_dir = osp.join(subjects_comb_dir, img_idx)
			
			# Check if required files exist
			main_img_path = osp.join(img_idx_dir, "main.jpg")
			cuboids_path = osp.join(cuboids_dir, subjects_comb, img_idx, "cuboids.jpg")
			pkl_path = osp.join(img_idx_dir, "main.pkl")
			
			assert osp.exists(main_img_path), f"Main image path {main_img_path} does not exist"  
			assert osp.exists(cuboids_path), f"Cuboids path {cuboids_path} does not exist"  
			assert osp.exists(pkl_path), f"PKL path {pkl_path} does not exist" 

			# Get all image types (depth_flux and rendering)
			img_names = os.listdir(img_idx_dir)
			
			# Process depth_flux images (prompt*.jpg)
			depth_flux_imgs = [img_name for img_name in img_names 
							 if img_name.endswith(".jpg") and img_name.find("prompt") != -1 and img_name.find("DEBUG") == -1]
			
			# Filter depth_flux images based on CLIP similarity if eval_dir is provided
			if clip_eval_dir is not None:
				filtered_depth_flux_imgs = []
				for img_name in depth_flux_imgs:
					total_depth_flux += 1
					min_similarity = load_clip_evaluation_results(clip_eval_dir, subjects_comb, img_idx, img_name)
					
					if min_similarity is not None and min_similarity >= min_clip_similarity:
						filtered_depth_flux_imgs.append(img_name)
					else:
						filtered_count += 1
				
				depth_flux_imgs = filtered_depth_flux_imgs
			
			all_imgs = depth_flux_imgs + ["main.jpg"] 
			# all_imgs = ["main.jpg"] 
			
			for img_name in all_imgs:
				img_path = osp.join(img_idx_dir, img_name)
				
				if img_name != "main.jpg":  
					# Extract prompt index and get corresponding prompt
					prompt_idx = int(img_name.replace("prompt", "").replace(".jpg", ""))
					print(f"{prompt_idx = }, {subjects_groups[-1] = }, {subjects_comb = }")
					prompt = PROMPTS[prompt_idx]
				else: 
					prompt = "a photo of PLACEHOLDER" 
				
				# Create placeholder text
				placeholder_text = ""
				for subject in subjects[:-1]:
					placeholder_text = placeholder_text + f"<placeholder> {subject} and "
				for subject in subjects[-1:]:
					placeholder_text = placeholder_text + f"<placeholder> {subject}"
				placeholder_text = placeholder_text.strip()

				subjects_embeds = [] 
				cuboids_segmasks_paths = [] 
				segmasks_dir = osp.join(data_dir, "cuboids_segmasks_cv", subjects_comb, img_idx) 
				assert osp.exists(segmasks_dir) 
				segmask_names = sorted(os.listdir(segmasks_dir))  
				for subject_idx, subject in enumerate(subjects):
					subject_embed_path = osp.join(subject_names_embeds_flux, f"{subject.replace(' ', '_')}.pth")
					assert osp.exists(subject_embed_path), f"Subject embed path {subject_embed_path} does not exist" 
					subject_embed_obj = torch.load(subject_embed_path) 
					subjects_embeds.append(subject_embed_obj) 
					cuboid_segmask_path = osp.join(data_dir, "cuboids_segmasks_cv", subjects_comb, img_idx, segmask_names[subject_idx])
					cuboid_segmask_path = osp.relpath(cuboid_segmask_path, osp.dirname(output_path)) 
					# assert osp.exists(cuboid_segmask_path), f"Cuboid segmask path {cuboid_segmask_path} does not exist"
					cuboids_segmasks_paths.append(cuboid_segmask_path) 
				placeholder_prompt = prompt 
				prompt = prompt.replace("PLACEHOLDER", placeholder_text) 
				call_ids = get_call_ids_from_placeholder_prompt_flux(prompt, tokenizer_two, subjects, subjects_embeds, debug=True)
				print(f"{call_ids = }")
				
				# Create relative paths from the output jsonl location
				rel_cuboids_path = osp.relpath(cuboids_path, osp.dirname(output_path))
				rel_img_path = osp.relpath(img_path, osp.dirname(output_path))

				# Create JSONL entry
				entry = {
					"cv": rel_cuboids_path,
					"PLACEHOLDER_prompts": placeholder_prompt,  
					"target": rel_img_path,
					"subjects": subjects, 
					"cuboids_segmasks": cuboids_segmasks_paths, 
					"call_ids": call_ids, 
				}
				jsonl_entries.append(entry)
	
	# Print filtering statistics
	if clip_eval_dir is not None:
		print(f"\n--- Filtering Statistics ---")
		print(f"Total depth_flux images evaluated: {total_depth_flux}")
		print(f"Images filtered out (min similarity < {min_clip_similarity}): {filtered_count}")
		print(f"Images retained: {total_depth_flux - filtered_count}")
		print(f"Retention rate: {((total_depth_flux - filtered_count) / total_depth_flux * 100):.2f}%")
		print("---------------------------\n")
	
	# Write JSONL file
	os.makedirs(osp.dirname(output_path), exist_ok=True)
	with open(output_path, 'w') as f:
		for entry in jsonl_entries:
			f.write(json.dumps(entry) + '\n')
	
	print(f"Generated {len(jsonl_entries)} entries in {output_path}")

if __name__ == "__main__":
	# Configuration
	data_dir = "/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv9"  # Replace with actual imgs_dir path
	output_path = "/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv9/cuboids_monochrome.jsonl"
	subjects_embeds_path = "/archive/vaibhav.agrawal/a-bev-of-the-latents/subject_names_embeds_flux"  # Path to subject embeddings JSON 
	clip_eval_dir = "/archive/vaibhav.agrawal/a-bev-of-the-latents/clip_evaluation__datasetv9"  # CLIP evaluation results directory
	min_clip_similarity = 0.26  # Minimum CLIP similarity threshold
	rendered_imgs_prompt = "An image of PLACEHOLDER"  # Customize as needed
	
	# You can also accept command line arguments
	import sys
	if len(sys.argv) >= 2:
		imgs_dir = sys.argv[1]
	if len(sys.argv) >= 3:
		output_path = sys.argv[2]
	if len(sys.argv) >= 4:
		rendered_imgs_prompt = sys.argv[3]

	tokenizer_one = CLIPTokenizer.from_pretrained(
		PRETRAINED_MODEL_NAME_OR_PATH, 
		subfolder="tokenizer",
		revision=None, 
	)
	tokenizer_two = T5TokenizerFast.from_pretrained(
		PRETRAINED_MODEL_NAME_OR_PATH, 
		subfolder="tokenizer_2",
		revision=None, 
	)

	placeholder_token_str = ["<placeholder>"]
	num_added_tokens = tokenizer_one.add_tokens(placeholder_token_str)   
	assert num_added_tokens == 1 
	num_added_tokens = tokenizer_two.add_tokens(placeholder_token_str)   
	assert num_added_tokens == 1 
	
	generate_cuboids_jsonl(data_dir, output_path, subjects_embeds_path, tokenizer_two, tokenizer_two,
						   clip_eval_dir=clip_eval_dir, min_clip_similarity=min_clip_similarity)