import os
import os.path as osp 
import sys
import numpy as np
import tempfile
import shutil 
import base64
import io
from PIL import Image
import gradio as gr
import time
import copy
import requests
import json
import pickle 
from concurrent.futures import ThreadPoolExecutor, as_completed
from object_scales import scales  
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
import pickle 
from datetime import datetime 
from infer_backend import initialize_inference_engine, run_inference_from_gradio

COLORS = [
    (1.0, 0.0, 0.0),    # Red
    (0.0, 0.8, 0.2),    # Green
    (0.0, 0.0, 1.0),    # Blue
    (1.0, 1.0, 0.0),    # Yellow
    (0.0, 1.0, 1.0),    # Cyan
    (1.0, 0.0, 1.0),    # Magenta
    (1.0, 0.6, 0.0),    # Orange
    (0.6, 0.0, 0.8),    # Purple
    (0.0, 0.4, 0.0),    # Dark Green
    (0.8, 0.8, 0.8),    # Light Gray
    (0.2, 0.2, 0.2)     # Dark Gray
]

CHECKPOINT_NAMES = [
    "rgb__r1/epoch-0__checkpoint-25917",
    "rgb__finetune_1024/epoch-0__checkpoint-3000", 
    "rgb__finetune_1024/epoch-1__checkpoint-4000", 
    "rgb__finetune_1024/epoch-1__checkpoint-5000", 
    "rgb__finetune_1024/epoch-1__checkpoint-6000", 
    "rgb__finetune_1024/epoch-1__checkpoint-7000", 
    "rgb__finetune_1024/epoch-1__checkpoint-7932", 
]

PRETRAINED_MODEL_NAME_OR_PATH = "black-forest-labs/FLUX.1-dev" 

tokenizer = T5TokenizerFast.from_pretrained(
    PRETRAINED_MODEL_NAME_OR_PATH, 
    subfolder="tokenizer_2",
    revision=None, 
)

placeholder_token_str = ["<placeholder>"]
num_added_tokens = tokenizer.add_tokens(placeholder_token_str)   
assert num_added_tokens == 1 

def generate_image_event(camera_elevation, camera_lens, surrounding_prompt, checkpoint_name, 
                        height, width, seed, guidance_scale, num_steps):
    """Generate final image with segmentation masks and run inference"""
    # Update scene manager's inference params before generation
    scene_manager.update_inference_params(height, width, seed, guidance_scale, num_steps, checkpoint_name)
    if not scene_manager.objects:
        return (
            "⚠️ No objects to render",
            gr.update(),
            Image.new('RGB', (512, 512), color='white')
        )
    
    # Get subject descriptions
    subject_descriptions = [obj['description'] for obj in scene_manager.objects]
    
    print(f"Surrounding prompt: {surrounding_prompt}")
    print(f"Subject descriptions: {subject_descriptions}")
    print(f"Selected checkpoint: {checkpoint_name}")

    placeholder_prompt = "a photo of PLACEHOLDER " + surrounding_prompt 

    # Create placeholder text
    subject_embeds = [] 
    for subject_idx, subject_desc in enumerate(subject_descriptions): 
        input_ids = tokenizer.encode(subject_desc, return_tensors="pt", max_length=77)[0] 
        subject_embed = {"input_ids_t5": input_ids.tolist()} 
        subject_embeds.append(subject_embed)

    placeholder_text = ""
    for subject in subject_descriptions[:-1]:
        placeholder_text = placeholder_text + f"<placeholder> {subject} and "
    for subject in subject_descriptions[-1:]:
        placeholder_text = placeholder_text + f"<placeholder> {subject}"
    placeholder_text = placeholder_text.strip()

    placeholder_token_prompt = placeholder_prompt.replace("PLACEHOLDER", placeholder_text) 

    call_ids = get_call_ids_from_placeholder_prompt_flux(prompt=placeholder_token_prompt, 
                                             subjects=subject_descriptions, 
                                             subjects_embeds=subject_embeds,
                                             debug=True
                                            ) 
    print(f"Generated call IDs: {call_ids}") 

    # Convert to server expected format
    subjects_data, camera_data = scene_manager._convert_to_blender_format()
    
    # Render final high-quality image using CYCLES (port 5002)
    final_img = scene_manager.render_client._send_render_request(
        scene_manager.render_client.final_server_url, 
        subjects_data, 
        camera_data
    )

    final_img.save("model_condition.jpg") 
    
    # Render segmentation masks
    success, segmask_images, error_msg = scene_manager.render_client.render_segmasks(subjects_data, camera_data)
    
    if not success:
        return (
            f"❌ Failed to render segmentation masks: {error_msg}",
            gr.update(),
            Image.new('RGB', (512, 512), color='white')
        )

    # Save all files to the correct location  
    root_save_dir = "/archive/vaibhav.agrawal/a-bev-of-the-latents/gradio_files/" 
    os.system(f"rm -f {root_save_dir}/*") 
    
    # Save final render to root directory
    final_render_path = osp.join(root_save_dir, "cv_render.jpg")
    final_img.save(final_render_path)
    
    # Move segmentation masks
    for subject_idx in range(len(subject_descriptions)): 
        shutil.move(
            f"{str(subject_idx).zfill(3)}_segmask_cv.png", 
            osp.join(root_save_dir, f"main__segmask_{str(subject_idx).zfill(3)}__{1.00}.png")
        ) 
    
    # Create JSONL
    jsonl = [{
        "cv": final_render_path,
        "target": final_render_path,
        "cuboids_segmasks": [
            osp.join(root_save_dir, f"main__segmask_{str(subject_idx).zfill(3)}__{1.00}.png") 
            for subject_idx in range(len(subject_descriptions))
        ],
        "PLACEHOLDER_prompts": placeholder_prompt, 
        "subjects": subject_descriptions, 
        "call_ids": call_ids, 
    }] 

    jsonl_path = osp.join(root_save_dir, "cuboids.jsonl")
    with open(jsonl_path, "w") as f: 
        json.dump(jsonl[0], f)
    
    # Run inference using the pre-loaded model
    print(f"\n{'='*60}")
    print(f"RUNNING INFERENCE")
    print(f"{'='*60}\n")
    
    inference_success, generated_image, inference_msg = run_inference_from_gradio(
        checkpoint_name=checkpoint_name,
        height=height,
        width=width,
        seed=seed,
        guidance_scale=guidance_scale,
        num_inference_steps=num_steps,
        jsonl_path=jsonl_path
    )
    
    if not inference_success:
        return (
            f"✅ Saved files but inference failed: {inference_msg}",
            final_img,
            Image.new('RGB', (512, 512), color='white')
        )
    
    status_msg = f"✅ Generated image using {checkpoint_name} with {len(segmask_images)} segmentation masks"

    # Render final high-quality image using CYCLES (port 5002)
    final_img = scene_manager.render_client._send_render_request(
        scene_manager.render_client.paper_figure_server_url, 
        subjects_data, 
        camera_data
    )
    
    return (
        status_msg,
        final_img,  # Display CV render in Camera View
        generated_image  # Display generated image in Generated Image section
    )


def get_call_ids_from_placeholder_prompt_flux(prompt: str, subjects, subjects_embeds: list, debug: bool):  
	assert prompt.find("<placeholder>") != -1, "Prompt must contain <placeholder> to get call ids" 

	# the placeholder token ID for all the tokenizers 
	placeholder_token_three = tokenizer.encode("<placeholder>", return_tensors="pt")[0][:-1].item()  
	prompt_tokens_three = tokenizer.encode(prompt, return_tensors="pt")[0].tolist()  

	placeholder_token_locations_three = [i for i, w in enumerate(prompt_tokens_three) if w == placeholder_token_three] 
	prompt = prompt.replace("<placeholder> ", "") 


	call_ids = [] 
	for subject_idx, (subject, subject_embed) in enumerate(zip(subjects, subjects_embeds)):  
		subject_prompt_ids_t5 = subject_embed["input_ids_t5"][:-1] # T5 has SOT token only  
		num_t5_tokens_subject = len(subject_prompt_ids_t5) 

		t5_call_ids_subject = [i + placeholder_token_locations_three[subject_idx] - 2 * subject_idx - 1 for i in range(num_t5_tokens_subject)] 
		call_ids.append(t5_call_ids_subject) 

		prompt_wo_placeholder = prompt.replace("<placeholder> ", "") 
		t5_call_strs = tokenizer.batch_decode(tokenizer.encode(prompt_wo_placeholder, return_tensors="pt")[0].tolist())  
		t5_call_strs = [t5_call_strs[i] for i in t5_call_ids_subject] 
		if debug: 
			print(f"{prompt = }, t5 CALL strs for {subject} = {t5_call_strs}") 

	return call_ids  


def map_point_to_rgb(x, y):
    """
    Map (x, y) inside the frustum to an RGB color with continuity and variation.
    """
    # Frustum boundaries
    X_MIN, X_MAX = -10.0, -1.0
    Y_MIN_AT_XMIN, Y_MAX_AT_XMIN = -4.5, 4.5
    Y_MIN_AT_XMAX, Y_MAX_AT_XMAX = -0.5, 0.5
    
    # Normalize x to [0, 1]
    x_norm = (x - X_MIN) / (X_MAX - X_MIN)
    # x_norm = np.clip(x_norm, 0, 1)

    # Compute current Y bounds at given x using linear interpolation
    y_min = Y_MIN_AT_XMIN + x_norm * (Y_MIN_AT_XMAX - Y_MIN_AT_XMIN)
    y_max = Y_MAX_AT_XMIN + x_norm * (Y_MAX_AT_XMAX - Y_MAX_AT_XMIN)

    # Normalize y to [0, 1] within current bounds
    if y_max != y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = 0.5
    y_norm = np.clip(y_norm, 0.0, 1.0)

    # Color mapping: more variation along x
    r = x_norm
    g = y_norm 
    b = 1.0 - x_norm

    return (r, g, b)

def rgb_to_hex(rgb_tuple):
    """Convert RGB tuple (0-1 range) to hex color string."""
    r, g, b = rgb_tuple
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"


class BlenderRenderClient:
    def __init__(self, cv_server_url="http://127.0.0.1:5001", segmask_server_url="http://127.0.0.1:5003", final_server_url="http://127.0.0.1:5002", paper_figure_server_url="http://127.0.0.1:5004"): 
        """
        Initialize the Blender render client.
        
        Args:
            cv_server_url (str): URL of the camera view render server
            segmask_server_url (str): URL of the segmentation mask render server
        """
        self.cv_server_url = cv_server_url
        self.segmask_server_url = segmask_server_url
        self.final_server_url = final_server_url 
        self.paper_figure_server_url = paper_figure_server_url 
        self.timeout = 30  # 30 second timeout for renders

    def render_segmasks(self, subjects_data: list, camera_data: dict) -> tuple:
        """
        Send a segmentation mask render request. 
        Returns (success: bool, segmask_images: list of PIL Images or None, error_message: str or None)
        """
        try:
            request_data = {
                "subjects_data": subjects_data,
                "camera_data": camera_data,
                "num_samples": 1
            }
            
            response = requests.post(
                f"{self.segmask_server_url}/render_segmasks",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    # Decode all segmentation masks
                    segmask_images = []
                    for img_base64 in result["segmasks_base64"]:
                        img_data = base64.b64decode(img_base64)
                        img = Image.open(io.BytesIO(img_data))
                        segmask_images.append(img)
                    
                    print(f"Successfully rendered {len(segmask_images)} segmentation masks")
                    return True, segmask_images, None
                else:
                    error_msg = result.get('error_message', 'Unknown error')
                    print(f"Segmask render failed: {error_msg}")
                    return False, None, error_msg
            else:
                error_msg = f"HTTP error {response.status_code}: {response.text}"
                print(error_msg)
                return False, None, error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "Segmask render request timed out"
            print(error_msg)
            return False, None, error_msg
        except Exception as e:
            error_msg = f"Segmask render request failed: {e}"
            print(error_msg)
            return False, None, error_msg

        
    def _send_render_request(self, server_url: str, subjects_data: list, camera_data: dict) -> Image.Image:
        """Send a render request to a server and return the image."""
        try:
            request_data = {
                "subjects_data": subjects_data,
                "camera_data": camera_data,
                "num_samples": 1
            }
            print(f"passing {subjects_data = } to server at {server_url}") 
            
            response = requests.post(
                f"{server_url}/render",
                json=request_data,
                timeout=self.timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                if result["success"]:
                    # Decode base64 image
                    img_data = base64.b64decode(result["image_base64"])
                    img = Image.open(io.BytesIO(img_data))
                    return img
                else:
                    print(f"Render failed: {result.get('error_message', 'Unknown error')}")
                    return self._create_error_image("red")
            else:
                print(f"HTTP error {response.status_code}: {response.text}")
                return self._create_error_image("orange")
                
        except requests.exceptions.Timeout:
            print("Render request timed out")
            return self._create_error_image("yellow")
        except Exception as e:
            print(f"Render request failed: {e}")
            return self._create_error_image("red")
    
    def _create_error_image(self, color: str) -> Image.Image:
        """Create a colored error image."""
        return Image.new('RGB', (512, 512), color=color)
    
# --- Scene Management Class ---
class SceneManager:
    def __init__(self):
        self.objects = []
        self.camera_elevation = 30.0
        self.camera_lens = 50.0
        self.surrounding_prompt = ""
        self.next_color_idx = 0
        self.colors = [
            (1.0, 0.0, 0.0),  # red
            (0.0, 0.0, 1.0),  # blue
            (0.0, 1.0, 0.0),  # green
            (0.5, 0.0, 0.5),  # purple
            (1.0, 0.5, 0.0),  # orange
            (1.0, 1.0, 0.0),  # yellow
            (0.0, 1.0, 1.0),  # cyan
            (1.0, 0.0, 1.0),  # magenta
        ]
        
        # Add inference parameters with defaults
        self.inference_params = {
            'height': 512,
            'width': 512,
            'seed': 42,
            'guidance_scale': 3.5,
            'num_inference_steps': 25,
            'checkpoint': CHECKPOINT_NAMES[0] if CHECKPOINT_NAMES else None
        }
        
        # Initialize BlenderRenderClient
        self.render_client = BlenderRenderClient()
        
        # Load asset dimensions
        self.asset_dimensions = self._load_asset_dimensions()


    def update_inference_params(self, height, width, seed, guidance_scale, num_steps, checkpoint):
        """Update inference parameters"""
        self.inference_params = {
            'height': height,
            'width': width,
            'seed': seed,
            'guidance_scale': guidance_scale,
            'num_inference_steps': num_steps,
            'checkpoint': checkpoint
        }


    def update_cuboid_description(self, obj_id, new_description):
        """Update the description of a cuboid"""
        if 0 <= obj_id < len(self.objects):
            if new_description.strip():  # Check not empty
                self.objects[obj_id]['description'] = new_description.strip()
                return True
        return False


    def save_scene_to_pkl(self, filepath=None):
        """Save current scene data to pkl file including inference parameters"""
        if filepath is None:
            # Auto-generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"scene_{timestamp}.pkl"
        
        # Convert to the expected format
        subjects_data = []
        for obj in self.objects:
            subject_dict = {
                'name': obj['description'],
                'type': obj['type'],  # Save the object type
                'dims': tuple(obj['size']),  # (width, depth, height)
                'x': [obj['position'][0] - 6.0],
                'y': [obj['position'][1]],
                'z': [obj['position'][2]],
                'azimuth': [np.radians(obj['azimuth'])],  # Convert to radians
                'bbox': [(0, 0, 0, 0)]  # Placeholder, can be computed if needed
            }
            subjects_data.append(subject_dict)
        
        camera_data = {
            'camera_elevation': np.radians(self.camera_elevation),
            'lens': self.camera_lens,
            'global_scale': 1.0  # Default value
        }
        
        scene_dict = {
            'subjects_data': subjects_data,
            'camera_data': camera_data,
            'surrounding_prompt': self.surrounding_prompt,
            'inference_params': self.inference_params.copy()
        }
        
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(scene_dict, f)
            return True, filepath, None
        except Exception as e:
            return False, None, str(e)


    def load_scene_from_pkl(self, filepath):
        """Load scene data from pkl file including inference parameters"""
        try:
            with open(filepath, 'rb') as f:
                scene_dict = pickle.load(f)
            
            # Clear existing objects
            self.objects = []
            self.next_color_idx = 0
            
            # Load subjects
            subjects_data = scene_dict.get('subjects_data', [])
            for subject_dict in subjects_data:
                name = subject_dict.get('name', 'Loaded Object')
                asset_type = subject_dict.get('type', 'Custom')  # Load the type
                dims = subject_dict.get('dims', (1.0, 1.0, 1.0))
                x = float(subject_dict.get('x', [0.0])[0]) + 6.0  
                y = float(subject_dict.get('y', [0.0])[0])
                z = float(subject_dict.get('z', [0.0])[0]) 
                azimuth_rad = float(subject_dict.get('azimuth', [0.0])[0])
                azimuth_deg = np.degrees(azimuth_rad)
                
                # Determine original_asset_size based on type
                if asset_type == "Custom" or asset_type not in self.asset_dimensions:
                    original_asset_size = None
                else:
                    # Look up the original asset dimensions
                    asset_dims = self.asset_dimensions[asset_type]
                    original_asset_size = [float(asset_dims[0]), float(asset_dims[1]), float(asset_dims[2])]
                
                # Create object
                obj_id = len(self.objects)
                size_list = [float(d) for d in dims]
                cuboid = {
                    'id': obj_id,
                    'description': name,
                    'type': asset_type,  # Use the loaded type
                    'position': [x, y, z],
                    'size': size_list,
                    'original_asset_size': original_asset_size,  # Restore from asset_dimensions
                    'azimuth': float(azimuth_deg),
                    'color': self._get_next_color()
                }
                self.objects.append(cuboid)
            
            # Load camera settings
            camera_data = scene_dict.get('camera_data', {})
            camera_elev_rad = float(camera_data.get('camera_elevation', np.radians(30.0)))
            self.camera_elevation = float(np.degrees(camera_elev_rad))
            self.camera_lens = float(camera_data.get('lens', 50.0))
            
            # Load surrounding prompt
            self.surrounding_prompt = scene_dict.get('surrounding_prompt', '')
            
            # Load inference parameters
            loaded_inference_params = scene_dict.get('inference_params', {})
            
            # Get checkpoint, fall back to first available if not found
            saved_checkpoint = loaded_inference_params.get('checkpoint')
            if saved_checkpoint and saved_checkpoint in CHECKPOINT_NAMES:
                checkpoint = saved_checkpoint
            else:
                checkpoint = CHECKPOINT_NAMES[0] if CHECKPOINT_NAMES else None
                if saved_checkpoint:
                    print(f"Warning: Saved checkpoint '{saved_checkpoint}' not found, using '{checkpoint}' instead")
            
            self.inference_params = {
                'height': loaded_inference_params.get('height', 512),
                'width': loaded_inference_params.get('width', 512),
                'seed': loaded_inference_params.get('seed', 42),
                'guidance_scale': loaded_inference_params.get('guidance_scale', 3.5),
                'num_inference_steps': loaded_inference_params.get('num_inference_steps', 25),
                'checkpoint': checkpoint
            }
            
            return True, len(subjects_data), None
        except FileNotFoundError:
            return False, 0, f"File not found: {filepath}"
        except Exception as e:
            return False, 0, f"Error loading file: {str(e)}"


    def _load_asset_dimensions(self):
        """Load asset dimensions from pickle file"""
        pkl_path = "asset_dimensions.pkl"
        if os.path.exists(pkl_path):
            try:
                with open(pkl_path, 'rb') as f:
                    return pickle.load(f)
            except Exception as e:
                print(f"Warning: Could not load asset dimensions: {e}")
                return {}
        else:
            print(f"Warning: asset_dimensions.pkl not found at {pkl_path}")
            return {}

    def get_asset_type_choices(self):
        """Get list of asset types for dropdown"""
        choices = ["Custom"]
        if self.asset_dimensions:
            choices.extend(sorted(self.asset_dimensions.keys()))
        return choices

    def _get_next_color(self):
        color = self.colors[self.next_color_idx % len(self.colors)]
        self.next_color_idx += 1
        return color


    def harmonize_scales(self):
        """
        Harmonize the scales of all non-Custom objects based on object scales.
        Always scales from original asset dimensions, ignoring any manual edits.
        Custom objects remain unchanged.
        """
        if not self.objects:
            return "No objects to harmonize"
        
        # Find objects that can be harmonized (non-Custom with valid scales and original_asset_size)
        harmonizable_objects = []
        for obj in self.objects:
            if (obj['type'] != "Custom" and 
                obj['type'] in scales and 
                obj['original_asset_size'] is not None):
                harmonizable_objects.append(obj)
        
        if not harmonizable_objects:
            return "No objects with defined scales to harmonize (all are Custom)"
        
        # Find the largest scale among harmonizable objects
        max_scale = max(scales[obj['type']] for obj in harmonizable_objects)
        
        if max_scale == 0:
            return "Invalid max scale (0)"
        
        # Harmonize each object by scaling from ORIGINAL ASSET dimensions
        for obj in harmonizable_objects:
            obj_scale = scales[obj['type']]
            scale_factor = obj_scale / max_scale
            
            # Scale from ORIGINAL ASSET dimensions, not current dimensions
            obj['size'][0] = obj['original_asset_size'][0] * scale_factor  # width
            obj['size'][1] = obj['original_asset_size'][1] * scale_factor  # depth
            obj['size'][2] = obj['original_asset_size'][2] * scale_factor  # height
            
            # Update z position to keep object on ground
            obj['position'][2] = 0.0 
        
        return f"Harmonized {len(harmonizable_objects)} objects based on largest scale: {max_scale}"


    def add_cuboid(self, description="New Cuboid", asset_type="Custom"):
        """Add a cuboid with dimensions based on asset type"""
        obj_id = len(self.objects)
        
        # Determine dimensions based on asset type
        if asset_type == "Custom" or asset_type not in self.asset_dimensions:
            size = [1.0, 1.0, 1.0]  # Default size
            original_asset_size = None  # Custom objects have no original asset size
        else:
            # Load dimensions from pkl file
            dims = self.asset_dimensions[asset_type]
            size = [float(dims[0]), float(dims[1]), float(dims[2])]  # [width, depth, height]
            original_asset_size = size.copy()  # Store the original asset dimensions
        
        cuboid = {
            'id': obj_id,
            'description': description,
            'type': asset_type,  # Store the asset type
            'position': [0.0, 0.0, 0.0],  # Place on ground (z = height/2)
            'size': size,
            'original_asset_size': original_asset_size,  # Store original asset dimensions
            'azimuth': 0.0,
            'color': self._get_next_color()
        }
        self.objects.append(cuboid)
        return obj_id


    def update_cuboid(self, obj_id, x, y, z, azimuth, width, depth, height):
        if 0 <= obj_id < len(self.objects):
            obj = self.objects[obj_id]
            obj['position'] = [x, y, z]
            obj['size'] = [width, depth, height]
            # Note: We do NOT update original_asset_size here - it stays unchanged
            obj['azimuth'] = azimuth
            return True
        return False


    def delete_cuboid(self, obj_id):
        if 0 <= obj_id < len(self.objects):
            del self.objects[obj_id]
            # Update IDs for remaining objects
            for i, obj in enumerate(self.objects):
                obj['id'] = i
            return True
        return False
            
    def set_camera_elevation(self, elevation_deg):
        assert type(elevation_deg) == float or type(elevation_deg) == int, f"{type(elevation_deg) = }" 
        self.camera_elevation = np.clip(elevation_deg, 0.0, 90.0)
        return f"Camera elevation set to {elevation_deg}°"

    def set_camera_lens(self, lens_value):
        self.camera_lens = np.clip(lens_value, 10.0, 200.0)
        return f"Camera lens set to {lens_value}mm"

    def set_surrounding_prompt(self, prompt):  # Add this method
        self.surrounding_prompt = prompt
        return f"Surrounding prompt updated"

    def _convert_to_blender_format(self):
        """Convert internal objects format to server expected format"""
        subjects_data = []
        
        for obj in self.objects:
            subject_data = {
                'subject_name': obj['description'],
                'x': float(obj['position'][0]),
                'y': float(obj['position'][1]), 
                'z': float(obj['position'][2]),
                'azimuth': float(obj['azimuth']),
                'width': float(obj['size'][0]),
                'depth': float(obj['size'][1]),
                'height': float(obj['size'][2]),
                'base_color': obj['color']
            }
            subjects_data.append(subject_data)
            
        camera_data = {
            'camera_elevation': float(np.radians(self.camera_elevation)),
            'lens': float(self.camera_lens),
            'global_scale': 1.0
        }
        
        return subjects_data, camera_data

    def render_cv_view(self, subjects_data: list, camera_data: dict) -> Image.Image:
        """Render only the CV view."""
        if not subjects_data:
            return Image.new('RGB', (512, 512), color='gray')
        
        return self.render_client._send_render_request(self.render_client.cv_server_url, subjects_data, camera_data)


    def render_scene(self, width=512, height=512):
        """Render only CV view using the render client."""
        print(f"calling render_scene")
        if not self.objects:
            # Return empty image if no objects
            empty_cv = Image.new('RGB', (width, height), color='gray')
            return empty_cv
        
        # Convert to server expected format
        subjects_data, camera_data = self._convert_to_blender_format()
        print(f"passing {subjects_data = } to render_cv_view in SceneManager") 
        
        # Render CV view only
        cv_img = self.render_cv_view(subjects_data, camera_data)
        
        return cv_img
    
# --- Gradio Interface Logic ---
scene_manager = SceneManager()

def get_cuboid_list_html():
    """Generate HTML for the cuboid list with position-based colors"""
    if not scene_manager.objects:
        return "<div style='text-align: center; padding: 20px; color: #888;'>No cuboids yet. Add one to get started!</div>"
    
    html = "<div style='display: flex; flex-direction: column; gap: 8px;'>"
    for obj_idx, obj in enumerate(scene_manager.objects):
        # Get position-based color
        # x, y = obj['position'][0], obj['position'][1]
        # rgb_color = map_point_to_rgb(x, y)
        rgb_color = COLORS[obj_idx % len(COLORS)]  
        hex_color = rgb_to_hex(rgb_color)
        
        # Create a lighter version for gradient end
        lighter_rgb = tuple(min(1.0, c + 0.2) for c in rgb_color)
        lighter_hex = rgb_to_hex(lighter_rgb)
        
        html += f"""
        <div style='background: linear-gradient(135deg, {hex_color} 0%, {lighter_hex} 100%); 
                    padding: 12px; border-radius: 8px; color: white; text-shadow: 1px 1px 2px rgba(0,0,0,0.5);'>
            <div style='font-weight: bold; font-size: 14px;'>{obj['description']}</div>
            <div style='font-size: 11px; opacity: 0.9; margin-top: 4px;'>
                Pos: ({obj['position'][0]:.1f}, {obj['position'][1]:.1f}, {obj['position'][2]:.1f}) | 
                Size: {obj['size'][0]:.1f}×{obj['size'][1]:.1f}×{obj['size'][2]:.1f}
            </div>
        </div>
        """
    html += "</div>"
    return html


def add_cuboid_event(description_input, asset_type, camera_elevation, camera_lens):
    """Add a new cuboid"""
    if not description_input.strip():
        description_input = "New Cuboid"
    
    new_id = scene_manager.add_cuboid(description_input, asset_type)
    cv_img = scene_manager.render_scene()
    
    # Create choices for radio buttons
    choices = [f"{obj['description']}" for obj in scene_manager.objects]
    
    # Get the new object data
    new_obj = scene_manager.objects[new_id]
    
    return (
        gr.update(value=""),  # Clear description input
        gr.update(value="Custom"),  # Reset type dropdown to Custom
        cv_img,
        get_cuboid_list_html(),
        gr.update(choices=choices, value=new_obj['description']),  # Radio with new selection
        gr.update(visible=True),  # Show editor
        gr.update(value=new_obj['description']),  # Set description in editor
        gr.update(value=new_obj['position'][0]),
        gr.update(value=new_obj['position'][1]),
        gr.update(value=new_obj['position'][2]),
        gr.update(value=new_obj['azimuth']),
        gr.update(value=new_obj['size'][0]),
        gr.update(value=new_obj['size'][1]),
        gr.update(value=new_obj['size'][2]),
        gr.update(value=1.0)  # Reset scale to 1.0
    )


def select_cuboid_event(selected_name):
    """When a cuboid is selected from radio buttons"""
    if not selected_name:
        return [gr.update(visible=False)] + [gr.update() for _ in range(9)]  # Changed from 8 to 9
    
    # Find the cuboid by description
    obj = None
    for o in scene_manager.objects:
        if o['description'] == selected_name:
            obj = o
            break
    
    if obj is None:
        return [gr.update(visible=False)] + [gr.update() for _ in range(9)]
    
    return (
        gr.update(visible=True),  # Show editor
        gr.update(value=obj['description']),  # Set description
        gr.update(value=obj['position'][0]),
        gr.update(value=obj['position'][1]),
        gr.update(value=obj['position'][2]),
        gr.update(value=obj['azimuth']),
        gr.update(value=obj['size'][0]),
        gr.update(value=obj['size'][1]),
        gr.update(value=obj['size'][2]),
        gr.update(value=1.0)  # Reset scale to 1.0
    )


def delete_selected_cuboid(selected_name, camera_elevation, camera_lens):
    """Delete the currently selected cuboid"""
    if not selected_name:
        return gr.update(), get_cuboid_list_html(), gr.update(), gr.update(visible=False)
    
    # Find and delete the cuboid
    obj_id = None
    for i, obj in enumerate(scene_manager.objects):
        if obj['description'] == selected_name:
            obj_id = i
            break
    
    if obj_id is not None:
        scene_manager.delete_cuboid(obj_id)
    
    cv_img = scene_manager.render_scene()
    
    # Update choices
    choices = [f"{obj['description']}" for obj in scene_manager.objects]
    
    return (
        cv_img,
        get_cuboid_list_html(),
        gr.update(choices=choices, value=None),
        gr.update(visible=False)
    )


def update_cuboid_event(selected_name, camera_elevation, camera_lens, description, x, y, z, azimuth, width, depth, height, scale):
    """Update the selected cuboid including description and scale"""
    scene_manager.set_camera_elevation(camera_elevation)
    scene_manager.set_camera_lens(camera_lens)
    
    if selected_name:
        # Find the cuboid by description
        obj_id = None
        for i, obj in enumerate(scene_manager.objects):
            if obj['description'] == selected_name:
                obj_id = i
                break
        
        if obj_id is not None:
            # Update description first if changed
            if description.strip() and description.strip() != selected_name:
                scene_manager.update_cuboid_description(obj_id, description.strip())
            
            # Apply scale to dimensions
            scaled_width = width * scale
            scaled_depth = depth * scale
            scaled_height = height * scale
            
            # Update other properties with scaled dimensions
            scene_manager.update_cuboid(obj_id, x, y, z, azimuth, scaled_width, scaled_depth, scaled_height)
            
            # Get updated object for return
            updated_obj = scene_manager.objects[obj_id]
            new_name = updated_obj['description']
    
    cv_img = scene_manager.render_scene()
    
    # Update choices with new descriptions
    choices = [f"{obj['description']}" for obj in scene_manager.objects]
    
    # Return updated HTML, image, radio choices, new selection, updated sliders, and reset scale
    return (
        get_cuboid_list_html(), 
        cv_img,
        gr.update(choices=choices, value=new_name if obj_id is not None else None),
        gr.update(value=scaled_width if obj_id is not None else width),   # Update width slider
        gr.update(value=scaled_depth if obj_id is not None else depth),   # Update depth slider
        gr.update(value=scaled_height if obj_id is not None else height), # Update height slider
        gr.update(value=1.0)  # Reset scale to 1.0
    )


def camera_change_event(camera_elevation, camera_lens):
    """Handle camera control changes"""
    scene_manager.set_camera_elevation(camera_elevation)
    scene_manager.set_camera_lens(camera_lens)
    cv_img = scene_manager.render_scene()
    return cv_img


def surrounding_prompt_change_event(prompt_text):  # Add this function
    """Handle surrounding prompt changes"""
    scene_manager.set_surrounding_prompt(prompt_text)
    return None  # No visual update needed


def render_segmask_event(camera_elevation, camera_lens, surrounding_prompt):
    """Render segmentation masks for all objects"""
    if not scene_manager.objects:
        return "⚠️ No objects to render", gr.update(visible=False), []
    
    # Get subject descriptions
    subject_descriptions = [obj['description'] for obj in scene_manager.objects]
    
    # Now you have access to:
    # - surrounding_prompt: the text from surrounding_prompt_input
    # - subject_descriptions: list of all subject descriptions
    
    print(f"Surrounding prompt: {surrounding_prompt}")
    print(f"Subject descriptions: {subject_descriptions}")

    placeholder_prompt = "a photo of PLACEHOLDER " + surrounding_prompt 

    # Create placeholder text
    subject_embeds = [] 
    for subject_idx, subject_desc in enumerate(subject_descriptions): 
        input_ids = tokenizer.encode(subject_desc, return_tensors="pt", max_length=77)[0] 
        subject_embed = {"input_ids_t5": input_ids.tolist()} 
        subject_embeds.append(subject_embed)

    placeholder_text = ""
    for subject in subject_descriptions[:-1]:
        placeholder_text = placeholder_text + f"<placeholder> {subject} and "
    for subject in subject_descriptions[-1:]:
        placeholder_text = placeholder_text + f"<placeholder> {subject}"
    placeholder_text = placeholder_text.strip()

    placeholder_token_prompt = placeholder_prompt.replace("PLACEHOLDER", placeholder_text) 

    call_ids = get_call_ids_from_placeholder_prompt_flux(prompt=placeholder_token_prompt, 
                                             subjects=subject_descriptions, 
                                             subjects_embeds=subject_embeds,
                                             debug=True
                                            ) 
    print(f"Generated call IDs: {call_ids}") 

    
    # Convert to server expected format
    subjects_data, camera_data = scene_manager._convert_to_blender_format()
    
    # You can add the prompt and descriptions to the request if needed
    # For example, add to subjects_data or camera_data before sending
    
    # Render segmentation masks
    success, segmask_images, error_msg = scene_manager.render_client.render_segmasks(subjects_data, camera_data)

    # copy all the data to the correct location  
    root_save_dir = "/archive/vaibhav.agrawal/a-bev-of-the-latents/gradio_files/" 
    os.system("rm /archive/vaibhav.agrawal/a-bev-of-the-latents/gradio_files/*") 
    shutil.move("cv_render.jpg", osp.join(root_save_dir, "cv_render.jpg")) 
    for subject_idx in range(len(subject_descriptions)): 
        shutil.move(f"{str(subject_idx).zfill(3)}_segmask_cv.png", osp.join(root_save_dir, f"main__segmask_{str(subject_idx).zfill(3)}__{1.00}.png")) 
    
    jsonl = [{
        "cv": osp.join(root_save_dir, "cv_render.jpg"),
        "target": osp.join(root_save_dir, "cv_render.jpg"),
        "cuboids_segmasks": [osp.join(root_save_dir, f"main__segmask_{str(subject_idx).zfill(3)}__{1.00}.png") for subject_idx in range(len(subject_descriptions))],
        "PLACEHOLDER_prompts": placeholder_prompt, 
        "subjects": subject_descriptions, 
        "call_ids": call_ids, 
    }] 

    with open(osp.join(root_save_dir, "cuboids.jsonl"), "w") as f: 
        for item in jsonl: 
            f.write(json.dumps(item) + "\n") 
    
    if success:
        return (
            f"✅ Successfully rendered {len(segmask_images)} segmentation masks",
            gr.update(visible=True),
            segmask_images
        )
    else:
        return (
            f"❌ Failed to render segmentation masks: {error_msg}",
            gr.update(visible=False),
            []
        )


def harmonize_event(selected_name, camera_elevation, camera_lens):
    """Harmonize all object scales and update the scene"""
    message = scene_manager.harmonize_scales()
    print(message)
    
    cv_img = scene_manager.render_scene()
    
    # If a cuboid is selected, update its sliders
    if selected_name:
        obj = None
        for o in scene_manager.objects:
            if o['description'] == selected_name:
                obj = o
                break
        
        if obj is not None:
            return (
                cv_img,
                get_cuboid_list_html(),
                gr.update(value=obj['position'][0]),
                gr.update(value=obj['position'][1]),
                gr.update(value=obj['position'][2]),
                gr.update(value=obj['azimuth']),
                gr.update(value=obj['size'][0]),
                gr.update(value=obj['size'][1]),
                gr.update(value=obj['size'][2])
            )
    
    # No object selected or object not found
    return (
        cv_img,
        get_cuboid_list_html(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update()
    )


def save_scene_event():
    """Save the current scene to a pkl file"""
    success, filepath, error = scene_manager.save_scene_to_pkl()
    
    if success:
        return f"✅ Scene saved successfully to: {filepath}\n📋 Saved parameters: {scene_manager.inference_params}"
    else:
        return f"❌ Failed to save scene: {error}"


def load_scene_event(filepath):
    """Load a scene from a pkl file and restore all parameters"""
    if not filepath.strip():
        return (
            "⚠️ Please enter a file path",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),  # surrounding_prompt
            gr.update(),  # checkpoint
            gr.update(),  # height
            gr.update(),  # width
            gr.update(),  # seed
            gr.update(),  # guidance
            gr.update()   # steps
        )
    
    success, num_objects, error = scene_manager.load_scene_from_pkl(filepath)
    
    if success:
        # Re-render the scene
        cv_img = scene_manager.render_scene()
        
        # Update UI components
        choices = [f"{obj['description']}" for obj in scene_manager.objects]
        
        params_msg = f"✅ Scene loaded: {num_objects} objects\n📋 Restored parameters: {scene_manager.inference_params}"
        
        return (
            params_msg,
            cv_img,
            get_cuboid_list_html(),
            gr.update(choices=choices, value=None),
            gr.update(visible=False),
            gr.update(value=scene_manager.camera_elevation),
            gr.update(value=scene_manager.camera_lens),
            gr.update(value=scene_manager.surrounding_prompt),
            gr.update(value=scene_manager.inference_params['checkpoint']),
            gr.update(value=scene_manager.inference_params['height']),
            gr.update(value=scene_manager.inference_params['width']),
            gr.update(value=scene_manager.inference_params['seed']),
            gr.update(value=scene_manager.inference_params['guidance_scale']),
            gr.update(value=scene_manager.inference_params['num_inference_steps'])
        )
    else:
        return (
            f"❌ {error}",
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update(),
            gr.update()
        )


# --- Gradio UI Layout ---
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="green",
        secondary_hue="gray",
        neutral_hue="gray"
    ),
    css="""
    .gradio-container {
        background: linear-gradient(135deg, #0d1117 0%, #1a3d2e 50%, #000000 100%) !important;
        color: #ffffff !important;
    }
    .block {
        background: rgba(15, 36, 25, 0.8) !important;
        border: 1px solid #2d5a41 !important;
        border-radius: 8px !important;
    }
    .form {
        background: rgba(15, 36, 25, 0.6) !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
    }
    .markdown {
        color: #e6e6e6 !important;
    }
    label {
        color: #cccccc !important;
    }
    .gr-button {
        background: linear-gradient(135deg, #2d5a41, #3d6a51) !important;
        border: 1px solid #4a7c59 !important;
        color: #ffffff !important;
    }
    .gr-button:hover {
        background: linear-gradient(135deg, #3d6a51, #4a7c59) !important;
    }
    .gr-input, .gr-textbox, .gr-dropdown {
        background: rgba(15, 36, 25, 0.8) !important;
        border: 1px solid #2d5a41 !important;
        color: #ffffff !important;
    }
    .gr-input:focus, .gr-textbox:focus {
        border-color: #4a7c59 !important;
        background: rgba(26, 61, 46, 0.8) !important;
    }
    .gr-slider input[type="range"] {
        background: #2d5a41 !important;
    }
    .gr-slider input[type="range"]::-webkit-slider-thumb {
        background: #4a7c59 !important;
    }
    .gr-radio label {
        color: #cccccc !important;
    }
    .gr-panel {
        background: rgba(15, 36, 25, 0.6) !important;
        border: 1px solid #2d5a41 !important;
    }
    """
) as demo:
    gr.Markdown("# [CVPR-2026] 3D Aware Occlusion Control in Text-to-Image Generation 🏞️🧱")
    # TOP ROW
    with gr.Row():
        # TOP LEFT - Edit Properties
        with gr.Column(scale=1):
            # Add description textbox at the top
            # with gr.Column(visible=False) as editor_section:
            #     gr.Markdown("## ✏️ Edit Properties")
                
            #     delete_btn = gr.Button("❌ Delete Selected Cuboid", variant="stop", size="sm")
                
            #     with gr.Row():
            #         edit_x = gr.Slider(-10, 10, value=0, step=0.1, label="X")
            #         edit_y = gr.Slider(-10, 10, value=0, step=0.1, label="Y")
            #         edit_z = gr.Slider(0, 10, value=1, step=0.1, label="Z")
                
            #     edit_azimuth = gr.Slider(-180, 180, value=0, step=1, label="Azimuth (°)")
                
            #     with gr.Row():
            #         edit_width = gr.Slider(0.1, 5, value=1, step=0.1, label="Width")
            #         edit_depth = gr.Slider(0.1, 5, value=1, step=0.1, label="Depth")
            #         edit_height = gr.Slider(0.1, 5, value=1, step=0.1, label="Height")
            with gr.Column(visible=False) as editor_section:
                gr.Markdown("## ✏️ Edit Properties")

                edit_description = gr.Textbox(
                    label="Description",
                    placeholder="Enter object description",
                    info="Description cannot be empty"
                )
                
                delete_btn = gr.Button("❌ Delete Selected Cuboid", variant="stop", size="sm")
                
                with gr.Row():
                    edit_x = gr.Slider(-10, 10, value=0, step=0.1, label="X")
                    edit_y = gr.Slider(-10, 10, value=0, step=0.1, label="Y")
                    edit_z = gr.Slider(0, 10, value=1, step=0.1, label="Z")
                
                edit_azimuth = gr.Slider(-180, 180, value=0, step=1, label="Azimuth (°)")
                
                with gr.Row():
                    edit_width = gr.Slider(0.1, 5, value=1, step=0.1, label="Width")
                    edit_depth = gr.Slider(0.1, 5, value=1, step=0.1, label="Depth")
                    edit_height = gr.Slider(0.1, 5, value=1, step=0.1, label="Height")
                
                # Add scale slider
                edit_scale = gr.Slider(
                    0.1, 3.0, value=1.0, step=0.1, 
                    label="Scale",
                    info="Multiplier for all dimensions (resets to 1.0 after update)"
                )
                
                # Add the Update Scene button
                update_scene_btn = gr.Button("🔄 Update Scene", variant="primary", size="sm")

        # TOP MIDDLE - Camera View
        with gr.Column(scale=1):
            gr.Markdown("## 📷 Camera View")
            cv_image_output = gr.Image(label="Camera View", height=400)

        # TOP RIGHT - Generated Image
        with gr.Column(scale=1):
            gr.Markdown("## 🎨 Generated Image")
            generated_image_output = gr.Image(label="Generated Image", height=400)

    # BOTTOM ROW
    with gr.Row():
        # BOTTOM LEFT - Cuboid List and Selection
        with gr.Column(scale=1):
            gr.Markdown("## 📦 Cuboids")
            cuboid_list_html = gr.HTML(get_cuboid_list_html())
            
            gr.Markdown("### Select Cuboid to Edit")
            cuboid_radio = gr.Radio(choices=[], label="", visible=True)

        # BOTTOM RIGHT - Camera Controls and Add New Cuboid
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Global Controls")
                    camera_elevation_slider = gr.Slider(0, 90, value=30, label="Camera Elevation (degrees)")
                    camera_lens_slider = gr.Slider(10, 200, value=50, label="Camera Lens (mm)")

                    # Add surrounding prompt textbox
                    surrounding_prompt_input = gr.Textbox(
                        placeholder="e.g., in a forest, in a city, on a beach",
                        label="Surrounding Prompt",
                        info="Describe the surrounding environment"
                    )
                    
                    gr.Markdown("## 🔧 Scene Tools")
                    harmonize_btn = gr.Button("⚖️ Harmonize Scales", variant="secondary")
                    
                    # Save/Load Section
                    gr.Markdown("## 💾 Save/Load Scene")
                    with gr.Row():
                        save_scene_btn = gr.Button("💾 Save Scene", variant="secondary")
                        load_scene_btn = gr.Button("📂 Load Scene", variant="secondary")
                    
                    load_path_input = gr.Textbox(
                        placeholder="/path/to/scene.pkl",
                        label="Load Scene Path",
                        info="Enter path to pkl file to load"
                    )
                    save_load_status = gr.Markdown("")
                
                with gr.Column():
                    gr.Markdown("## ➕ Add New Cuboid")
                    add_cuboid_description_input = gr.Textbox(placeholder="Enter cuboid description", label="Description")
                    asset_type_dropdown = gr.Dropdown(
                        choices=scene_manager.get_asset_type_choices(),
                        value="Custom",
                        label="Type",
                        info="Select asset type to load dimensions, or choose Custom"
                    )
                    add_cuboid_btn = gr.Button("Add Cuboid", variant="primary")
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                    
                    # Add checkpoint dropdown
                    checkpoint_dropdown = gr.Dropdown(
                        choices=CHECKPOINT_NAMES,
                        value=CHECKPOINT_NAMES[0] if CHECKPOINT_NAMES else None,
                        label="Checkpoint",
                        info="Select model checkpoint for generation"
                    )

                    # Inference Parameters
                    gr.Markdown("### Inference Parameters")
                    
                    with gr.Row():
                        inference_height = gr.Slider(
                            minimum=256, maximum=1024, value=512, step=64,
                            label="Height"
                        )
                        inference_width = gr.Slider(
                            minimum=256, maximum=1024, value=512, step=64,
                            label="Width"
                        )
                    
                    inference_seed = gr.Number(
                        value=42, label="Random Seed", precision=0
                    )
                    
                    inference_guidance = gr.Slider(
                        minimum=1.0, maximum=10.0, value=3.5, step=0.5,
                        label="Guidance Scale"
                    )
                    
                    inference_steps = gr.Slider(
                        minimum=10, maximum=50, value=25, step=1,
                        label="Inference Steps"
                    )
    
    # Event Handlers
    def add_cuboid_with_auto_update(description_input, asset_type, camera_elevation, camera_lens):
        """Add cuboid and auto-update scene"""
        result = add_cuboid_event(description_input, asset_type, camera_elevation, camera_lens)
        return result
    
    # Update add_cuboid_btn.click event handler (around line 850):
    add_cuboid_btn.click(
        add_cuboid_with_auto_update,
        inputs=[add_cuboid_description_input, asset_type_dropdown, camera_elevation_slider, camera_lens_slider],
        outputs=[
            add_cuboid_description_input,
            asset_type_dropdown,
            cv_image_output,
            cuboid_list_html,
            cuboid_radio,
            editor_section,
            edit_description,
            edit_x, edit_y, edit_z,
            edit_azimuth,
            edit_width, edit_depth, edit_height,
            edit_scale  # Add this
        ]
    )
    
    # Update the cuboid_radio.change event handler (around line 860):
    cuboid_radio.change(
        select_cuboid_event,
        inputs=[cuboid_radio],
        outputs=[
            editor_section,
            edit_description,
            edit_x, edit_y, edit_z,
            edit_azimuth,
            edit_width, edit_depth, edit_height,
            edit_scale  # Add this
        ]
    )
    
    delete_btn.click(
        delete_selected_cuboid,
        inputs=[cuboid_radio, camera_elevation_slider, camera_lens_slider],
        outputs=[cv_image_output, cuboid_list_html, cuboid_radio, editor_section]
    )

    # Save/Load handlers
    save_scene_btn.click(
        save_scene_event,
        inputs=[],
        outputs=[save_load_status]
    )
    
    load_scene_btn.click(
        load_scene_event,
        inputs=[load_path_input],
        outputs=[
            save_load_status,
            cv_image_output,
            cuboid_list_html,
            cuboid_radio,
            editor_section,
            camera_elevation_slider,
            camera_lens_slider,
            surrounding_prompt_input,
            checkpoint_dropdown,
            inference_height,
            inference_width,
            inference_seed,
            inference_guidance,
            inference_steps
        ]
    )
    
    # Auto-update scene when sliders change
    # for slider in [edit_x, edit_y, edit_z, edit_azimuth, edit_width, edit_depth, edit_height]:
    #     slider.change(
    #         update_cuboid_event,
    #         inputs=[
    #             cuboid_radio,
    #             camera_elevation_slider,
    #             camera_lens_slider,
    #             edit_x, edit_y, edit_z,
    #             edit_azimuth,
    #             edit_width, edit_depth, edit_height
    #         ],
    #         outputs=[cuboid_list_html, cv_image_output]
    #     )
    # Update the update_scene_btn.click event handler (around line 920):
    update_scene_btn.click(
        update_cuboid_event,
        inputs=[
            cuboid_radio,
            camera_elevation_slider,
            camera_lens_slider,
            edit_description,
            edit_x, edit_y, edit_z,
            edit_azimuth,
            edit_width, edit_depth, edit_height,
            edit_scale  # Add this
        ],
        outputs=[
            cuboid_list_html, 
            cv_image_output, 
            cuboid_radio,
            edit_width,   # Add this
            edit_depth,   # Add this
            edit_height,  # Add this
            edit_scale    # Add this (to reset to 1.0)
        ]
    )


    # Update generate button click handler
    generate_btn.click(
        generate_image_event,
        inputs=[
            camera_elevation_slider, 
            camera_lens_slider, 
            surrounding_prompt_input, 
            checkpoint_dropdown,
            inference_height,
            inference_width,
            inference_seed,
            inference_guidance,
            inference_steps
        ],
        outputs=[save_load_status, cv_image_output, generated_image_output]
    )

    
    harmonize_btn.click(
        harmonize_event,
        inputs=[cuboid_radio, camera_elevation_slider, camera_lens_slider],
        outputs=[
            cv_image_output,
            cuboid_list_html,
            edit_x, edit_y, edit_z,
            edit_azimuth,
            edit_width, edit_depth, edit_height
        ]
    )
    
    # Camera controls
    for control in [camera_elevation_slider, camera_lens_slider]:
        control.change(
            camera_change_event,
            inputs=[camera_elevation_slider, camera_lens_slider],
            outputs=[cv_image_output]
        )

    # Surrounding prompt control
    surrounding_prompt_input.change(
        surrounding_prompt_change_event,
        inputs=[surrounding_prompt_input],
        outputs=[]
    )


    # Initial render
    def initial_render():
        cv_img = scene_manager.render_scene()
        gen_img = Image.new('RGB', (512, 512), color='white')
        return cv_img, gen_img
    
    demo.load(
        initial_render, 
        outputs=[cv_image_output, generated_image_output]
    )


if __name__ == "__main__":
    import os 
    os.system("./launch_blender_backend.sh &")
    # Initialize inference engine (load model once at startup)
    initialize_inference_engine(base_model_path="black-forest-labs/FLUX.1-dev")
    demo.launch(share=True)