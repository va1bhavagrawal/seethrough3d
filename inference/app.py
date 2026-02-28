import os
os.environ["GRADIO_TEMP_DIR"]="./gradio_tmp" 
os.environ["GRADIO_TMP_DIR"]="./gradio_tmp" 
os.environ["TEMPDIR"]="./gradio_tmp" 
os.environ["TMP_DIR"]="./gradio_tmp" 
os.environ["TEMP_DIR"]="./gradio_tmp" 
os.environ["TMPDIR"]="./gradio_tmp" 
import os.path as osp 
import sys
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import tempfile
import shutil 
import base64
import io
import threading
import queue
from PIL import Image
import gradio as gr
import time
import copy
import requests
import json
import pickle 
from concurrent.futures import ThreadPoolExecutor, as_completed
from inference.object_scales import scales  
from transformers import CLIPTokenizer, PretrainedConfig, T5TokenizerFast
import pickle 
from datetime import datetime 
from inference.infer_backend import initialize_inference_engine, run_inference_from_gradio
import inference.config as config 

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

tokenizer = T5TokenizerFast.from_pretrained(
    config.PRETRAINED_MODEL_NAME_OR_PATH, 
    subfolder="tokenizer_2",
    revision=None, 
)

placeholder_token_str = ["<placeholder>"]
num_added_tokens = tokenizer.add_tokens(placeholder_token_str)   
assert num_added_tokens == 1 

def generate_image_event(camera_elevation, camera_lens, surrounding_prompt, checkpoint_name, 
                        image_size, seed, guidance_scale, num_steps, scene_manager):
    """Generate final image with segmentation masks and run inference"""
    # Update scene manager's inference params before generation
    scene_manager.update_inference_params(image_size, image_size, seed, guidance_scale, num_steps, checkpoint_name)
    if not scene_manager.objects:
        return (
            "⚠️ No objects to render",
            gr.update(),
            Image.new('RGB', (512, 512), color='white'),
            scene_manager
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
    
    # Acquire a Blender worker for all rendering in this generation
    render_client = blender_pool.acquire()
    
    # Render final high-quality image using CYCLES
    final_img = render_client._send_render_request(
        render_client.final_server_url, 
        subjects_data, 
        camera_data
    )

    final_img.save("model_condition.jpg") 
    
    # Render segmentation masks
    success, segmask_images, error_msg = render_client.render_segmasks(subjects_data, camera_data)
    
    if not success:
        blender_pool.release(render_client)
        return (
            f"❌ Failed to render segmentation masks: {error_msg}",
            gr.update(),
            Image.new('RGB', (512, 512), color='white'),
            scene_manager
        )

    # Save all files to the per-session directory  
    root_save_dir = scene_manager.session_files_dir 
    os.makedirs(root_save_dir, exist_ok=True)
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
    
    # Run inference using the pre-loaded model (GPU lock ensures serial execution)
    print(f"\n{'='*60}")
    print(f"RUNNING INFERENCE")
    print(f"{'='*60}\n")
    
    with gpu_lock:
        inference_success, generated_image, inference_msg = run_inference_from_gradio(
            checkpoint_name=checkpoint_name,
            height=image_size,
            width=image_size,
            seed=seed,
            guidance_scale=guidance_scale,
            num_inference_steps=num_steps,
            jsonl_path=jsonl_path
        )
    
    if not inference_success:
        blender_pool.release(render_client)
        return (
            f"✅ Saved files but inference failed: {inference_msg}",
            final_img,
            Image.new('RGB', (512, 512), color='white'),
            scene_manager
        )
    
    status_msg = f"✅ Generated image using {checkpoint_name} with {len(segmask_images)} segmentation masks"

    # Render final image for camera view
    final_img = render_client._send_render_request(
        render_client.cv_server_url, 
        subjects_data, 
        camera_data
    )
    
    blender_pool.release(render_client)
    
    return (
        status_msg,
        final_img,  # Display CV render in Camera View
        generated_image,  # Display generated image in Generated Image section
        scene_manager
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
    def __init__(self, cv_server_url: str, segmask_server_url: str, final_server_url: str): 
        """
        Initialize the Blender render client.
        
        Args:
            cv_server_url (str): URL of the camera view render server
            segmask_server_url (str): URL of the segmentation mask render server
            final_server_url (str): URL of the final view render server
        """
        self.cv_server_url = cv_server_url
        self.segmask_server_url = segmask_server_url
        self.final_server_url = final_server_url 
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


# --- Blender Worker Pool (global, shared across all sessions) ---
class BlenderWorkerPool:
    """Thread-safe pool of BlenderRenderClient instances.
    
    Callers use acquire() / release() to borrow a client from the pool.
    If the pool is empty, acquire() blocks until a client is returned.
    """

    def __init__(self):
        self._pool = queue.Queue()

    def add_worker(self, client: BlenderRenderClient):
        self._pool.put(client)

    def acquire(self) -> BlenderRenderClient:
        """Block until a BlenderRenderClient is available and return it."""
        return self._pool.get()

    def release(self, client: BlenderRenderClient):
        """Return a BlenderRenderClient to the pool."""
        self._pool.put(client)


# Global resources shared across all user sessions
blender_pool = BlenderWorkerPool()
gpu_lock = threading.Lock()


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
            'checkpoint': config.CHECKPOINT_NAMES[0] if config.CHECKPOINT_NAMES else None
        }
        
        # Per-session temp directory for rendered files (isolated per user)
        self.session_files_dir = tempfile.mkdtemp(prefix="gradio_session_")
        
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
            # Auto-generate filename with timestamp, saved into configuration saved scenes dir
            save_dir = config.SAVED_SCENES_DIR
            os.makedirs(save_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(save_dir, f"scene_{timestamp}.pkl")
        
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
        # Automatically look in saved_scenes if a basic filename was provided
        if os.path.sep not in filepath:
            filepath = os.path.join(config.SAVED_SCENES_DIR, filepath)
            
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
            if saved_checkpoint and saved_checkpoint in config.CHECKPOINT_NAMES:
                checkpoint = saved_checkpoint
            else:
                checkpoint = config.CHECKPOINT_NAMES[0] if config.CHECKPOINT_NAMES else None
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

    def render_cv_view(self, subjects_data: list, camera_data: dict, render_client: BlenderRenderClient) -> Image.Image:
        """Render only the CV view."""
        if not subjects_data:
            return Image.new('RGB', (512, 512), color='gray')
        
        return render_client._send_render_request(render_client.cv_server_url, subjects_data, camera_data)


    def render_scene(self, width=512, height=512):
        """Render only CV view by acquiring a worker from the global pool."""
        print(f"calling render_scene")
        if not self.objects:
            # Return empty image if no objects
            empty_cv = Image.new('RGB', (width, height), color='gray')
            return empty_cv
        
        # Convert to server expected format
        subjects_data, camera_data = self._convert_to_blender_format()
        print(f"passing {subjects_data = } to render_cv_view in SceneManager") 
        
        # Acquire a Blender worker, render, then release
        render_client = blender_pool.acquire()
        cv_img = self.render_cv_view(subjects_data, camera_data, render_client)
        blender_pool.release(render_client)
        
        return cv_img
    
# --- Gradio Interface Logic ---
# NOTE: No global scene_manager — each user gets their own via gr.State(SceneManager)

def get_cuboid_list_html(scene_manager=None):
    """Generate HTML for the cuboid list with position-based colors"""
    if scene_manager is None or not scene_manager.objects:
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


def make_radio_choices(scene_manager):
    """Generate unique radio button labels for all cuboids."""
    return [f"[{i}] {obj['description']}" for i, obj in enumerate(scene_manager.objects)]


def find_obj_by_radio(selected_name, scene_manager):
    """Extract cuboid index from radio label. Returns (obj_id, obj) or (None, None)."""
    if not selected_name or not selected_name.startswith("["):
        return None, None
    idx = int(selected_name[1:selected_name.index("]")])
    if 0 <= idx < len(scene_manager.objects):
        return idx, scene_manager.objects[idx]
    return None, None


def make_radio_value(obj_id, scene_manager):
    """Generate radio label for a specific cuboid by index."""
    if 0 <= obj_id < len(scene_manager.objects):
        return f"[{obj_id}] {scene_manager.objects[obj_id]['description']}"
    return None


def add_cuboid_event(description_input, asset_type, camera_elevation, camera_lens, scene_manager):
    """Add a new cuboid"""
    if not description_input.strip():
        description_input = "New Cuboid"
    
    new_id = scene_manager.add_cuboid(description_input, asset_type)
    cv_img = scene_manager.render_scene()
    
    # Create choices for radio buttons
    choices = make_radio_choices(scene_manager)
    
    # Get the new object data
    new_obj = scene_manager.objects[new_id]
    
    return (
        gr.update(value=""),  # Clear description input
        gr.update(value="Custom"),  # Reset type dropdown to Custom
        cv_img,
        get_cuboid_list_html(scene_manager),
        gr.update(choices=choices, value=make_radio_value(new_id, scene_manager)),  # Radio with new selection
        gr.update(visible=True),  # Show editor
        gr.update(value=new_obj['description']),  # Set description in editor
        gr.update(value=round(new_obj['position'][0], 2)),
        gr.update(value=round(new_obj['position'][1], 2)),
        gr.update(value=round(new_obj['position'][2], 2)),
        gr.update(value=new_obj['azimuth']),
        gr.update(value=round(new_obj['size'][0], 2)),
        gr.update(value=round(new_obj['size'][1], 2)),
        gr.update(value=round(new_obj['size'][2], 2)),
        gr.update(value=1.0),  # Reset scale to 1.0
        scene_manager
    )


def select_cuboid_event(selected_name, scene_manager):
    """When a cuboid is selected from radio buttons"""
    if not selected_name:
        return [gr.update(visible=False)] + [gr.update() for _ in range(9)] + [scene_manager]
    
    # Find the cuboid by radio label
    _, obj = find_obj_by_radio(selected_name, scene_manager)
    
    if obj is None:
        return [gr.update(visible=False)] + [gr.update() for _ in range(9)] + [scene_manager]
    
    return (
        gr.update(visible=True),  # Show editor
        gr.update(value=obj['description']),  # Set description
        gr.update(value=round(obj['position'][0], 2)),
        gr.update(value=round(obj['position'][1], 2)),
        gr.update(value=round(obj['position'][2], 2)),
        gr.update(value=obj['azimuth']),
        gr.update(value=round(obj['size'][0], 2)),
        gr.update(value=round(obj['size'][1], 2)),
        gr.update(value=round(obj['size'][2], 2)),
        gr.update(value=1.0),  # Reset scale to 1.0
        scene_manager
    )


def delete_selected_cuboid(selected_name, camera_elevation, camera_lens, scene_manager):
    """Delete the currently selected cuboid"""
    if not selected_name:
        return gr.update(), get_cuboid_list_html(scene_manager), gr.update(), gr.update(visible=False), scene_manager
    
    # Find and delete the cuboid
    obj_id, _ = find_obj_by_radio(selected_name, scene_manager)
    
    if obj_id is not None:
        scene_manager.delete_cuboid(obj_id)
    
    cv_img = scene_manager.render_scene()
    
    # Update choices
    choices = make_radio_choices(scene_manager)
    
    return (
        cv_img,
        get_cuboid_list_html(scene_manager),
        gr.update(choices=choices, value=None),
        gr.update(visible=False),
        scene_manager
    )


def update_cuboid_event(selected_name, camera_elevation, camera_lens, description, x, y, z, azimuth, width, depth, height, scale, scene_manager):
    """Update the selected cuboid including description and scale"""
    scene_manager.set_camera_elevation(camera_elevation)
    scene_manager.set_camera_lens(camera_lens)
    
    obj_id = None
    scaled_width = width
    scaled_depth = depth
    scaled_height = height
    
    if selected_name:
        # Find the cuboid by radio label
        obj_id, _ = find_obj_by_radio(selected_name, scene_manager)
        
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
    
    cv_img = scene_manager.render_scene()
    
    # Update choices with new descriptions
    choices = make_radio_choices(scene_manager)
    
    # Return updated HTML, image, radio choices, new selection, updated sliders, and reset scale
    return (
        get_cuboid_list_html(scene_manager), 
        cv_img,
        gr.update(choices=choices, value=make_radio_value(obj_id, scene_manager) if obj_id is not None else None),
        gr.update(value=round(scaled_width, 2) if obj_id is not None else round(width, 2)),
        gr.update(value=round(scaled_depth, 2) if obj_id is not None else round(depth, 2)),
        gr.update(value=round(scaled_height, 2) if obj_id is not None else round(height, 2)),
        gr.update(value=1.0),  # Reset scale to 1.0
        scene_manager
    )


def camera_change_event(camera_elevation, camera_lens, scene_manager):
    """Handle camera control changes"""
    scene_manager.set_camera_elevation(camera_elevation)
    scene_manager.set_camera_lens(camera_lens)
    cv_img = scene_manager.render_scene()
    return cv_img, scene_manager


def surrounding_prompt_change_event(prompt_text, scene_manager):
    """Handle surrounding prompt changes"""
    scene_manager.set_surrounding_prompt(prompt_text)
    return scene_manager


def render_segmask_event(camera_elevation, camera_lens, surrounding_prompt, scene_manager):
    """Render segmentation masks for all objects"""
    if not scene_manager.objects:
        return "⚠️ No objects to render", gr.update(visible=False), [], scene_manager
    
    # Get subject descriptions
    subject_descriptions = [obj['description'] for obj in scene_manager.objects]
    
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
    
    # Acquire a Blender worker for rendering
    render_client = blender_pool.acquire()
    
    # Render segmentation masks
    success, segmask_images, error_msg = render_client.render_segmasks(subjects_data, camera_data)

    blender_pool.release(render_client)

    # copy all the data to the per-session directory  
    root_save_dir = scene_manager.session_files_dir 
    os.makedirs(root_save_dir, exist_ok=True)
    os.system(f"rm -f {root_save_dir}/*") 
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
            segmask_images,
            scene_manager
        )
    else:
        return (
            f"❌ Failed to render segmentation masks: {error_msg}",
            gr.update(visible=False),
            [],
            scene_manager
        )


def harmonize_event(selected_name, camera_elevation, camera_lens, scene_manager):
    """Harmonize all object scales and update the scene"""
    message = scene_manager.harmonize_scales()
    print(message)
    
    cv_img = scene_manager.render_scene()
    
    # If a cuboid is selected, update its sliders
    if selected_name:
        _, obj = find_obj_by_radio(selected_name, scene_manager)
        
        if obj is not None:
            return (
                cv_img,
                get_cuboid_list_html(scene_manager),
                gr.update(value=round(obj['position'][0], 2)),
                gr.update(value=round(obj['position'][1], 2)),
                gr.update(value=round(obj['position'][2], 2)),
                gr.update(value=obj['azimuth']),
                gr.update(value=round(obj['size'][0], 2)),
                gr.update(value=round(obj['size'][1], 2)),
                gr.update(value=round(obj['size'][2], 2)),
                scene_manager
            )
    
    # No object selected or object not found
    return (
        cv_img,
        get_cuboid_list_html(scene_manager),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        gr.update(),
        scene_manager
    )


def save_scene_event(scene_manager):
    """Save the current scene to a pkl file"""
    success, filepath, error = scene_manager.save_scene_to_pkl()
    
    if success:
        return f"✅ Scene saved successfully to: {filepath}\n📋 Saved parameters: {scene_manager.inference_params}", scene_manager
    else:
        return f"❌ Failed to save scene: {error}", scene_manager


def load_scene_event(filepath, scene_manager):
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
            gr.update(),  # image_size
            gr.update(),  # seed
            gr.update(),  # guidance
            gr.update(),  # steps
            scene_manager
        )
    
    success, num_objects, error = scene_manager.load_scene_from_pkl(filepath)
    
    if success:
        # Re-render the scene
        cv_img = scene_manager.render_scene()
        
        # Update UI components
        choices = make_radio_choices(scene_manager)
        
        params_msg = f"✅ Scene loaded: {num_objects} objects\n📋 Restored parameters: {scene_manager.inference_params}"
        
        return (
            params_msg,
            cv_img,
            get_cuboid_list_html(scene_manager),
            gr.update(choices=choices, value=None),
            gr.update(visible=False),
            gr.update(value=scene_manager.camera_elevation),
            gr.update(value=scene_manager.camera_lens),
            gr.update(value=scene_manager.surrounding_prompt),
            gr.update(value=scene_manager.inference_params['checkpoint']),
            gr.update(value=scene_manager.inference_params['height']),
            gr.update(value=scene_manager.inference_params['seed']),
            gr.update(value=scene_manager.inference_params['guidance_scale']),
            gr.update(value=scene_manager.inference_params['num_inference_steps']),
            scene_manager
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
            scene_manager
        )


# --- Gradio UI Layout ---
with gr.Blocks(
    theme=gr.themes.Soft(
        primary_hue="slate",
        secondary_hue="slate",
        neutral_hue="slate"
    ),
    css="""
    /* ── Global ── */
    .gradio-container {
        background: linear-gradient(160deg, #0f0f1a 0%, #161625 40%, #1a1a2e 100%) !important;
        color: #e0e0e8 !important;
        font-family: 'Arial', sans-serif !important;
    }
    * { font-family: 'Arial', sans-serif !important; }

    /* ── Cards / blocks ── */
    .block {
        background: rgba(22, 22, 38, 0.65) !important;
        border: none !important;
        border-radius: 10px !important;
        box-shadow: 0 1px 4px rgba(0,0,0,0.30) !important;
    }
    .form {
        background: transparent !important;
        border: none !important;
    }

    /* ── Typography ── */
    h1, h2, h3, h4, h5, h6 { color: #f0f0f5 !important; }
    .markdown { color: #d0d0da !important; }
    label { color: #b0b0be !important; }

    /* ── Buttons ── */
    .gr-button {
        background: linear-gradient(135deg, #3a3a5c, #4a4a6e) !important;
        border: none !important;
        color: #ffffff !important;
        border-radius: 6px !important;
        transition: background 0.2s ease !important;
    }
    .gr-button:hover {
        background: linear-gradient(135deg, #4a4a6e, #5a5a80) !important;
    }

    /* ── Inputs ── */
    .gr-input, .gr-textbox, .gr-dropdown,
    textarea, input[type="text"], input[type="number"] {
        background: rgba(30, 30, 50, 0.7) !important;
        border: 1px solid rgba(255,255,255,0.08) !important;
        color: #e0e0e8 !important;
        border-radius: 6px !important;
    }
    .gr-input:focus, .gr-textbox:focus,
    textarea:focus, input[type="text"]:focus, input[type="number"]:focus {
        border-color: rgba(130, 130, 200, 0.5) !important;
        background: rgba(35, 35, 58, 0.8) !important;
    }

    /* ── Slider track ── */
    input[type="range"] {
        -webkit-appearance: none !important;
        appearance: none !important;
        height: 3px !important;
        width: calc(100% - 16px) !important;
        margin-left: 8px !important;
        margin-right: 8px !important;
        background: rgba(255,255,255,0.12) !important;
        border-radius: 2px !important;
        outline: none !important;
        overflow: visible !important;
    }
    input[type="range"]::-webkit-slider-thumb {
        -webkit-appearance: none !important;
        appearance: none !important;
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        background: #8888bb !important;
        cursor: pointer !important;
        border: 2px solid #c8c8e0 !important;
        box-sizing: border-box !important;
    }
    input[type="range"]::-moz-range-thumb {
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        background: #8888bb !important;
        cursor: pointer !important;
        border: 2px solid #c8c8e0 !important;
        box-sizing: border-box !important;
    }
    input[type="range"]::-moz-range-track {
        height: 3px !important;
        background: rgba(255,255,255,0.12) !important;
        border-radius: 2px !important;
    }

    /* ── Gradio custom slider thumbs ── */
    .range_slider .thumb,
    .range_slider .thumb::after,
    .range_slider [role="slider"],
    .noUi-handle,
    .slider .thumb,
    .gradio-slider .thumb {
        width: 16px !important;
        height: 16px !important;
        border-radius: 50% !important;
        background: #8888bb !important;
        border: 2px solid #c8c8e0 !important;
        box-sizing: border-box !important;
        cursor: pointer !important;
    }

    /* ── Hide ugly scrollbars on slider containers ── */
    .gradio-slider, .gradio-slider > div, .gradio-slider > div > div,
    .wrap, .range_slider,
    div[class*="slider"], div[class*="Slider"],
    input[type="range"],
    .gradio-slider *,
    .axis-box-red *, .axis-box-green *, .axis-box-white * {
        overflow: visible !important;
        overflow-x: hidden !important;
        scrollbar-width: none !important;
        -ms-overflow-style: none !important;
    }
    .gradio-slider::-webkit-scrollbar,
    .gradio-slider > div::-webkit-scrollbar,
    .gradio-slider > div > div::-webkit-scrollbar,
    .gradio-slider *::-webkit-scrollbar,
    .wrap::-webkit-scrollbar,
    .range_slider::-webkit-scrollbar,
    div[class*="slider"]::-webkit-scrollbar,
    div[class*="Slider"]::-webkit-scrollbar,
    .axis-box-red *::-webkit-scrollbar,
    .axis-box-green *::-webkit-scrollbar,
    .axis-box-white *::-webkit-scrollbar {
        display: none !important;
        width: 0 !important;
        height: 0 !important;
    }
    /* Number input boxes next to sliders */
    .gradio-slider input[type="number"] {
        overflow: hidden !important;
    }

    /* ── Axis slider boxes ── */
    .axis-box-red {
        background: rgba(40, 20, 20, 0.5) !important;
        border: none !important;
        border-left: 3px solid #e05555 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        overflow: visible !important;
    }
    .axis-box-red input[type="range"]::-webkit-slider-thumb {
        background: #e05555 !important;
        border-color: #ffaaaa !important;
        border-radius: 50% !important;
    }
    .axis-box-red input[type="range"]::-moz-range-thumb {
        background: #e05555 !important;
        border-color: #ffaaaa !important;
        border-radius: 50% !important;
    }
    .axis-box-red .range_slider .thumb,
    .axis-box-red .range_slider [role="slider"] {
        background: #e05555 !important;
        border-color: #ffaaaa !important;
        border-radius: 50% !important;
    }
    .axis-box-green {
        background: rgba(20, 40, 20, 0.5) !important;
        border: none !important;
        border-left: 3px solid #55cc55 !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        overflow: visible !important;
    }
    .axis-box-green input[type="range"]::-webkit-slider-thumb {
        background: #55cc55 !important;
        border-color: #aaffaa !important;
        border-radius: 50% !important;
    }
    .axis-box-green input[type="range"]::-moz-range-thumb {
        background: #55cc55 !important;
        border-color: #aaffaa !important;
        border-radius: 50% !important;
    }
    .axis-box-green .range_slider .thumb,
    .axis-box-green .range_slider [role="slider"] {
        background: #55cc55 !important;
        border-color: #aaffaa !important;
        border-radius: 50% !important;
    }
    .axis-box-white {
        background: rgba(40, 40, 40, 0.5) !important;
        border: none !important;
        border-left: 3px solid #cccccc !important;
        border-radius: 8px !important;
        padding: 10px 14px !important;
        overflow: visible !important;
    }
    .axis-box-white input[type="range"]::-webkit-slider-thumb {
        background: #cccccc !important;
        border-color: #ffffff !important;
        border-radius: 50% !important;
    }
    .axis-box-white input[type="range"]::-moz-range-thumb {
        background: #cccccc !important;
        border-color: #ffffff !important;
        border-radius: 50% !important;
    }
    .axis-box-white .range_slider .thumb,
    .axis-box-white .range_slider [role="slider"] {
        background: #cccccc !important;
        border-color: #ffffff !important;
        border-radius: 50% !important;
    }

    .gr-radio label { color: #b0b0be !important; }
    .gr-panel {
        background: rgba(22, 22, 38, 0.5) !important;
        border: none !important;
    }
    """
) as demo:
    gr.Markdown("# [CVPR-2026] Occlusion Aware 3D Control in Text-to-Image Generation")
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
                    with gr.Column(elem_classes=["axis-box-red"], min_width=120):
                        gr.Markdown("**Away / Towards Camera**")
                        edit_x = gr.Slider(-10, 10, value=0, step=0.01, label="", show_label=False)
                    with gr.Column(elem_classes=["axis-box-green"], min_width=120):
                        gr.Markdown("**Left / Right**")
                        edit_y = gr.Slider(-10, 10, value=0, step=0.01, label="", show_label=False)
                    with gr.Column(elem_classes=["axis-box-white"], min_width=120):
                        gr.Markdown("**Up / Down**")
                        edit_z = gr.Slider(0, 10, value=1, step=0.01, label="", show_label=False)
                
                edit_azimuth = gr.Slider(-180, 180, value=0, step=1, label="Azimuth (°)")
                
                with gr.Row():
                    edit_width = gr.Slider(0.0, 5, value=1, step=0.01, label="Width")
                    edit_depth = gr.Slider(0.0, 5, value=1, step=0.01, label="Depth")
                    edit_height = gr.Slider(0.0, 5, value=1, step=0.01, label="Height")
                
                # Add scale slider
                edit_scale = gr.Slider(
                    0.0, 3.0, value=1.0, step=0.01, 
                    label="Scale",
                    info="Multiplier for all dimensions (resets to 1.0 after update)"
                )
                
                # Add the Update Scene button
                update_scene_btn = gr.Button("🔄 Update Scene", variant="primary", size="sm")

        # TOP MIDDLE - Camera View
        with gr.Column(scale=1):
            gr.Markdown("## 🧊 Layout Visualization")
            cv_image_output = gr.Image(label="Camera View", height=400)

        # TOP RIGHT - Generated Image
        with gr.Column(scale=1):
            gr.Markdown("## 🎨 Generated Image")
            generated_image_output = gr.Image(label="Generated Image", height=400)

    # BOTTOM ROW
    with gr.Row():
        # BOTTOM LEFT - Cuboid List and Selection
        with gr.Column(scale=1):
            gr.Markdown("## 📦 Scene Objects")
            cuboid_list_html = gr.HTML(get_cuboid_list_html())
            
            gr.Markdown("### Select Object to Edit")
            cuboid_radio = gr.Radio(choices=[], label="", visible=True)

        # BOTTOM RIGHT - Camera Controls and Add New Cuboid
        with gr.Column(scale=2):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("## Global Scene Controls")
                    camera_elevation_slider = gr.Slider(0, 90, value=20, label="Camera Elevation (degrees)")
                    camera_lens_slider = gr.Slider(10, 200, value=50, label="Camera Lens (mm)")

                    # Add surrounding prompt textbox
                    surrounding_prompt_input = gr.Textbox(
                        placeholder="e.g., in a forest, in a city, on a beach",
                        label="Surrounding Prompt",
                        info="Describe the surrounding environment"
                    )
                    
                    gr.Markdown("## 🔧 Scene Tools")
                    harmonize_btn = gr.Button("⚖️ Adjust Object Scales", variant="secondary")
                    
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
                    
                    example_files = []
                    for i in range(5):
                        webp_path = os.path.join(config.SAVED_SCENES_DIR, f"example{i}.webp")
                        pkl_name = f"example{i}.pkl"
                        if os.path.exists(webp_path):
                            example_files.append((webp_path, pkl_name))
                            
                    if example_files:
                        gr.Markdown("## 🖼️ Examples")
                        example_gallery = gr.Gallery(
                            value=[img for img, pkl in example_files],
                            label="Click an example to load its scene",
                            show_label=True,
                            columns=max(len(example_files), 1),
                            rows=1,
                            allow_preview=False,
                            object_fit="contain",
                            height=120
                        )
                        example_gallery_state = gr.State([pkl for img, pkl in example_files])
                
                with gr.Column():
                    gr.Markdown("## ➕ Add New Object")
                    add_cuboid_description_input = gr.Textbox(placeholder="Enter cuboid description", label="Description")
                    asset_type_dropdown = gr.Dropdown(
                        choices=SceneManager().get_asset_type_choices(),
                        value="Custom",
                        label="Type",
                        info="Select asset type to load dimensions, or choose Custom"
                    )
                    add_cuboid_btn = gr.Button("Add Object", variant="primary")
                    generate_btn = gr.Button("🎨 Generate Image", variant="primary")
                    
                    # Add checkpoint dropdown
                    checkpoint_dropdown = gr.Dropdown(
                        choices=config.CHECKPOINT_NAMES,
                        value=config.CHECKPOINT_NAMES[0] if config.CHECKPOINT_NAMES else None,
                        label="Checkpoint",
                        info="Select model checkpoint for generation"
                    )

                    # Inference Parameters
                    gr.Markdown("### Inference Parameters")
                    
                    inference_image_size = gr.Slider(
                        minimum=256, maximum=1024, value=1024, step=64,
                        label="Image Size"
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
    
    # Session state: each user gets their own SceneManager via gr.State
    session_state = gr.State(SceneManager)

    # Event Handlers
    add_cuboid_btn.click(
        add_cuboid_event,
        inputs=[add_cuboid_description_input, asset_type_dropdown, camera_elevation_slider, camera_lens_slider, session_state],
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
            edit_scale,
            session_state
        ]
    )
    
    cuboid_radio.change(
        select_cuboid_event,
        inputs=[cuboid_radio, session_state],
        outputs=[
            editor_section,
            edit_description,
            edit_x, edit_y, edit_z,
            edit_azimuth,
            edit_width, edit_depth, edit_height,
            edit_scale,
            session_state
        ]
    )
    
    delete_btn.click(
        delete_selected_cuboid,
        inputs=[cuboid_radio, camera_elevation_slider, camera_lens_slider, session_state],
        outputs=[cv_image_output, cuboid_list_html, cuboid_radio, editor_section, session_state]
    )

    # Save/Load handlers
    save_scene_btn.click(
        save_scene_event,
        inputs=[session_state],
        outputs=[save_load_status, session_state]
    )
    
    load_scene_btn.click(
        load_scene_event,
        inputs=[load_path_input, session_state],
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
            inference_image_size,
            inference_seed,
            inference_guidance,
            inference_steps,
            session_state
        ]
    )
    
    def load_from_gallery(state, scene_manager, evt: gr.SelectData):
        pkl_name = state[evt.index]
        return load_scene_event(pkl_name, scene_manager)

    if 'example_gallery' in locals():
        example_gallery.select(
            load_from_gallery,
            inputs=[example_gallery_state, session_state],
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
                inference_image_size,
                inference_seed,
                inference_guidance,
                inference_steps,
                session_state
            ]
        )
    
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
            edit_scale,
            session_state
        ],
        outputs=[
            cuboid_list_html, 
            cv_image_output, 
            cuboid_radio,
            edit_width,
            edit_depth,
            edit_height,
            edit_scale,
            session_state
        ]
    )


    # Generate button click handler
    generate_btn.click(
        generate_image_event,
        inputs=[
            camera_elevation_slider, 
            camera_lens_slider, 
            surrounding_prompt_input, 
            checkpoint_dropdown,
            inference_image_size,
            inference_seed,
            inference_guidance,
            inference_steps,
            session_state
        ],
        outputs=[save_load_status, cv_image_output, generated_image_output, session_state]
    )

    
    harmonize_btn.click(
        harmonize_event,
        inputs=[cuboid_radio, camera_elevation_slider, camera_lens_slider, session_state],
        outputs=[
            cv_image_output,
            cuboid_list_html,
            edit_x, edit_y, edit_z,
            edit_azimuth,
            edit_width, edit_depth, edit_height,
            session_state
        ]
    )
    
    # Camera controls
    for control in [camera_elevation_slider, camera_lens_slider]:
        control.change(
            camera_change_event,
            inputs=[camera_elevation_slider, camera_lens_slider, session_state],
            outputs=[cv_image_output, session_state]
        )

    # Surrounding prompt control
    surrounding_prompt_input.change(
        surrounding_prompt_change_event,
        inputs=[surrounding_prompt_input, session_state],
        outputs=[session_state]
    )


    # Initial render
    def initial_render(scene_manager):
        cv_img = scene_manager.render_scene()
        gen_img = Image.new('RGB', (512, 512), color='white')
        return cv_img, gen_img, scene_manager
    
    demo.load(
        initial_render, 
        inputs=[session_state],
        outputs=[cv_image_output, generated_image_output, session_state]
    )


if __name__ == "__main__":
    import os 
    import time as _time
    
    # Launch multiple Blender backend workers
    for worker_idx in range(config.NUM_BLENDER_WORKERS):
        cv_port, final_port, segmask_port = config.get_blender_ports(worker_idx)
        cmd = f"./launch_blender_backend.sh {cv_port} {final_port} {segmask_port} {segmask_port + 1} &"
        print(f"Launching Blender worker {worker_idx}: ports {cv_port}/{final_port}/{segmask_port}")
        os.system(cmd)
    
    # Populate the global Blender worker pool
    for worker_idx in range(config.NUM_BLENDER_WORKERS):
        cv_url, final_url, segmask_url = config.get_blender_urls(worker_idx)
        client = BlenderRenderClient(
            cv_server_url=cv_url,
            final_server_url=final_url,
            segmask_server_url=segmask_url
        )
        blender_pool.add_worker(client)
        print(f"Added Blender worker {worker_idx} to pool: {cv_url}, {final_url}, {segmask_url}")
    
    # Initialize inference engine (load model once at startup — shared across all sessions)
    initialize_inference_engine(base_model_path=config.PRETRAINED_MODEL_NAME_OR_PATH)
    
    # Enable Gradio queuing so multiple users see "In queue..." and launch
    demo.queue().launch(share=True)