import bpy 
import bpy_extras 
import numpy as np 
import bmesh
import copy 
import PIL 
from PIL import Image 
import matplotlib.pyplot as plt 
import colorsys
import os 
import os.path as osp 
import shutil 
import sys 
import math 
import mathutils 
import random 
import cv2 
from object_scales import scales 
import matplotlib.colors as mcolors
import torch 

def map_point_to_rgb(x, y, z):
    """
    Map (x, y) inside the frustum to an RGB color with continuity and variation.
    """
    # Frustum boundaries
    X_MIN, X_MAX = -12.0, -1.0
    Y_MIN_AT_XMIN, Y_MAX_AT_XMIN = -4.5, 4.5
    Y_MIN_AT_XMAX, Y_MAX_AT_XMAX = -0.5, 0.5
    Z_MIN, Z_MAX = 0.0, 2.50   
    # Normalize x to [0, 1]
    x_norm = (x - X_MIN) / (X_MAX - X_MIN)
    x_norm = np.clip(x_norm, 0, 1)

    # Compute current Y bounds at given x using linear interpolation
    y_min = Y_MIN_AT_XMIN + x_norm * (Y_MIN_AT_XMAX - Y_MIN_AT_XMIN)
    y_max = Y_MAX_AT_XMIN + x_norm * (Y_MAX_AT_XMAX - Y_MAX_AT_XMIN)

    # Normalize y to [0, 1] within current bounds
    if y_max != y_min:
        y_norm = (y - y_min) / (y_max - y_min)
    else:
        y_norm = 0.5
    y_norm = np.clip(y_norm, 0, 1)

    z_norm = (z - Z_MIN) / (Z_MAX - Z_MIN) 

    # Color mapping: more variation along x
    r = x_norm
    # g = 0.5 * y_norm + 0.25 * x_norm
    g = y_norm 
    b = z_norm 

    return (r, g, b)


def set_world_color(color=(0.1, 0.1, 0.1)):
    """
    Sets the world background color to match the grid floor.
    
    Args:
        color (tuple): RGB color values (0-1 range)
    """
    scene = bpy.context.scene
    
    # Create a new world if it doesn't exist
    if scene.world is None:
        world = bpy.data.worlds.new(name="World")
        scene.world = world
    else:
        world = scene.world
    
    # Enable use of nodes for the world
    world.use_nodes = True
    
    # Get the node tree
    nodes = world.node_tree.nodes
    links = world.node_tree.links
    
    # Find or create the Background node
    background_node = None
    for node in nodes:
        if node.type == 'BACKGROUND':
            background_node = node
            break
    
    if background_node is None:
        # Clear existing nodes and create new ones
        nodes.clear()
        background_node = nodes.new(type='ShaderNodeBackground')
        output_node = nodes.new(type='ShaderNodeOutputWorld')
        links.new(background_node.outputs['Background'], output_node.inputs['Surface'])
    
    # Set the background color
    background_node.inputs['Color'].default_value = (*color, 1.0)
    background_node.inputs['Strength'].default_value = 1.0


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

def do_z_pass(seg_masks: torch.Tensor, dist_values: torch.Tensor) -> torch.Tensor:
    """
    Performs a z-pass on segmentation masks based on distance values to the camera.
    For each pixel, if multiple subjects' masks are active, only the one with the smallest distance (closest) remains active.
    
    Args:
        seg_masks (torch.Tensor): Binary segmentation masks of shape (n_subjects, h, w) with dtype uint8.
        dist_values (torch.Tensor): Distance values for each subject of shape (n_subjects,).
    
    Returns:
        torch.Tensor: Processed segmentation masks after z-pass, same shape and dtype as seg_masks.
    """
    # Ensure tensors are on the same device
    device = seg_masks.device
    
    # Get dimensions
    n_subjects, h, w = seg_masks.shape
    
    # Reshape distance values for broadcasting across spatial dimensions
    dist_values_expanded = dist_values.view(n_subjects, 1, 1)
    
    # Create a tensor where active pixels have their distance, others have a high value (1e10)
    masked_dist = torch.where(seg_masks.bool(), dist_values_expanded, torch.tensor(1e10, device=device))
    
    # Find the subject index with the minimum distance for each pixel (shape (h, w))
    closest_indices = torch.argmin(masked_dist, dim=0)
    
    # Initialize output tensor with zeros
    output = torch.zeros_like(seg_masks)
    
    # Scatter 1s into the output tensor where the closest subject's indices are
    # closest_indices.unsqueeze(0) adds a dummy dimension to match scatter's expected shape
    output.scatter_(
        dim=0,
        index=closest_indices.unsqueeze(0),
        src=torch.ones_like(closest_indices.unsqueeze(0), dtype=output.dtype)
    )
    
    # Zero out any positions where the original mask was inactive
    output = output * seg_masks
    
    return output


def get_image_to_world_matrix(camera_obj, render):
    """
    Calculates the matrix to transform a point from clip space to world space.

    Args:
        camera_obj (bpy.types.Object): The camera object.
        render (bpy.types.RenderSettings): The scene's render settings.

    Returns:
        mathutils.Matrix: The 4x4 matrix for clip-to-world transformation.
    """
    # Get the camera's view matrix (world to camera)
    view_matrix = camera_obj.matrix_world.inverted()

    # Get the camera's projection matrix
    # This matrix depends on the render resolution, so it's best to calculate it
    # for the specific dimensions you're using.
    projection_matrix = camera_obj.calc_matrix_camera(
        bpy.context.evaluated_depsgraph_get(),
        x=render.resolution_x,
        y=render.resolution_y,
        scale_x=render.pixel_aspect_x,
        scale_y=render.pixel_aspect_y,
    )

    # Combine and invert to get the clip-to-world matrix
    clip_to_world_matrix = (projection_matrix @ view_matrix).inverted()
    
    return clip_to_world_matrix


def unproject_image_point(camera_obj, image_coord, depth):
    """
    Transforms a 2D image coordinate with a depth value into a 3D world coordinate.

    Args:
        camera_obj (bpy.types.Object): The camera used for rendering.
        image_coord (tuple or list): The (x, y) pixel coordinate.
        depth (float): The depth value at that coordinate (from the Z-pass).

    Returns:
        mathutils.Vector: The calculated 3D point in world space.
    """
    render = bpy.context.scene.render
    
    # 1. Get the clip-to-world transformation matrix
    clip_to_world_mat = get_image_to_world_matrix(camera_obj, render)

    # 2. Convert image coordinates to Normalized Device Coordinates (NDC)
    #    (from [0, res] to [-1, 1])
    ndc_x = (image_coord[0] / render.resolution_x) * 2 - 1
    ndc_y = (image_coord[1] / render.resolution_y) * 2 - 1

    # In Blender's Z-pass, the depth value is the distance from the camera's plane.
    # We can use Blender's utility function to find the 3D vector for the pixel.
    # This vector is in camera space and points from the camera towards the pixel.
    view_vector = bpy_extras.view3d_utils.region_2d_to_vector_3d(
        bpy.context.region,
        bpy.context.space_data.region_3d,
        image_coord
    )

    # 4. Project the view vector into world space and scale by depth
    # The view_vector is normalized and in camera space.
    # To get the point in world space, we transform the vector by the camera's
    # world matrix (not the view matrix).
    world_vector = camera_obj.matrix_world.to_3x3() @ view_vector
    
    # The depth from the Z-pass is the distance along the camera's local Z-axis.
    # To find the true distance along the ray, we must account for the angle.
    # We can calculate the scaling factor 't' for our world_vector.
    camera_forward = -camera_obj.matrix_world.col[2].xyz
    t = depth / world_vector.dot(camera_forward)

    # 5. Calculate the final world coordinate
    # Start from the camera's location and move along the ray.
    world_point = camera_obj.matrix_world.translation + (t * world_vector)

    return world_point

# --- Example Usage ---
# This example assumes you have an active scene with a camera and have rendered an image.
# You would typically run this after rendering, where you can access the depth map.


def multiply_random_color(obj, random_color):
    """
    Multiplies the existing base color of an object's materials
    with a random color.
    """
    for material_slot in obj.material_slots:
        if material_slot.material:
            material = material_slot.material
            if material.use_nodes:
                nodes = material.node_tree.nodes
                links = material.node_tree.links

                # Find the Principled BSDF node
                principled_bsdf = nodes.get("Principled BSDF")
                if not principled_bsdf:
                    continue

                # Get the node connected to the Base Color input
                base_color_input = principled_bsdf.inputs.get("Base Color")
                if not base_color_input:
                    continue

                # Create a MixRGB node and set it to multiply
                mix_rgb_node = nodes.new(type='ShaderNodeMixRGB')
                mix_rgb_node.blend_type = 'MULTIPLY'
                mix_rgb_node.inputs['Fac'].default_value = 2.00  
                mix_rgb_node.location = (principled_bsdf.location.x - 200, principled_bsdf.location.y)

                # Set the second color to a random color
                mix_rgb_node.inputs['Color2'].default_value = random_color

                # If a node is already connected to the Base Color,
                # connect it to the first color input of the MixRGB node.
                if base_color_input.is_linked:
                    original_link = base_color_input.links[0]
                    original_node = original_link.from_node
                    original_socket = original_link.from_socket
                    links.new(original_node.outputs[original_socket.name], mix_rgb_node.inputs['Color1'])
                    links.remove(original_link)
                else:
                    # If no node is connected, use the original default color
                    original_color = base_color_input.default_value
                    mix_rgb_node.inputs['Color1'].default_value = original_color

                # Connect the MixRGB node to the Principled BSDF's Base Color
                links.new(mix_rgb_node.outputs['Color'], base_color_input)


OUTPUT_DIR = "four_subject_renders" 
OBJECTS_DIR = "obja_2units_along_y/glbs" 

NUM_AZIMUTH_BINS = 1      
NUM_LIGHTS = 1    

MAX_TRIES = 25   

IMG_DIM = 1024 

MASK_RES = 50  

THRESHOLD_LOWER = 150 
THRESHOLD_UPPER = 768  

ROOT_OBJS_DIR = "/ssd_scratch/vaibhav.agrawal/a-bev-of-the-latents/glb_files/"  

OBJ_SIDE_LENGTH = 2.0 

def calculate_iou(box1, box2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters:
    box1, box2: Each box is defined by a tuple (x1, y1, x2, y2)
                where (x1, y1) is the top-left corner and (x2, y2) is the bottom-right corner.

    Returns:
    float: IoU value
    """
    # Unpack coordinatesO
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    
    # Determine the coordinates of the intersection rectangle
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    
    # Compute the area of intersection rectangle
    inter_width = max(0, inter_x_max - inter_x_min)
    inter_height = max(0, inter_y_max - inter_y_min)
    intersection_area = inter_width * inter_height
    
    # Compute the area of both bounding boxes
    box1_area = (x1_max - x1_min) * (y1_max - y1_min)
    box2_area = (x2_max - x2_min) * (y2_max - y2_min)
    
    # Compute the area of the union
    union_area = box1_area + box2_area - intersection_area
    
    # Compute IoU
    iou = intersection_area / union_area if union_area > 0 else 0
    
    return iou


def get_object_2d_bbox(empty_obj, scene):
    """
    Get the 2D bounding box coordinates of an object in the rendered image.
    
    Args:
        empty_obj (bpy.types.Object): The empty object containing the child mesh objects.
        scene (bpy.types.Scene): The current scene.
        
    Returns:
        tuple: A tuple containing the 2D bounding box coordinates in pixel space
              in the format (min_x, min_y, max_x, max_y).
    """
    # Get the render settings
    render = scene.render
    res_x = render.resolution_x
    res_y = render.resolution_y
    
    # Initialize the bounding box coordinates
    min_x, min_y = float('inf'), float('inf')
    max_x, max_y = float('-inf'), float('-inf')

    depsgraph = bpy.context.evaluated_depsgraph_get()

    # Iterate through the child mesh objects
    for obj in empty_obj.children:
        if obj.type == 'MESH':
            # Get the bounding box coordinates in world space
            bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
            
            # Transform the bounding box corners to camera space
            for corner in bbox_corners:
                corner_2d = bpy_extras.object_utils.world_to_camera_view(scene, scene.camera, corner)

                # Scale the coordinates to pixel space
                x = corner_2d.x * res_x
                y = (1 - corner_2d.y) * res_y  # Flip Y since Blender renders from bottom to top
                
                # Update the bounding box coordinates
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)
    
    # Return the 2D bounding box coordinates in pixel space
    return (int(min_x), int(min_y), int(max_x), int(max_y))

def reset_cameras(scene) -> None:
    """Resets the cameras in the scene to a single default camera."""
    # Delete all existing cameras
    bpy.ops.object.select_all(action="DESELECT")
    bpy.ops.object.select_by_type(type="CAMERA")
    bpy.ops.object.delete()
    
    # Create a new camera with default properties
    bpy.ops.object.camera_add()
    
    # Get the camera by searching for it (it will be the only camera)
    new_camera = None
    for obj in scene.objects:
        if obj.type == 'CAMERA':
            new_camera = obj
            break
    
    new_camera.name = "Camera"
    
    # Set the new camera as the active camera for the scene
    scene.camera = new_camera


def add_plane():
    print(f"in add_plane")
    
    # Create mesh data
    mesh = bpy.data.meshes.new("Plane")
    backdrop = bpy.data.objects.new("Plane", mesh)
    bpy.context.scene.collection.objects.link(backdrop)
    
    # Create plane geometry using bmesh
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=25.0)  # size=25 gives 50x50 plane
    bm.to_mesh(mesh)
    bm.free()
    
    # Add material
    mat_backdrop = bpy.data.materials.new(name="WhiteMaterial")
    mat_backdrop.diffuse_color = (0, 0, 0, 1)  # Black
    backdrop.data.materials.append(mat_backdrop)


def add_plane_cycles():
    print(f"in add_plane")
    
    # Create mesh data
    mesh = bpy.data.meshes.new("Plane")
    backdrop = bpy.data.objects.new("Plane", mesh)
    bpy.context.scene.collection.objects.link(backdrop)
    
    # Create plane geometry using bmesh
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=25.0)  # size=25 gives 50x50 plane
    bm.to_mesh(mesh)
    bm.free()
    
    # Add material
    mat_backdrop = bpy.data.materials.new(name="WhiteMaterial")
    mat_backdrop.diffuse_color = (0.050, 0.050, 0.050, 1)  # White
    backdrop.data.materials.append(mat_backdrop)


def remove_all_planes():
    # Deselect all objects first
    bpy.ops.object.select_all(action='DESELECT')

    # Select all plane objects in the scene
    for obj in bpy.data.objects:
        if obj.type == 'MESH' and obj.name.startswith('Plane'):
            obj.select_set(True)
    
    # Delete all selected planes
    bpy.ops.object.delete()


def remove_all_lights():
    """Remove all lights from the scene without using operators."""
    lights_to_remove = [obj for obj in bpy.data.objects if obj.type == 'LIGHT']
    
    for light in lights_to_remove:
        bpy.data.objects.remove(light, do_unlink=True)
    
    # Clean up orphaned light data blocks
    for light_data in bpy.data.lights:
        if light_data.users == 0:
            bpy.data.lights.remove(light_data)


def set_lights_cv(radius, center, num_points, intensity):
    print(f"in set_lights_cv") 
    radius = radius + 10.0 
    phi = np.random.uniform(-np.pi / 2, np.pi / 2, num_points)         # azimuthal angle
    cos_theta = np.random.uniform(0.50, 1.0, num_points)          # cos of polar angle
    theta = np.arccos(cos_theta)                              # polar angle
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = cos_theta  # cos(theta) == z on unit sphere
    # Scale to radius and shift to center
    points = np.stack([x, y, z], axis=1) * radius + center
    for point in points: 
        # Track objects before adding light
        before_objs = set(bpy.data.objects)
        bpy.ops.object.light_add(type='POINT', location=point)  
        after_objs = set(bpy.data.objects)
        
        # Get the newly created light
        diff_objs = after_objs - before_objs
        light = list(diff_objs)[0]
        
        light.data.energy = intensity
        light.data.use_shadow = True   
        # light.data.shadow_soft_size = 1.0  # Adjust shadow softness if needed 
    return points


def adjust_color_brightness(rgb_color, factor):
    """
    Adjusts the brightness of an RGB color by a multiplicative factor.

    Args:
        rgb_color (tuple): The base color as an (R, G, B) or (R, G, B, A) tuple.
        factor (float): The factor to multiply the brightness by.
                        > 1.0 makes it lighter, < 1.0 makes it darker.

    Returns:
        tuple: The new (R, G, B, A) color.
    """
    # Use only RGB for conversion, keep alpha separate
    h, s, v = colorsys.rgb_to_hsv(rgb_color[0], rgb_color[1], rgb_color[2])
    
    # Multiply the Value (brightness) by the factor, and clamp it between 0 and 1
    v = max(0, min(1, v * factor))
    
    new_rgb = colorsys.hsv_to_rgb(h, s, v)
    
    # Return as an RGBA tuple, preserving original alpha if it exists
    alpha = rgb_color[3] if len(rgb_color) == 4 else 1.0
    return (new_rgb[0], new_rgb[1], new_rgb[2], alpha)


def get_primitive_object_translucent(base_color=(0.0, 1.0, 0.0), edge_color=None, face_opacity=0.025):
    """
    Spawns a cuboid primitive with individually colored faces and highlighted edges.

    Args:
        base_color (tuple): The base RGB color for the faces.
        edge_color (tuple): The RGBA color for the edges (defaults to white).
        face_opacity (float): The opacity of the cuboid faces (0.0 = invisible, 1.0 = opaque). Default is 0.2.
    """
    # --- Create the Cuboid and Parent ---
    bpy.ops.object.empty_add(type="PLAIN_AXES")
    # empty_object = bpy.context.object
    empty_object = bpy.data.objects.new("Empty", None)
    before_objs = set(bpy.data.objects)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0))
    after_objs = set(bpy.data.objects)
    diff_objs = after_objs - before_objs

    obj = None
    for o in diff_objs:
        obj = o
        obj.parent = empty_object
        world_matrix = obj.matrix_world
        obj.matrix_world = world_matrix

    # --- Create and Assign Materials for Each Face ---
    if obj:
        # left front right back bottom top 
        brightness_factors = [
            0.30, 0.30, 0.30, 0.30, 1.00, 0.30, 
        ]
        colors = [adjust_color_brightness(base_color, factor) for factor in brightness_factors]

        for i, color in enumerate(colors):
            material = bpy.data.materials.new(name=f"FaceColor_{i}")
            material.use_nodes = True
            obj.data.materials.append(material)

            nodes = material.node_tree.nodes
            links = material.node_tree.links
            nodes.clear()

            # Create Principled BSDF instead of Emission for proper transparency
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Alpha'].default_value = face_opacity  # Set face opacity
            bsdf.inputs['Emission Color'].default_value = color[:3] + (1.0,)  # Fixed: Use 'Emission Color' instead of 'Emission'
            bsdf.inputs['Emission Strength'].default_value = 1.0  # Emission strength
            
            material_output = nodes.new(type="ShaderNodeOutputMaterial")
            material_output.location = (200, 0)
            links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])
            
            # Enable transparency settings for the material
            material.blend_method = 'BLEND'
            material.show_transparent_back = False

        if len(obj.data.polygons) == len(colors):
            for i, poly in enumerate(obj.data.polygons):
                poly.material_index = i
        else:
            print("Warning: The number of colors does not match the number of faces.")

        # --- Add Wireframe Edges ---
        # edge_material = bpy.data.materials.new(name="EdgeDelimiterMaterial")
        # edge_material.use_nodes = True
        
        # nodes = edge_material.node_tree.nodes
        # links = edge_material.node_tree.links
        # nodes.clear()

        # if edge_color is None: 
        #     edge_color = adjust_color_brightness(base_color, 0.10) 

        # edge_emission_node = nodes.new(type="ShaderNodeEmission")
        # edge_emission_node.inputs['Color'].default_value = edge_color
        # edge_output_node = nodes.new(type="ShaderNodeOutputMaterial")
        # links.new(edge_emission_node.outputs['Emission'], edge_output_node.inputs['Surface'])

        # obj.data.materials.append(edge_material)
        
        # wire_mod = obj.modifiers.new(name="EdgeDelimiter", type='WIREFRAME')
        # wire_mod.thickness = 0.01
        # wire_mod.use_replace = False
        # wire_mod.material_offset = len(obj.data.materials) - 1

    # --- Bounding Box Calculation ---
    bbox_corners = []
    bpy.context.view_layer.update()
    for child in empty_object.children:
        for corner in child.bound_box:
            world_corner = child.matrix_world @ mathutils.Vector(corner)
            bbox_corners.append(world_corner)

    if not bbox_corners:
        return 0, empty_object

    min_x = min(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)

    max_x = max(corner.x for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)

    return max_z, empty_object


def get_primitive_object_translucent_rgb(base_color=(0.0, 1.0, 0.0), edge_color=None, face_opacity=0.025):
    """
    Spawns a cuboid primitive with individually colored faces and highlighted edges.

    Args:
        base_color (tuple): The base RGB color for the faces.
        edge_color (tuple): The RGBA color for the edges (defaults to white).
        face_opacity (float): The opacity of the cuboid faces (0.0 = invisible, 1.0 = opaque). Default is 0.2.
    """
    # --- Create the Cuboid and Parent ---
    bpy.ops.object.empty_add(type="PLAIN_AXES")
    # empty_object = bpy.context.object
    empty_object = bpy.data.objects.new("Empty", None)
    before_objs = set(bpy.data.objects)
    bpy.ops.mesh.primitive_cube_add(size=0.5, location=(0, 0, 0))
    after_objs = set(bpy.data.objects)
    diff_objs = after_objs - before_objs

    obj = None
    for o in diff_objs:
        obj = o
        obj.parent = empty_object
        world_matrix = obj.matrix_world
        obj.matrix_world = world_matrix

    # --- Create and Assign Materials for Each Face ---
    if obj:
        # left front right back bottom top 
        brightness_factors = [
            0.50, 0.50, 0.50, 0.50, 0.50, 0.50, 
        ]
        red = (1.0, 0.0, 0.0, 1.0) 
        green = (0.0, 1.0, 0.0, 1.0)  
        blue = (0.0, 0.0, 1.0, 1.0) 
        colors = [adjust_color_brightness(green, factor) for factor in brightness_factors[:4]] + [adjust_color_brightness(blue, brightness_factors[4])] + [adjust_color_brightness(red, brightness_factors[5])] 
        colors = [colors[-2], colors[-1], colors[0], colors[1], colors[2], colors[3]] 

        for i, color in enumerate(colors):
            material = bpy.data.materials.new(name=f"FaceColor_{i}")
            material.use_nodes = True
            obj.data.materials.append(material)

            nodes = material.node_tree.nodes
            links = material.node_tree.links
            nodes.clear()

            # Create Principled BSDF instead of Emission for proper transparency
            bsdf = nodes.new(type="ShaderNodeBsdfPrincipled")
            bsdf.location = (0, 0)
            bsdf.inputs['Base Color'].default_value = color
            bsdf.inputs['Alpha'].default_value = face_opacity  # Set face opacity
            bsdf.inputs['Emission Color'].default_value = color[:3] + (1.0,)  # Fixed: Use 'Emission Color' instead of 'Emission'
            bsdf.inputs['Emission Strength'].default_value = 1.0  # Emission strength
            
            material_output = nodes.new(type="ShaderNodeOutputMaterial")
            material_output.location = (200, 0)
            links.new(bsdf.outputs['BSDF'], material_output.inputs['Surface'])
            
            # Enable transparency settings for the material
            material.blend_method = 'BLEND'
            material.show_transparent_back = False

        if len(obj.data.polygons) == len(colors):
            for i, poly in enumerate(obj.data.polygons):
                poly.material_index = i
        else:
            print("Warning: The number of colors does not match the number of faces.")

        # --- Add Wireframe Edges ---
        edge_material = bpy.data.materials.new(name="EdgeDelimiterMaterial")
        edge_material.use_nodes = True
        
        nodes = edge_material.node_tree.nodes
        links = edge_material.node_tree.links
        nodes.clear()

        if edge_color is None: 
            edge_color = adjust_color_brightness(base_color, 0.10) 

        edge_emission_node = nodes.new(type="ShaderNodeEmission")
        edge_emission_node.inputs['Color'].default_value = edge_color
        edge_output_node = nodes.new(type="ShaderNodeOutputMaterial")
        links.new(edge_emission_node.outputs['Emission'], edge_output_node.inputs['Surface'])

        obj.data.materials.append(edge_material)
        
        wire_mod = obj.modifiers.new(name="EdgeDelimiter", type='WIREFRAME')
        wire_mod.thickness = 0.01
        wire_mod.use_replace = False
        wire_mod.material_offset = len(obj.data.materials) - 1

    # --- Bounding Box Calculation ---
    bbox_corners = []
    bpy.context.view_layer.update()
    for child in empty_object.children:
        for corner in child.bound_box:
            world_corner = child.matrix_world @ mathutils.Vector(corner)
            bbox_corners.append(world_corner)

    if not bbox_corners:
        return 0, empty_object

    min_x = min(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)

    max_x = max(corner.x for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)

    return max_z, empty_object



def get_primitive_object(base_color=(0.0, 1.0, 0.0), edge_color=None):
    """
    Spawns a cuboid primitive with individually colored faces and highlighted edges.

    Args:
        base_color (tuple): The base RGB color for the faces.
        edge_color (tuple): The RGBA color for the edges (defaults to white).
    """
    # --- Create the Empty Parent ---
    empty_object = bpy.data.objects.new("Empty", None)
    bpy.context.scene.collection.objects.link(empty_object)
    empty_object.empty_display_type = 'PLAIN_AXES'
    
    # --- Create the Cuboid using bmesh ---
    mesh = bpy.data.meshes.new("Cube")
    obj = bpy.data.objects.new("Cube", mesh)
    bpy.context.scene.collection.objects.link(obj)
    
    # Create cube geometry
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=0.5)
    bm.to_mesh(mesh)
    bm.free()
    
    # Set parent
    obj.parent = empty_object
    world_matrix = obj.matrix_world
    obj.matrix_world = world_matrix

    # --- Create and Assign Materials for Each Face ---
    if obj:
        # left front right back bottom top 
        brightness_factors = [
            0.35, 0.20, 0.65, 0.90, 0.50, 0.50   
        ]
        colors = [adjust_color_brightness(base_color, factor) for factor in brightness_factors]

        for i, color in enumerate(colors):
            material = bpy.data.materials.new(name=f"FaceColor_{i}")
            material.use_nodes = True
            obj.data.materials.append(material)

            nodes = material.node_tree.nodes
            links = material.node_tree.links
            nodes.clear()

            emission_node = nodes.new(type="ShaderNodeEmission")
            emission_node.inputs['Color'].default_value = color
            material_output = nodes.new(type="ShaderNodeOutputMaterial")
            links.new(emission_node.outputs['Emission'], material_output.inputs['Surface'])
            
            material.blend_method = 'BLEND'
            material.show_transparent_back = False

        if len(obj.data.polygons) == len(colors):
            for i, poly in enumerate(obj.data.polygons):
                poly.material_index = i
        else:
            print("Warning: The number of colors does not match the number of faces.")

        # --- MODIFICATION START: Add White Edges ---

        # 1. Create a new material for the wireframe edges
        edge_material = bpy.data.materials.new(name="EdgeDelimiterMaterial")
        edge_material.use_nodes = True
        
        # Set up the nodes for a simple white emission shader
        nodes = edge_material.node_tree.nodes
        links = edge_material.node_tree.links
        nodes.clear()

        if edge_color is None: 
            edge_color = adjust_color_brightness(base_color, 0.10) 

        edge_emission_node = nodes.new(type="ShaderNodeEmission")
        edge_emission_node.inputs['Color'].default_value = edge_color
        edge_output_node = nodes.new(type="ShaderNodeOutputMaterial")
        links.new(edge_emission_node.outputs['Emission'], edge_output_node.inputs['Surface'])

        # 2. Add the edge material to the object's material slots
        obj.data.materials.append(edge_material)
        
        # 3. Add and configure the Wireframe modifier
        wire_mod = obj.modifiers.new(name="EdgeDelimiter", type='WIREFRAME')
        wire_mod.thickness = 0.01  # The thickness of the edge lines
        wire_mod.use_replace = False  # Set to False to keep the original faces
        # This offset tells the modifier to use the last material we added (the white one)
        wire_mod.material_offset = len(obj.data.materials) - 1

        # --- MODIFICATION END ---


    # --- Bounding Box Calculation (remains the same) ---
    bbox_corners = []
    # Update the dependency graph to ensure modifiers are accounted for
    bpy.context.view_layer.update()
    for child in empty_object.children:
        # Use child.bound_box which is in object's local space
        for corner in child.bound_box:
            # Convert corner to world space
            world_corner = child.matrix_world @ mathutils.Vector(corner)
            bbox_corners.append(world_corner)

    if not bbox_corners:
        return 0, empty_object # Return a default value if no corners found

    min_x = min(corner.x for corner in bbox_corners)
    min_y = min(corner.y for corner in bbox_corners)
    min_z = min(corner.z for corner in bbox_corners)

    max_x = max(corner.x for corner in bbox_corners)
    max_y = max(corner.y for corner in bbox_corners)
    max_z = max(corner.z for corner in bbox_corners)

    return max_z, empty_object

class BlenderCuboidRenderer:
    def __init__(self, render_engine): 
        """
        Initialize the Blender cuboid renderer.
        
        Args:
            img_dim (int): Image dimensions (square)
            render_engine (str): Blender render engine ('EEVEE' or 'CYCLES')
            num_lights (int): Number of lights to add
            max_tries (int): Maximum tries for placement
        """
        self.img_dim = 1024 
        self.render_engine = render_engine
        self.blender_grid_dims = scales

        self.radius = 6.0 
        self.center = -6.0 
        
        # Scene references
        self.context = None
        self.scene = None
        self.camera = None
        self.render = None

        # Setup the scene
        self.setup_scene()

        
    def setup_scene(self):
        """
        Setup the basic Blender scene with camera, lighting, and render settings.
        
        Args:
            camera_data (dict): Camera configuration containing elevation, lens, global_scale, etc.
        """
        # Get all objects in the scene
        objects_to_remove = []
        
        for obj in bpy.data.objects:
            # Remove default cube, plane, camera, and lights
            if obj.type in {'MESH', 'LIGHT', 'CAMERA'}:
                objects_to_remove.append(obj)
        
        # Delete the objects
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Also clear orphaned data
        for mesh in bpy.data.meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        
        for light in bpy.data.lights:
            if light.users == 0:
                bpy.data.lights.remove(light)
        
        for camera in bpy.data.cameras:
            if camera.users == 0:
                bpy.data.cameras.remove(camera)

        bpy.context.scene.world = None 

        # Initialize Blender scene
        # bpy.ops.wm.read_factory_settings(use_empty=True)
        self.context = bpy.context 
        self.scene = self.context.scene 
        if self.render_engine == "CYCLES":  
            self.scene.cycles.samples = 32 
        self.render = self.scene.render 
        
        # Set render engine and resolution
        self.render.engine = self.render_engine  
        self.context.scene.render.resolution_x = self.img_dim  
        self.context.scene.render.resolution_y = self.img_dim  
        self.context.scene.render.resolution_percentage = 100  

        # Setup compositing nodes
        self._setup_compositing()
        
        
    def _setup_compositing(self):
        """Setup Blender compositing nodes for depth and RGB output."""
        self.context.scene.use_nodes = True
        tree = self.context.scene.node_tree
        links = tree.links

        self.context.scene.render.use_compositing = True 
        self.context.view_layer.use_pass_z = True
        
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
        
        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')      

        map_node = tree.nodes.new(type="CompositorNodeMapValue")
        map_node.size = [0.05]
        map_node.use_min = True
        map_node.min = [0]
        map_node.use_max = True
        map_node.max = [65336]
        links.new(rl.outputs[2], map_node.inputs[0])

        invert = tree.nodes.new(type="CompositorNodeInvert")
        links.new(map_node.outputs[0], invert.inputs[1])
        
        # create output node
        v = tree.nodes.new('CompositorNodeViewer')   
        v.use_alpha = True 

        # create a file output node and set the path
        fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        fileOutput.base_path = "."
        links.new(invert.outputs[0], fileOutput.inputs[0])

        # Links
        links.new(rl.outputs[0], v.inputs[0])  # link Image to Viewer Image RGB
        links.new(rl.outputs['Depth'], v.inputs[1])  # link Render Z to Viewer Image Alpha

        # Update scene to apply changes
        self.context.view_layer.update() 


    def _setup_camera_cv(self, camera_data):
        """Setup camera position and orientation."""
        reset_cameras(self.scene) 
        self.camera = self.scene.objects["Camera"] 
        
        elevation = camera_data["camera_elevation"] 
        tan_elevation = np.tan(elevation) 
        cos_elevation = np.cos(elevation) 
        sin_elevation = np.sin(elevation) 

        radius = self.radius 
        center = self.center 
        
        self.camera.location = mathutils.Vector((radius * cos_elevation + center, 0, radius * sin_elevation))  
        direction = mathutils.Vector((-1, 0, -tan_elevation)) 
        self.context.scene.camera = self.camera 
        rot_quat = direction.to_track_quat("-Z", "Y") 
        self.camera.rotation_euler = rot_quat.to_euler() 
        self.camera.data.lens = camera_data["lens"]  

    def _create_cuboid_objects_translucent(self, subjects_data, opacity=0.025):
        """Create primitive cuboid objects for all subjects."""
        for subject_idx, subject_data in enumerate(subjects_data): 
            # rgb_color = map_point_to_rgb(x, y) 
            rgb_color = COLORS[subject_idx % len(COLORS)] 
            _, prim_obj = get_primitive_object_translucent(base_color=rgb_color, face_opacity=opacity) 
            prim_obj.location = np.array([100, 0, 0]) 
            subject_data["prim_obj"] = prim_obj

    def _create_cuboid_objects_translucent_rgb(self, subjects_data, opacity=0.025):
        """Create primitive cuboid objects for all subjects."""
        for subject_idx, subject_data in enumerate(subjects_data): 
            x = subject_data["x"][0] 
            y = subject_data["y"][0] 
            z = subject_data["z"][0] 
            base_color = map_point_to_rgb(x, y, z)
            _, prim_obj = get_primitive_object_translucent_rgb(base_color=base_color, face_opacity=opacity) 
            prim_obj.location = np.array([100, 0, 0]) 
            subject_data["prim_obj"] = prim_obj

            
    def _place_objects(self, subjects_data, camera_data):
        """Place objects in the scene according to their data."""
        global_scale = camera_data["global_scale"] 
        
        for subject_data in subjects_data: 
            x = subject_data["x"][0]  
            y = subject_data["y"][0]  
            z = global_scale * subject_data["dims"][2] / 2.0 + subject_data["z"][0]  
            subject_data["prim_obj"].location = np.array([x, y, z])  
            subject_data["prim_obj"].scale = global_scale * np.array(subject_data["dims"]) * 2.0 
            subject_data["prim_obj"].rotation_euler[2] = subject_data["azimuth"][0]  

    def render_cv(self, subjects_data, camera_data, num_samples=1, output_path="main.jpg"):
        """
        Main render method that takes subjects data and renders the scene.
        
        Args:
            subjects_data (list): List of subject dictionaries containing position, dims, etc.
            camera_data (dict): Camera configuration
            num_samples (int): Number of samples to render (currently only supports 1)
            output_path (str): Path to save the rendered image
            
        Returns:
            None
        """
        center = (-6.0, 0.0, 0.0) 
        radius = 6.0 

        print(f"render_cv received {subjects_data = }") 

        # print(f"render_cv received {subjects_data = }")
        for subject_data in subjects_data: 
            subject_data["azimuth"][0] = np.deg2rad(subject_data["azimuth"][0]) 
            subject_data["x"][0] = subject_data["x"][0] + center[0] 
            subject_data["y"][0] = subject_data["y"][0] + center[1] 
            subject_data["z"][0] = subject_data["z"][0] + center[2] 
        # Setup camera
        self._setup_camera_cv(camera_data)
        
        set_lights_cv(self.radius, np.array([self.center, 0, 0]), 20, intensity=7000.0)
        
        # Add ground plane
        add_plane()

        assert num_samples == 1, "for now, only implemented for a single sample"  
        assert "global_scale" in camera_data.keys(), "global_scale must be set for EEVEE" 
        
        # Create primitive objects for subjects
        self._create_cuboid_objects_translucent(subjects_data, opacity=0.025)
        # self._create_cuboid_objects(subjects_data)
        
        # Place objects in scene
        self._place_objects(subjects_data, camera_data)
        
        # Perform rendering
        print(f"SUCCESS, rendering...")
        self.context.scene.render.filepath = output_path 
        self.context.scene.render.image_settings.file_format = "JPEG" 
        bpy.ops.render.render(write_still=True) 
        
        print(f"Rendered scene saved to: {output_path}")

        self.cleanup() 

    def render_final_representation(self, subjects_data, camera_data, num_samples=1, output_path="main.jpg"):
        """
        Main render method that takes subjects data and renders the scene.
        
        Args:
            subjects_data (list): List of subject dictionaries containing position, dims, etc.
            camera_data (dict): Camera configuration
            num_samples (int): Number of samples to render (currently only supports 1)
            output_path (str): Path to save the rendered image
            
        Returns:
            None
        """
        assert self.render.engine == "CYCLES", "render_final_representation only works with CYCLES render engine" 
        center = (-6.0, 0.0, 0.0) 
        radius = 6.0 

        print(f"render_cv received {subjects_data = }") 

        # print(f"render_cv received {subjects_data = }")
        for subject_data in subjects_data: 
            subject_data["azimuth"][0] = np.deg2rad(subject_data["azimuth"][0]) 
            subject_data["x"][0] = subject_data["x"][0] + center[0] 
            subject_data["y"][0] = subject_data["y"][0] + center[1] 
            subject_data["z"][0] = subject_data["z"][0] + center[2] 
        # Setup camera
        self._setup_camera_cv(camera_data)
        
        print(f"setting lights in cycles...")
        set_lights_cv(self.radius, np.array([self.center, 0, 0]), 5, intensity=700.0)
        
        # Add ground plane
        print(f"adding plane in cycles...")
        add_plane_cycles()

        assert num_samples == 1, "for now, only implemented for a single sample"  
        assert "global_scale" in camera_data.keys(), "global_scale must be set for EEVEE" 
        
        # Create primitive objects for subjects
        self._create_cuboid_objects_translucent_rgb(subjects_data, opacity=0.025)
        # self._create_cuboid_objects(subjects_data)
        
        # Place objects in scene
        self._place_objects(subjects_data, camera_data)
        
        # Perform rendering
        print(f"SUCCESS, rendering...")
        self.context.scene.render.filepath = output_path 
        self.context.scene.render.image_settings.file_format = "JPEG" 
        bpy.ops.render.render(write_still=True) 
        
        print(f"Rendered scene saved to: {output_path}")

        self.cleanup() 


    def render_paper_figure(self, subjects_data, camera_data, num_samples=1, output_path="main.jpg"):
        """
        Main render method that takes subjects data and renders the scene.
        
        Args:
            subjects_data (list): List of subject dictionaries containing position, dims, etc.
            camera_data (dict): Camera configuration
            num_samples (int): Number of samples to render (currently only supports 1)
            output_path (str): Path to save the rendered image
            
        Returns:
            None
        """
        assert self.render.engine == "CYCLES", "render_final_representation only works with CYCLES render engine" 
        center = (-6.0, 0.0, 0.0) 
        radius = 6.0 

        print(f"render_cv received {subjects_data = }") 

        set_world_color((1.0, 1.0, 1.0))  # white background

        # print(f"render_cv received {subjects_data = }")
        for subject_data in subjects_data: 
            subject_data["azimuth"][0] = np.deg2rad(subject_data["azimuth"][0]) 
            subject_data["x"][0] = subject_data["x"][0] + center[0] 
            subject_data["y"][0] = subject_data["y"][0] + center[1] 
            subject_data["z"][0] = subject_data["z"][0] + center[2] 
        # Setup camera
        self._setup_camera_cv(camera_data)
        
        print(f"setting lights in cycles...")
        set_lights_cv(self.radius, np.array([self.center, 0, 0]), 5, intensity=7000.0)
        
        # Add ground plane
        print(f"adding plane in cycles...")

        assert num_samples == 1, "for now, only implemented for a single sample"  
        assert "global_scale" in camera_data.keys(), "global_scale must be set for EEVEE" 
        
        # Create primitive objects for subjects
        self._create_cuboid_objects_translucent(subjects_data, opacity=0.35)
        # self._create_cuboid_objects(subjects_data)
        
        # Place objects in scene
        self._place_objects(subjects_data, camera_data)
        
        # Perform rendering
        print(f"SUCCESS, rendering...")
        self.context.scene.render.filepath = output_path 
        self.context.scene.render.image_settings.file_format = "JPEG" 
        bpy.ops.render.render(write_still=True) 
        
        print(f"Rendered scene saved to: {output_path}")

        self.cleanup() 


    def cleanup(self):
        """Clean up the scene for next render."""
        # Remove all lights
        remove_all_lights()
        
        # Remove all other objects (meshes, empties, etc.)
        objects_to_remove = [obj for obj in bpy.data.objects]
        
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Clean up orphaned data blocks
        for mesh in bpy.data.meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        
        for material in bpy.data.materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        
        for light_data in bpy.data.lights:
            if light_data.users == 0:
                bpy.data.lights.remove(light_data)


class BlenderSegmaskRenderer:
    def __init__(self): 
        """
        Initialize the Blender cuboid renderer.
        
        Args:
            img_dim (int): Image dimensions (square)
            render_engine (str): Blender render engine ('EEVEE' or 'CYCLES')
            num_lights (int): Number of lights to add
            max_tries (int): Maximum tries for placement
        """
        self.img_dim = 1024 
        self.render_engine = "BLENDER_WORKBENCH"
        self.blender_grid_dims = scales

        self.radius = 6.0 
        self.center = -6.0 
        
        # Scene references
        self.context = None
        self.scene = None
        self.camera = None
        self.render = None

        # Setup the scene
        self.setup_scene()

        
    def setup_scene(self):
        """
        Setup the basic Blender scene with camera, lighting, and render settings.
        
        Args:
            camera_data (dict): Camera configuration containing elevation, lens, global_scale, etc.
        """
        # Get all objects in the scene
        objects_to_remove = []
        
        for obj in bpy.data.objects:
            # Remove default cube, plane, camera, and lights
            if obj.type in {'MESH', 'LIGHT', 'CAMERA'}:
                objects_to_remove.append(obj)
        
        # Delete the objects
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Also clear orphaned data
        for mesh in bpy.data.meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        
        for light in bpy.data.lights:
            if light.users == 0:
                bpy.data.lights.remove(light)
        
        for camera in bpy.data.cameras:
            if camera.users == 0:
                bpy.data.cameras.remove(camera)

        bpy.context.scene.world = None 

        # Initialize Blender scene
        # bpy.ops.wm.read_factory_settings(use_empty=True)
        self.context = bpy.context 
        self.scene = self.context.scene 
        self.render = self.scene.render 
        
        # Set render engine and resolution
        self.render.engine = self.render_engine  
        self.context.scene.render.resolution_x = self.img_dim  
        self.context.scene.render.resolution_y = self.img_dim  
        self.context.scene.render.resolution_percentage = 100  

        # Setup compositing nodes
        self._setup_compositing()
        
        
    def _setup_compositing(self):
        """Setup Blender compositing nodes for depth and RGB output."""
        self.context.scene.use_nodes = True
        tree = self.context.scene.node_tree
        links = tree.links

        self.context.scene.render.use_compositing = True 
        self.context.view_layer.use_pass_z = True
        
        # clear default nodes
        for n in tree.nodes:
            tree.nodes.remove(n)
        
        # create input render layer node
        rl = tree.nodes.new('CompositorNodeRLayers')      

        map_node = tree.nodes.new(type="CompositorNodeMapValue")
        map_node.size = [0.05]
        map_node.use_min = True
        map_node.min = [0]
        map_node.use_max = True
        map_node.max = [65336]
        links.new(rl.outputs[2], map_node.inputs[0])

        invert = tree.nodes.new(type="CompositorNodeInvert")
        links.new(map_node.outputs[0], invert.inputs[1])
        
        # create output node
        v = tree.nodes.new('CompositorNodeViewer')   
        v.use_alpha = True 

        # create a file output node and set the path
        fileOutput = tree.nodes.new(type="CompositorNodeOutputFile")
        fileOutput.base_path = "."
        links.new(invert.outputs[0], fileOutput.inputs[0])

        # Links
        links.new(rl.outputs[0], v.inputs[0])  # link Image to Viewer Image RGB
        links.new(rl.outputs['Depth'], v.inputs[1])  # link Render Z to Viewer Image Alpha

        # Update scene to apply changes
        self.context.view_layer.update() 


    def _setup_camera_cv(self, camera_data):
        """Setup camera position and orientation."""
        reset_cameras(self.scene) 
        self.camera = self.scene.objects["Camera"] 
        
        elevation = camera_data["camera_elevation"] 
        tan_elevation = np.tan(elevation) 
        cos_elevation = np.cos(elevation) 
        sin_elevation = np.sin(elevation) 

        radius = self.radius 
        center = self.center 
        
        self.camera.location = mathutils.Vector((radius * cos_elevation + center, 0, radius * sin_elevation))  
        direction = mathutils.Vector((-1, 0, -tan_elevation)) 
        self.context.scene.camera = self.camera 
        rot_quat = direction.to_track_quat("-Z", "Y") 
        self.camera.rotation_euler = rot_quat.to_euler() 
        self.camera.data.lens = camera_data["lens"]  
        
    def _create_cuboid_objects(self, subjects_data):
        """Create primitive cuboid objects for all subjects."""
        for subject_idx, subject_data in enumerate(subjects_data): 
            x = subject_data["x"][0] 
            y = subject_data["y"][0] 
            z = subject_data["z"][0] 
            rgb_color = map_point_to_rgb(x, y, z) 
            _, prim_obj = get_primitive_object(rgb_color) 
            prim_obj.location = np.array([100, 0, 0]) 
            subject_data["prim_obj"] = prim_obj

    def _place_objects(self, subjects_data, camera_data):
        """Place objects in the scene according to their data."""
        global_scale = camera_data["global_scale"] 
        
        for subject_data in subjects_data: 
            x = subject_data["x"][0]  
            y = subject_data["y"][0]  
            z = global_scale * subject_data["dims"][2] / 2.0 + subject_data["z"][0]  
            subject_data["prim_obj"].location = np.array([x, y, z])  
            subject_data["prim_obj"].scale = global_scale * np.array(subject_data["dims"]) * 2.0 
            subject_data["prim_obj"].rotation_euler[2] = subject_data["azimuth"][0]  

    def render_cv(self, subjects_data, camera_data, num_samples=1):
        """
        Main render method that takes subjects data and renders the scene.
        
        Args:
            subjects_data (list): List of subject dictionaries containing position, dims, etc.
            camera_data (dict): Camera configuration
            num_samples (int): Number of samples to render (currently only supports 1)
            output_path (str): Path to save the rendered image
            
        Returns:
            None
        """
        # Setup camera
        center = (-6.0, 0.0, 0.0) 
        radius = 6.0 

        for subject_data in subjects_data: 
            subject_data["azimuth"][0] = np.deg2rad(subject_data["azimuth"][0]) 
            subject_data["x"][0] = subject_data["x"][0] + center[0] 
            subject_data["y"][0] = subject_data["y"][0] + center[1] 
            subject_data["z"][0] = subject_data["z"][0] + center[2] 

        print(f"in segmask render, {subjects_data = }") 

        self._setup_camera_cv(camera_data)
        
        assert num_samples == 1, "for now, only implemented for a single sample"  
        assert "global_scale" in camera_data.keys(), "global_scale must be set" 
        
        # Create primitive objects for subjects
        self._create_cuboid_objects(subjects_data)

        def make_segmask(image): 
            alpha = image[:, :, 3] 
            _, mask = cv2.threshold(alpha, 0, 255, cv2.THRESH_BINARY) 
            return mask  


        for subject_idx, subject_data in enumerate(subjects_data): 
            # Place objects in scene
            self._place_objects([subject_data], camera_data) 
        
            # Perform rendering
            print(f"SUCCESS, rendering...")
            self.context.scene.render.filepath = "tmp.png"
            self.context.scene.render.image_settings.file_format = "PNG" 
            bpy.ops.render.render(write_still=True) 
            img = cv2.imread("tmp.png", cv2.IMREAD_UNCHANGED) 
            segmask = make_segmask(img)  
            print(f"{segmask.shape = }") 
            cv2.imwrite(f"{str(subject_idx).zfill(3)}_segmask_cv.png", segmask) 
            print(f"saved {str(subject_idx).zfill(3)}_segmask_cv.png") 

            subject_data["prim_obj"].location = np.array([100, 0, 0]) # move out of view 
        
        self.cleanup() 


    def cleanup(self):
        """Clean up the scene for next render."""
        # Remove all lights
        remove_all_lights()
        
        # Remove all other objects (meshes, empties, etc.)
        objects_to_remove = [obj for obj in bpy.data.objects]
        
        for obj in objects_to_remove:
            bpy.data.objects.remove(obj, do_unlink=True)
        
        # Clean up orphaned data blocks
        for mesh in bpy.data.meshes:
            if mesh.users == 0:
                bpy.data.meshes.remove(mesh)
        
        for material in bpy.data.materials:
            if material.users == 0:
                bpy.data.materials.remove(material)
        
        for light_data in bpy.data.lights:
            if light_data.users == 0:
                bpy.data.lights.remove(light_data)
        

        
# Update the main execution
if __name__ == '__main__':
    subjects_data = [
        {
            "name": "sedan", 
            "x": [-5.0],
            "y": [0.0], 
            "dims": [1.0, 2.0, 1.5],
            "azimuth": [0.0]
        }, 
    ]
    camera_data = {
        "camera_elevation": np.arctan(0.45), 
        "lens": 70,
        "global_scale": 1.0
    }
    
    # Create renderer instance
    renderer = BlenderCuboidRenderer(
        img_dim=1024,
        render_engine='EEVEE',
        num_lights=1,
    )
    
    # Render the scene
    renderer.render(
        subjects_data=subjects_data, 
        camera_data=camera_data, 
        num_samples=1,
        output_path="main.jpg"
    )