# SeeThrough3D Gradio Demo Guide

<div align="center">
  <img src="assets/gradio_demo.png" width="100%">
</div>

This guide provides an overview of the SeeThrough3D Gradio interface and instructions on how to use its various features for occlusion-aware 3D control in text-to-image generation.

## Interface Overview

The interface is divided into two main rows.

### Top Row

1.  **✏️ Edit Properties (Left Column)**
    *   This section becomes visible when you select an object from the "Scene Objects" list.
    *   **Description**: A textbox to update the textual description of the object.
    *   **❌ Delete Selected Cuboid**: Removes the currently selected object from the scene.
    *   **Position Sliders**:
        *   **Away / Towards Camera (X-axis)**: Moves the object along the X-axis (depth relative to standard view). Note that the internal representation shifts this by 6.0 units for rendering.
        *   **Left / Right (Y-axis)**: Moves the object horizontally.
        *   **Up / Down (Z-axis)**: Moves the object vertically. Default is 0 (ground level).
    *   **Azimuth (°)**: Rotates the object around its Z-axis (-180 to 180 degrees).
    *   **Size Sliders**:
        *   **Width, Depth, Height**: Adjust the individual dimensions of the bounding box.
    *   **Scale Slider**: A multiplier to scale the width, depth, and height uniformly. Resets to 1.0 after applying.
    *   **🔄 Update Scene**: Click this button to apply any changes made in the Edit Properties panel and refresh the layout visualization.

2.  **🧊 Layout Visualization (Center Column)**
    *   Displays the 3D layout as rendered by Blender (Camera View). It updates automatically when you add, update, or harmonize objects, or change camera settings.

3.  **🎨 Generated Image (Right Column)**
    *   Displays the final output image after running layout-conditioned diffusion inference.

### Bottom Row

4.  **📦 Scene Objects (Left Column)**
    *   Lists all currently added cuboids in the scene with their properties (Description, Position, Size).
    *   Features a radio button list to select a specific object. Selecting an object makes the "Edit Properties" panel appear in the top-left area.

5.  **Global Scene Controls (Right Column)**
    *   **Camera Elevation (degrees)**: Adjusts the vertical angle of the camera (0 to 90 degrees). Default is 20.
    *   **Camera Lens (mm)**: Adjusts the focal length of the camera (10 to 200 mm). Default is 50.
    *   **Surrounding Prompt**: A textbox to describe the overall environment or background (e.g., "in a forest", "on a street").

6.  **🔧 Scene Tools**
    *   **⚖️ Adjust Object Scales**: Clicking this harmonizes the scales of all "non-Custom" objects based on their base asset sizes, updating their bounding boxes to be proportional to each other.

7.  **💾 Save/Load Scene**
    *   **Save Scene**: Saves the current layout (objects, camera, prompt, and inference parameters) to a `.pkl` file in the `saved_scenes` directory.
    *   **Load Scene Path**: Textbox to input the path of a `.pkl` file you want to load.
    *   **📂 Load Scene**: Loads the layout and parameters from the specified path.

8.  **🖼️ Examples**
    *   A gallery of pre-saved scenes. Clicking an image thumbnail automatically loads the corresponding layout and parameters into the interface.

9.  **➕ Add New Object & Inference Parameters**
    *   **Description**: Enter a short text prompt for the new subject.
    *   **Type**: Dropdown to select a predefined asset type (which loads default geometric dimensions) or "Custom".
    *   **Add Object**: Adds the new object to the scene and updates the layout visualization. Let the new object appear at coordinates (0,0,0) by default.
    *   **🎨 Generate Image**: Initiates the full pipeline: rendering segmentation masks, building internal structures, and running FLUX diffusion inference.
    *   **Checkpoint**: Select the LoRA checkpoint to use for generation.
    *   **Inference Parameters Sliders**: Expandable options to tweak Image Size (default 512), Random Seed, Guidance Scale, and Inference Steps.

## Workflow Example

1.  **Load or Create Scene**: Start by clicking an example in the 🖼️ Examples gallery, or manually click "Add Object" to create new cuboids.
2.  **Define Layout**: Use the "Global Scene Controls" to adjust the camera and provide a "Surrounding Prompt".
3.  **Edit Objects**: Select an object under "Scene Objects". Use the "Edit Properties" panel to change its position, size, and rotation. Click "🔄 Update Scene" to see the changes.
4.  **Harmonize (Optional)**: If you added standard asset types, click "⚖️ Adjust Object Scales" to make their relative sizes realistic.
5.  **Generate**: Adjust any desired model parameters at the bottom right, select a Checkpoint, and click "🎨 Generate Image".
6.  **Save**: If you create a layout you like, click "💾 Save Scene" to store it for later use.
