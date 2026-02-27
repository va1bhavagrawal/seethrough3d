import os
import sys
import tempfile
import shutil
import base64
import io
from PIL import Image
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import uvicorn
import argparse

# Import BlenderCuboidRenderer
from blender_backend import BlenderCuboidRenderer

class RenderRequest(BaseModel):
    subjects_data: List[Dict[str, Any]]
    camera_data: Dict[str, Any]
    num_samples: int = 1

class RenderResponse(BaseModel):
    success: bool
    image_base64: str = None
    error_message: str = None

class BlenderRenderServer:
    def __init__(self, render_mode: str):
        """
        Initialize the Blender render server.
        
        Args:
            render_mode (str): Either 'cv' for camera view or 'bev' for bird's eye view
        """
        self.render_mode = render_mode
        if self.render_mode == "cv":
            self.renderer = BlenderCuboidRenderer("BLENDER_EEVEE_NEXT")
        elif self.render_mode == "final":
            self.renderer = BlenderCuboidRenderer("CYCLES")
        elif self.render_mode == "paper":
            self.renderer = BlenderCuboidRenderer("CYCLES")
        
    def process_render_request(self, request: RenderRequest) -> RenderResponse:
        """Process a render request and return the result."""
        # Create temporary directory for this render
        output_path = os.path.join(f"{self.render_mode}_render.jpg")
        
        # Convert subjects_data format if needed
        converted_subjects_data = self._convert_subjects_data(request.subjects_data)
        
        # Add required camera_data fields
        camera_data = request.camera_data.copy()
        camera_data["global_scale"] = camera_data.get("global_scale", 1.0)
        
        # Perform the render based on mode
        if self.render_mode == "cv":
            self.renderer.render_cv(
                subjects_data=converted_subjects_data,
                camera_data=camera_data,
                num_samples=request.num_samples,
                output_path=output_path
            )
        elif self.render_mode == "final":
            self.renderer.render_final_representation(
                subjects_data=converted_subjects_data,
                camera_data=camera_data,
                num_samples=request.num_samples,
                output_path=output_path
            )
        elif self.render_mode == "paper":
            self.renderer.render_paper_figure(
                subjects_data=converted_subjects_data,
                camera_data=camera_data,
                num_samples=request.num_samples,
                output_path=output_path
            )
        else:
            raise ValueError(f"Invalid render mode: {self.render_mode}")
        
        # Read and encode the rendered image
        if os.path.exists(output_path):
            with open(output_path, "rb") as img_file:
                img_data = img_file.read()
                img_base64 = base64.b64encode(img_data).decode('utf-8')
            
            return RenderResponse(success=True, image_base64=img_base64)
        else:
            return RenderResponse(
                success=False, 
                error_message="Render output file not found"
            )
                
    def _convert_subjects_data(self, subjects_data: List[Dict]) -> List[Dict]:
        """Convert subjects data to the format expected by BlenderCuboidRenderer."""
        converted = []
        
        for subject in subjects_data:
            # Convert to the expected format with lists for x, y, azimuth
            converted_subject = {
                "name": subject.get("subject_name", "cuboid"),
                "x": [subject["x"]],
                "y": [subject["y"]],
                "z": [subject["z"]],
                "dims": [subject["width"], subject["depth"], subject["height"]],
                "azimuth": [subject["azimuth"]]
            }
            converted.append(converted_subject)
            
        return converted

# Create FastAPI app
app = FastAPI(title="Blender Render Server")

# Global server instance
server = None

@app.on_event("startup")
def startup_event():
    global server
    render_mode = os.environ.get("RENDER_MODE") 
    server = BlenderRenderServer(render_mode)
    print(f"Blender Render Server started in {render_mode.upper()} mode")

@app.post("/render", response_model=RenderResponse)
def render_scene(request: RenderRequest):
    """Render a scene and return the result."""
    if server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    return server.process_render_request(request)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "render_mode": server.render_mode if server else "unknown"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Render Server")
    parser.add_argument("--mode", choices=["cv", "final", "paper"], required=True, 
                       help="Render mode: cv for camera view, bev for bird's eye view")
    parser.add_argument("--port", type=int, default=5001, help="Port to run server on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server to")
    
    args = parser.parse_args()
    
    # Set environment variable for the startup event
    os.environ["RENDER_MODE"] = args.mode
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")