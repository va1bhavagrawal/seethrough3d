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

# Import BlenderSegmaskRenderer
from blender_backend import BlenderSegmaskRenderer

class SegmaskRenderRequest(BaseModel):
    subjects_data: List[Dict[str, Any]]
    camera_data: Dict[str, Any]
    num_samples: int = 1

class SegmaskRenderResponse(BaseModel):
    success: bool
    segmasks_base64: List[str] = None
    error_message: str = None

class BlenderSegmaskRenderServer:
    def __init__(self):
        """Initialize the Blender segmentation mask render server."""
        self.renderer = BlenderSegmaskRenderer()
        
    def process_render_request(self, request: SegmaskRenderRequest) -> SegmaskRenderResponse:
        """Process a segmentation mask render request and return the result."""
        try:
            # Create temporary directory for this render
            # Convert subjects_data format if needed
            converted_subjects_data = self._convert_subjects_data(request.subjects_data)
            
            # Add required camera_data fields
            camera_data = request.camera_data.copy()
            camera_data["global_scale"] = camera_data.get("global_scale", 1.0)
            
            # Perform the render
            self.renderer.render_cv(
                subjects_data=converted_subjects_data,
                camera_data=camera_data,
                num_samples=request.num_samples
            )
            
            # Read and encode all segmentation masks in order
            segmasks_base64 = []
            num_subjects = len(converted_subjects_data)
            
            for subject_idx in range(num_subjects):
                segmask_path = os.path.join(f"{str(subject_idx).zfill(3)}_segmask_cv.png")
                
                if os.path.exists(segmask_path):
                    with open(segmask_path, "rb") as img_file:
                        img_data = img_file.read()
                        img_base64 = base64.b64encode(img_data).decode('utf-8')
                        segmasks_base64.append(img_base64)
                else:
                    # Return error if any segmask is missing
                    return SegmaskRenderResponse(
                        success=False,
                        error_message=f"Segmentation mask for subject {subject_idx} not found"
                    )
            
            
            return SegmaskRenderResponse(
                success=True,
                segmasks_base64=segmasks_base64
            )
                
        except Exception as e:
            # Change back to original directory on error
            
            return SegmaskRenderResponse(
                success=False,
                error_message=f"Segmentation mask render failed: {str(e)}"
            )
    
    def _convert_subjects_data(self, subjects_data: List[Dict]) -> List[Dict]:
        """Convert subjects data to the format expected by BlenderSegmaskRenderer."""
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
app = FastAPI(title="Blender Segmentation Mask Render Server")

# Global server instance
server = None

@app.on_event("startup")
def startup_event():
    global server
    server = BlenderSegmaskRenderServer()
    print("Blender Segmentation Mask Render Server started")

@app.post("/render_segmasks", response_model=SegmaskRenderResponse)
def render_segmasks(request: SegmaskRenderRequest):
    """Render segmentation masks and return the results."""
    if server is None:
        raise HTTPException(status_code=500, detail="Server not initialized")
    
    return server.process_render_request(request)

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "type": "segmentation_mask_renderer"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blender Segmentation Mask Render Server")
    parser.add_argument("--port", type=int, default=5003, help="Port to run server on")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind server to")
    
    args = parser.parse_args()
    
    uvicorn.run(app, host=args.host, port=args.port, log_level="info")