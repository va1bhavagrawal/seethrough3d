#!/usr/bin/env python3
"""
Flask web server to visualize inference results for 2-subject cases.
Port: 7023
"""

import os
import json
from pathlib import Path
from flask import Flask, render_template, send_from_directory
import base64

app = Flask(__name__)

# Paths
DATASET_FILE = "/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv7_superhard_eval/cuboids_segmentation.jsonl"
DATASET_ROOT = "/archive/vaibhav.agrawal/a-bev-of-the-latents/datasetv7_superhard_eval"
RESULTS_DIR = "/archive/vaibhav.agrawal/a-bev-of-the-latents/VAL/results/omini_seg_baseline_r2_epoch-0_checkpoint-20000"

def load_2_subject_cases():
    """Load all 2-subject cases from the dataset."""
    cases = []
    with open(DATASET_FILE, 'r') as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            if len(data['subjects']) == 2:
                cases.append({
                    'dataset_index': idx,
                    'subjects': data['subjects'],
                    'prompt': data['prompt'],
                    'target': data['target'],
                    'cv': data['cv']
                })
    return cases

# Load cases on startup
TWO_SUBJECT_CASES = load_2_subject_cases()
print(f"Loaded {len(TWO_SUBJECT_CASES)} 2-subject cases")

def get_image_path(case, image_type):
    """Get the path for a specific image type."""
    if image_type == 'ground_truth':
        return os.path.join(DATASET_ROOT, case['target'])
    elif image_type == 'segmentation':
        return os.path.join(DATASET_ROOT, case['cv'])
    elif image_type == 'generated':
        # Find the generated image in results
        viz_dir = os.path.join(RESULTS_DIR, 'generated_images')
        # Pattern: sample_{sample_idx:04d}_idx_{dataset_index}_seed_{seed}.jpg
        # We need to find the file that matches the dataset_index
        if os.path.exists(viz_dir):
            for filename in os.listdir(viz_dir):
                if f"_idx_{case['dataset_index']}_" in filename:
                    return os.path.join(viz_dir, filename)
    return None

@app.route('/')
def index():
    """Main page showing the first 2-subject case."""
    return show_case(0)

@app.route('/case/<int:case_idx>')
def show_case(case_idx):
    """Display a specific case."""
    if case_idx < 0 or case_idx >= len(TWO_SUBJECT_CASES):
        return "Case not found", 404
    
    case = TWO_SUBJECT_CASES[case_idx]
    
    # Get image paths
    gt_path = get_image_path(case, 'ground_truth')
    seg_path = get_image_path(case, 'segmentation')
    gen_path = get_image_path(case, 'generated')
    
    # Check if files exist
    gt_exists = os.path.exists(gt_path) if gt_path else False
    seg_exists = os.path.exists(seg_path) if seg_path else False
    gen_exists = os.path.exists(gen_path) if gen_path else False
    
    return render_template('viewer.html',
                         case_idx=case_idx,
                         total_cases=len(TWO_SUBJECT_CASES),
                         subjects=', '.join(case['subjects']),
                         prompt=case['prompt'].replace('PLACEHOLDER', ', '.join(case['subjects'])),
                         dataset_index=case['dataset_index'],
                         gt_exists=gt_exists,
                         seg_exists=seg_exists,
                         gen_exists=gen_exists,
                         prev_idx=case_idx - 1 if case_idx > 0 else None,
                         next_idx=case_idx + 1 if case_idx < len(TWO_SUBJECT_CASES) - 1 else None)

@app.route('/image/<int:case_idx>/<image_type>')
def serve_image(case_idx, image_type):
    """Serve the requested image."""
    if case_idx < 0 or case_idx >= len(TWO_SUBJECT_CASES):
        return "Case not found", 404
    
    case = TWO_SUBJECT_CASES[case_idx]
    image_path = get_image_path(case, image_type)
    
    if image_path and os.path.exists(image_path):
        directory = os.path.dirname(image_path)
        filename = os.path.basename(image_path)
        return send_from_directory(directory, filename)
    else:
        return "Image not found", 404

if __name__ == '__main__':
    # Create templates directory if it doesn't exist
    os.makedirs('templates', exist_ok=True)
    
    # Run server on all interfaces (0.0.0.0) for remote access
    print(f"Starting server on port 7023...")
    print(f"Access at: http://<your-host-ip>:7023")
    app.run(host='0.0.0.0', port=7023, debug=True)
