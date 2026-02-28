# [CVPR-26 🎉] SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation

<!-- ---
pipeline_tag: occlusion aware 3d control in text-to-image generation    # Specify the task
library_name: diffusers       # Specify the library
language:
  - en                           # List languages your model supports
license: apache-2.0              # Specify a license
datasets:
  - va1bhavagrawal/seethrough3d_dataset            # List datasets used for training
base_model: black-forest-labs/FLUX.1-dev  # If applicable (your model is a fine-tune, quantized, merged version of another model)
tags:                            # Add extra tags which would make the repo searchable using the tag
  - controllable text-to-image generation
  - diffusion models 
--- -->

[![arXiv](https://img.shields.io/badge/arXiv-2602.23359-b31b1b.svg)](https://arxiv.org/abs/2602.23359)
[![Project Page](https://img.shields.io/badge/Project-Page-blue.svg)](https://seethrough3d.github.io)


<div align="center">
  <img src="assets/teaser_camera_ready.svg" width="50%">
</div>

## 🚀 Getting Started 

We recommend creating a `conda` environment named `st3d` with Python 3.11:

```bash
conda create -n st3d python=3.11
conda activate st3d
```

Install the dependencies using the provided `requirements.txt`, then install the project itself in editable mode:

```bash
pip install -r requirements.txt
pip install -e .
```

## 🎨 Inference 

Inference of this model requires ~38 GB VRAM on the GPU. Note that the inference runs Blender in EEVEE mode, which runs faster on workstation GPUs like the NVIDIA RTX A6000, compared to data center GPUs like the NVIDIA H100.  

### 🌐 Downloading pre-trained checkpoint 

We use [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) as the base model. To download the SeeThrough3D LoRA checkpoint,
```bash
cd checkpoints 
``` 
<div align="center">
  <img src="assets/gradio_demo.png" width="100%">
</div>

### 🤗 Gradio Interface 

It is best to perform inference using the 🤗 Gradio interface, which makes it easy to specify 3D layouts. To launch the interface, run
```
cd inference 
conda activate st3d 
python3 app.py 
``` 

For a detailed guide on how to use the Gradio interface, please refer to the [Gradio Wiki](gradio_wiki.md).

The created 3D layouts can be saved by clicking the `💾 Save Scene` button. This functionality stores the 3D layout along with other information such as seed, image size, prompt, etc. in a pickle file. We also provide various example layouts under the `🖼️ Examples` section.  


The interface requires some available ports on the host machine, these can be configured in `inference/config.py`. 

### 📒 Notebook Inference  

The inference notebook is located at `infer.ipynb`. It is able to load a scene saved by the 🤗 Gradio interface (described above), visualize the inputs to the model, and perform inference. The inference notebook also requires some available ports on the host machine, these can be configured in `inference/config.py`.  

## 🏋 Training  

### 🌐 Downloading the Dataset

```bash
cd dataset 

```

### 🏃 Run Training

We train the model for a single epoch at resolution 512, effective batch size of 2 (~25K steps). This requires 2x 80 GB GPUs (one image per GPU). 
```bash
cd train 
# edit the default_config.yaml to specify GPU configuration  
conda activate st3d 
./train.sh 
```
The training takes ~6 hours on 2x NVIDIA H100 GPUs. 

We further do an optional second stage finetuning (~5K steps) at resolution 1024, which improves control and realism during inference. The number of finetuning steps for this stage can be controlled by setting the flag `--stage2_steps`. This stage requires 2x 96 GB GPUs. 

The training VRAM requirement can be reduced to 2x 48 GB GPUs (for the first stage) by **caching text embeddings**. 
To cache the text embeddings, run the following.
```bash
cd train/caching 
# change `DATASET_JSONL` global var in `train/caching/cache.py` to point to training dataset jsonl  
conda activate st3d 
./cache_text_embeddings.sh 
``` 

Now, set the flag `--inference_embeds_dir` in `train/train.sh` to the location of the cached text embeddings.

> **Note:** The VRAM requirements can be further reduced using training time optimizations such as gradient checkpointing. We plan to implement this in the future. We are also welcome to any PRs regarding this.  

## Citation

If you find this work useful please cite:

```bibtex
@misc{agrawal2026seethrough3docclusionaware3d,
      title={SeeThrough3D: Occlusion Aware 3D Control in Text-to-Image Generation}, 
      author={Vaibhav Agrawal and Rishubh Parihar and Pradhaan Bhat and Ravi Kiran Sarvadevabhatla and R. Venkatesh Babu},
      year={2026},
      eprint={2602.23359},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2602.23359}, 
}
```