from PIL import Image
from datasets import load_dataset
from torchvision import transforms
import random
import torch.nn.functional as F 
import torch
import os 
import os.path as osp 
import cv2

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

Image.MAX_IMAGE_PIXELS = None

def multiple_16(num: float):
    return int(round(num / 16) * 16)

def get_random_resolution(min_size=512, max_size=1280, multiple=16):
    resolution = random.randint(min_size // multiple, max_size // multiple) * multiple
    return resolution

def load_image_safely(image_path, size):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print("file error: "+image_path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{image_path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))
    
def make_train_dataset(args, tokenizer, accelerator):
    if args.current_train_data_dir is not None:
        print("load_data")
        dataset = load_dataset('json', data_files=args.current_train_data_dir)

    # Add index column to the dataset
    dataset = dataset.map(lambda examples, indices: {**examples, 'index': indices}, with_indices=True, batched=True)

    column_names = dataset["train"].column_names
    
    # 6. Get the column names for input/target.
    target_column = args.target_column
    if args.subject_column is not None:
        subject_columns = args.subject_column.split(",")
    if args.spatial_column is not None:
        spatial_columns= args.spatial_column.split(",")
    
    size = args.cond_size
    # by default the noise size would be randomly sampled from (512, 1024) 
    # noise_size = get_random_resolution(max_size=args.noise_size) # maybe 768 or higher
    noise_size = get_random_resolution(min_size=512, max_size=512) # maybe 768 or higher
    # subject_cond_train_transforms = transforms.Compose(
    #     [
    #         transforms.Lambda(lambda img: img.resize((
    #             multiple_16(size * img.size[0] / max(img.size)),
    #             multiple_16(size * img.size[1] / max(img.size))
    #         ), resample=Image.BILINEAR)),
    #         transforms.RandomHorizontalFlip(p=0.7),
    #         transforms.RandomRotation(degrees=20),
    #         transforms.Lambda(lambda img: transforms.Pad(
    #             padding=(
    #                 int((size - img.size[0]) / 2),
    #                 int((size - img.size[1]) / 2),
    #                 int((size - img.size[0]) / 2),
    #                 int((size - img.size[1]) / 2) 
    #             ),
    #             fill=0 
    #         )(img)),
    #         transforms.ToTensor(),
    #         transforms.Normalize([0.5], [0.5]),
    #     ]
    # )
    cond_train_transforms = transforms.Compose(
        [
            transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop((size, size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    subject_cond_train_transforms = cond_train_transforms  

    def train_transforms(image, noise_size):
        train_transforms_ = transforms.Compose(
            [
                transforms.Lambda(lambda img: img.resize((
                    multiple_16(noise_size * img.size[0] / max(img.size)),
                    multiple_16(noise_size * img.size[1] / max(img.size))
                ), resample=Image.BILINEAR)),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        transformed_image = train_transforms_(image)
        return transformed_image
    
    def load_and_transform_cond_images(images):
        transformed_images = [cond_train_transforms(image) for image in images]
        concatenated_image = torch.cat(transformed_images, dim=1)
        return concatenated_image
    
    def load_and_transform_subject_images(images):
        transformed_images = [subject_cond_train_transforms(image) for image in images]
        concatenated_image = torch.cat(transformed_images, dim=1)
        return concatenated_image
    
    tokenizer_clip = tokenizer[0]
    tokenizer_t5 = tokenizer[1]

    def retrieve_prompt_embeds_from_disk(args, examples):
        captions = []
        for caption in examples["prompts"]:
            if isinstance(caption, str):
                if random.random() < 0.1:
                    captions.append(" ")  # 将文本设为空
                else:
                    captions.append(caption)
            elif isinstance(caption, list):
                raise NotImplementedError("list of captions not supported yet")
                # take a random caption if there are multiple
                if random.random() < 0.1:
                    captions.append(" ")
                else:
                    captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )

        all_prompt_embeds = [] 
        all_pooled_prompt_embeds = [] 
        for caption in captions: 
            if caption == " ": 
                prompt_file_name = "space_prompt.pth" 
            else: 
                prompt_file_name = "_".join(caption.split(" ")) + ".pth" 
            if osp.exists(osp.join(args.inference_embeds_dir, prompt_file_name)): 
                prompt_embeds = torch.load(osp.join(args.inference_embeds_dir, prompt_file_name), map_location="cpu") 
                pooled_prompt_embeds = prompt_embeds["pooled_prompt_embeds"] 
                prompt_embeds = prompt_embeds["prompt_embeds"] 
            else: 
                # raise FileNotFoundError(f"Prompt embeddings for '{caption}' not found in {args.inference_embeds_dir}. Please precompute and save them.") 
                prompt_embeds = torch.zeros((1, 77, 768))  # Placeholder tensor
                pooled_prompt_embeds = torch.zeros((1, 768))  # Placeholder tensor
            all_prompt_embeds.append(prompt_embeds.squeeze(0)) 
            all_pooled_prompt_embeds.append(pooled_prompt_embeds.squeeze(0)) 
        return all_prompt_embeds, all_pooled_prompt_embeds 


    def tokenize_prompt_clip_t5(examples):
        captions = []
        for caption in examples["prompts"]:
            if isinstance(caption, str):
                if random.random() < 0.1:
                    captions.append(" ")  # 将文本设为空
                else:
                    captions.append(caption)
            elif isinstance(caption, list):
                # take a random caption if there are multiple
                if random.random() < 0.1:
                    captions.append(" ")
                else:
                    captions.append(random.choice(caption))
            else:
                raise ValueError(
                    f"Caption column should contain either strings or lists of strings."
                )
        text_inputs = tokenizer_clip(
            captions,
            padding="max_length",
            max_length=77,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_1 = text_inputs.input_ids

        text_inputs = tokenizer_t5(
            captions,
            padding="max_length",
            max_length=512,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids_2 = text_inputs.input_ids
        return text_input_ids_1, text_input_ids_2

    def preprocess_train(examples):
        _examples = {}
        train_data_dir = osp.dirname(args.current_train_data_dir)
        if args.subject_column is not None:
            subject_images = [[load_image_safely(osp.join(train_data_dir, examples[column][i]), args.cond_size) for column in subject_columns] for i in range(len(examples[target_column]))]
            _examples["subject_pixel_values"] = [load_and_transform_subject_images(subject) for subject in subject_images]
        if args.spatial_column is not None:
            # this now has two conditions 
            spatial_images = [[load_image_safely(osp.join(train_data_dir, examples[column][i]), args.cond_size) for column in spatial_columns] for i in range(len(examples[target_column]))]
            _examples["cond_pixel_values"] = [load_and_transform_cond_images(spatial) for spatial in spatial_images]
        target_images = [load_image_safely(osp.join(train_data_dir, image_path), args.cond_size) for image_path in examples[target_column]]
        _examples["pixel_values"] = [train_transforms(image, noise_size) for image in target_images]
        _examples["PLACEHOLDER_prompts"] = examples["PLACEHOLDER_prompts"]
        subjects = examples["subjects"] 
        _examples["subjects"] = subjects 
        subjects_ = ["_".join(subject) for subject in subjects] # get the subject names with "_" instead of space 
        _examples["prompts"] = [] 
        # getting the prompts by replacing the PLACEHOLDER in the prompt with the actual subject names  
        for i in range(len(examples["subjects"])): 
            # replace the subjects string in the PLACEHOLDER 
            prompt = examples["PLACEHOLDER_prompts"][i] 
            placeholder_string = " and ".join(subjects[i]) 
            prompt = prompt.replace("PLACEHOLDER", placeholder_string) 
            _examples["prompts"].append(prompt) 
        _examples["prompt_embeds"], _examples["pooled_prompt_embeds"] = retrieve_prompt_embeds_from_disk(args, _examples) 
        # gettin the z passed cuboids segmentation mask 
        _examples["cuboids_segmasks"] = [] 

        def generous_resize_batch(masks, new_h, new_w):
            """
            masks: torch.Tensor of shape (B, H, W), values in {0,1}
            new_h, new_w: desired output size
            """
            B, H, W = masks.shape
            masks = masks.unsqueeze(1).float()   # -> (B,1,H,W)

            # Compute pooling kernel/stride
            kh = H // new_h
            kw = W // new_w
            assert H % new_h == 0 and W % new_w == 0, \
                "H and W must be divisible by new_h and new_w for exact block pooling"

            out = F.max_pool2d(masks, kernel_size=(kh, kw), stride=(kh, kw))
            return out.squeeze(1).byte()   # -> (B,new_h,new_w)

        for i in range(len(_examples["subjects"])): 
            segmasks_this_example = examples["cuboids_segmasks"][i] 
            # the name of the segmask is of the format "segmask_00<subject_idx>__<depth_value>.png" 
            depth_values_this_example = [osp.basename(segmasks_this_example[j]).split("__")[-1].split(".png")[0] for j in range(len(subjects[i]))] 
            depth_values_this_example = torch.as_tensor([float(depth) for depth in depth_values_this_example]) 
            assert len(segmasks_this_example) == len(subjects[i]), f"Number of segmentation masks {len(segmasks_this_example)} does not match number of subjects {len(subjects[i])} for example {i}" 
            segmasks_this_example = [cv2.imread(osp.join(train_data_dir, segmasks_this_example[j]), cv2.IMREAD_UNCHANGED) for j in range(len(subjects[i]))] 
            # segmasks_this_example = [cv2.resize(segmask, (32, 32), interpolation=cv2.INTER_NEAREST) for segmask in segmasks_this_example] 
            segmasks_this_example = [torch.as_tensor(segmask, dtype=torch.uint8) for segmask in segmasks_this_example] 
            segmasks_this_example = torch.stack(segmasks_this_example, dim=0) # (n_subjects, h, w) 
            mask = segmasks_this_example > 128   
            segmasks_this_example[mask] = 1 
            segmasks_this_example[~mask] = 0 
            segmasks_this_example = generous_resize_batch(segmasks_this_example, 32, 32) 
            assert segmasks_this_example.shape == (len(subjects[i]), 32, 32), f"Segmentation masks shape {segmasks_this_example.shape} does not match expected shape {(len(subjects[i]), 32, 32)} for example {i}" 
            # z_passed_segmask = do_z_pass(segmasks_this_example, depth_values_this_example) 
            # print(f"{z_passed_segmask.shape = }, {segmasks_this_example.shape = }")
            # _examples["cuboids_segmasks"].append(z_passed_segmask) 
            _examples["cuboids_segmasks"].append(segmasks_this_example) 

        _examples["token_ids_clip"], _examples["token_ids_t5"] = tokenize_prompt_clip_t5(_examples)
        _examples["call_ids"] = examples["call_ids"] 
        _examples["index"] = examples["index"] 

        return _examples

    if accelerator is not None:
        with accelerator.main_process_first():
            train_dataset = dataset["train"].with_transform(preprocess_train)
    else:
        train_dataset = dataset["train"].with_transform(preprocess_train)

    return train_dataset


def collate_fn(examples):
    if examples[0].get("cond_pixel_values") is not None:
        cond_pixel_values = torch.stack([example["cond_pixel_values"] for example in examples])
        cond_pixel_values = cond_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        cond_pixel_values = None
    if examples[0].get("subject_pixel_values") is not None: 
        subject_pixel_values = torch.stack([example["subject_pixel_values"] for example in examples])
        subject_pixel_values = subject_pixel_values.to(memory_format=torch.contiguous_format).float()
    else:
        subject_pixel_values = None

    target_pixel_values = torch.stack([example["pixel_values"] for example in examples])
    target_pixel_values = target_pixel_values.to(memory_format=torch.contiguous_format).float()
    token_ids_clip = torch.stack([torch.tensor(example["token_ids_clip"]) for example in examples])
    token_ids_t5 = torch.stack([torch.tensor(example["token_ids_t5"]) for example in examples])
    prompt_embeds = torch.stack([example["prompt_embeds"] for example in examples], dim=0)
    pooled_prompt_embeds = torch.stack([example["pooled_prompt_embeds"] for example in examples], dim=0) 
    prompts = [example["prompts"] for example in examples] 
    call_ids = [example["call_ids"] for example in examples] 
    cuboids_segmasks = [example["cuboids_segmasks"] for example in examples] if examples[0].get("cuboids_segmasks") is not None else None 
    indices = [example["index"] for example in examples]  # Add this line

    return {
        "cond_pixel_values": cond_pixel_values,
        "subject_pixel_values": subject_pixel_values,
        "pixel_values": target_pixel_values,
        "text_ids_1": token_ids_clip,
        "text_ids_2": token_ids_t5,
        "prompt_embeds": prompt_embeds,
        "pooled_prompt_embeds": pooled_prompt_embeds,
        "prompts": prompts, 
        "call_ids": call_ids, 
        "cuboids_segmasks": cuboids_segmasks, 
        "index": indices, 
    }