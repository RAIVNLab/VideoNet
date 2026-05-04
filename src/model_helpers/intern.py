import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoTokenizer, AutoModel
from decord import VideoReader, cpu
from . import shared_utils
from pathlib import Path
from PIL import Image
import torch, os, signal
import numpy as np

#### InternVL-specific video preprocessing functions; largely copied from InternVL repo

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
	MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
	transform = T.Compose([
		T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
		T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
		T.ToTensor(),
		T.Normalize(mean=MEAN, std=STD)
	])
	return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
	best_ratio_diff = float('inf')
	best_ratio = (1, 1)
	area = width * height
	for ratio in target_ratios:
		target_aspect_ratio = ratio[0] / ratio[1]
		ratio_diff = abs(aspect_ratio - target_aspect_ratio)
		if ratio_diff < best_ratio_diff:
			best_ratio_diff = ratio_diff
			best_ratio = ratio
		elif ratio_diff == best_ratio_diff:
			if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
				best_ratio = ratio
	return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=6, image_size=448, use_thumbnail=True):
	orig_width, orig_height = image.size
	aspect_ratio = orig_width / orig_height

	# calculate the existing image aspect ratio
	target_ratios = set(
		(i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
		i * j <= max_num and i * j >= min_num)
	target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

	# find the closest aspect ratio to the target
	target_aspect_ratio = find_closest_aspect_ratio(
		aspect_ratio, target_ratios, orig_width, orig_height, image_size)

	# calculate the target width and height
	target_width = image_size * target_aspect_ratio[0]
	target_height = image_size * target_aspect_ratio[1]
	blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

 	# resize the image
	resized_img = image.resize((target_width, target_height))
	processed_images = []
	for i in range(blocks):
		box = (
			(i % (target_width // image_size)) * image_size,
			(i // (target_width // image_size)) * image_size,
			((i % (target_width // image_size)) + 1) * image_size,
			((i // (target_width // image_size)) + 1) * image_size
		)
		# split the image
		split_img = resized_img.crop(box)
		processed_images.append(split_img)
	assert len(processed_images) == blocks
	if use_thumbnail and len(processed_images) != 1:
		thumbnail_img = image.resize((image_size, image_size))
		processed_images.append(thumbnail_img)
	return processed_images

def get_index(bound, fps, max_frame, first_idx=0, num_segments=32):
	if bound:
		start, end = bound[0], bound[1]
	else:
		start, end = -100000, 100000
	start_idx = max(first_idx, round(start * fps))
	end_idx = min(round(end * fps), max_frame)
	seg_size = float(end_idx - start_idx) / num_segments
	frame_indices = np.array([
		int(start_idx + (seg_size / 2) + np.round(seg_size * idx))
		for idx in range(num_segments)
	])
	return frame_indices

def load_video(video_path, bound=None, input_size=448, max_num=1, num_segments=32):
	vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
	max_frame = len(vr) - 1
	fps = float(vr.get_avg_fps())

	pixel_values_list, num_patches_list = [], []
	transform = build_transform(input_size=input_size)
	frame_indices = get_index(bound, fps, max_frame, first_idx=0, num_segments=num_segments)
	for frame_index in frame_indices:
		img = Image.fromarray(vr[frame_index].asnumpy()).convert('RGB')
		img = dynamic_preprocess(img, image_size=input_size, use_thumbnail=True, max_num=max_num)
		pixel_values = [transform(tile) for tile in img]
		pixel_values = torch.stack(pixel_values)
		num_patches_list.append(pixel_values.shape[0])
		pixel_values_list.append(pixel_values)
	pixel_values = torch.cat(pixel_values_list)
	return pixel_values, num_patches_list

def process_videos(vid_paths: list[str], frames_per_video: int) -> tuple[str, torch.Tensor, list[int]]:
	combined_pixel_values: list[torch.Tensor] = []
	combined_num_patches_list: list[int] = []
	for vid_path in vid_paths:
		curr_pixel_values, curr_num_patches_list = load_video(vid_path, max_num=1, num_segments=frames_per_video)
		curr_pixel_values = curr_pixel_values.to(torch.bfloat16).cuda()
		combined_pixel_values.append(curr_pixel_values)
		combined_num_patches_list.extend(curr_num_patches_list)
	return torch.cat(combined_pixel_values, dim=0), combined_num_patches_list

#### Custom preprocessing for VideoNet integration; written by VideoNet authors

def validate_config(model: str, max_frames: int, fps: int, reasoning_level: str, use_flash_attn: bool = False) -> dict[str, int]:
    if reasoning_level is not None:
        print('\n\tERROR: InternVL models do not support specifying the reasoning level. \
            Please remove the `--reasoning-level` flag.\n')
        exit(1)
        
    if fps is not None:
        print('\n\tERROR: For InternVL models, you must specify a uniform number of frames to sample via `--max-frames`. \
            Please remove the `--fps` flag and control sampling via `--max-frames` instead.\n')
        exit(1)
    
    if max_frames is None:
        max_frames = 48
    
    if max_frames > 64:
        print('\n\tERROR: InternVL models cannot be used with a `--max-frames` above 64.\n')
        exit(1)
    else:
        print(f'\n\tINFO: Will uniformly sample {max_frames} frames across all the video inputs.')
    
    if use_flash_attn:
        print(f'\n\tINFO: Attempting to use Flash Attention 2, since the `--enable-flash-attn` flag was provided.')

    generation_config = dict(max_new_tokens=12000, do_sample=False, eos_token_id=151645, pad_token_id=151645)

    return {'version': model, 'max_frames': max_frames, 'use_flash_attn': use_flash_attn, 'generation_config': generation_config}

def init_model(model_version: str, use_flash_attn: bool = False) -> tuple[AutoModel, AutoTokenizer]:
    # ensure user's GPU supports CUDA & BF16
    if not torch.cuda.is_available():
        print(f'\n\tERROR: cuda is NOT available on your device, but is required for {model_version}.\n')
    if not torch.cuda.is_bf16_supported():
        print(f'\n\tERROR: bfloat16 is NOT supported on your device, but is required for {model_version}.\n')
        exit(1)
    
    # specify where HF downloads models to
    if (models_dir := os.getenv('VN_MODELS_DIR')) is None:
        models_dir = Path('models').resolve()
    
    # setup model and proceessor
    print()
    model_name = 'OpenGVLab/' + model_version
    model = AutoModel.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        cache_dir=models_dir
    ).eval().cuda()
        
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        use_fast=False, 
        cache_dir=models_dir
    )
    
    return model, tokenizer

def prepare_content(question: list[dict], k: int, config: dict) -> tuple:
    MAX_INTERNVL_FRAMES = config['max_frames']
    in_context_text, in_context_videos, test_text, test_video = shared_utils.breakdown_question(question, k)
    
    num_videos = k + 1
    frames_per_video = MAX_INTERNVL_FRAMES // num_videos
    all_videos: list[str] = in_context_videos + [test_video]
    all_videos_absolute_paths: list[Path] = [shared_utils.video_to_path(vid).resolve() for vid in all_videos]
    all_videos_absolute_paths: list[str] = [str(abs_path) for abs_path in all_videos_absolute_paths]    # InternVL codebase requires paths via strings
    pixel_values, num_patches_list = process_videos(all_videos_absolute_paths, frames_per_video)
    video_prefixes = [''.join([f'Frame{j+1}: <image>\n' for j in range(i*frames_per_video,(i+1)*frames_per_video)]) for i in range(num_videos)]
    
    if k == 0:
        message = ''.join([
            video_prefixes[0],
            test_text
        ])
    else:
        in_context_video_content = []
        for i in range(k): 
            in_context_video_content.extend( [f'\nVideo{i+1}:\n', video_prefixes[i]] )
        message = ''.join([
			*in_context_video_content,
			in_context_text + "\n",
			f"\nVideo{num_videos}:\n", video_prefixes[-1],
            test_text
		])
        
    return message, pixel_values, num_patches_list
        

def generate_response(key: str, question: dict, k: int, config: dict, model, tokenizer, timeout=300) -> str | None:
    generation_config = config['generation_config']
    message, pixel_values, num_patches_list = prepare_content(question, k, config)
    
    # Implement timeout (default is 5 minutes per question)
    signal.signal(signal.SIGALRM, shared_utils.timeout_handler)
    signal.alarm(timeout)
    try:
        response, history = model.chat(tokenizer, pixel_values, message, generation_config, num_patches_list=num_patches_list, history=None, return_history=True)
        return response
    finally:
        signal.alarm(0)