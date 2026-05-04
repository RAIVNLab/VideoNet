from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info
from . import shared_utils
from pathlib import Path
import torch, os, signal

def validate_config(model: str, max_frames: int, fps: int, reasoning_level: str, use_flash_attn: bool = False) -> dict[str, int]:
    if reasoning_level is not None:
        print('\n\tERROR: QwenVL models do not support specifying the reasoning level. \
            Please remove the `--reasoning-level` flag.\n')
        exit(1)
    
    if max_frames is not None:
        print('\n\tERROR: we do not allow users to modify the max # of frames sampled for QwenVL models. \
            To do this, please manually change the Qwen3-VL processor. Nonetheless, please \
            remove the `--max-frames` flag.')
        exit(1)
    
    if use_flash_attn:
        print(f'\n\tINFO: Attempting to use Flash Attention 2, since the `--enable-flash-attn` flag was provided.')
    
    if fps is None:
        if model.startswith('Qwen3-VL'):
            fps = 2
        elif model.startswith('Qwen2.5-VL'):
            fps = 1
    print(f'\n\tINFO: Will sample {fps} frames per second from video inputs.')
    
    return {'version': model, 'fps': fps, 'use_flash_attn': use_flash_attn}

def init_model(model_version: str, use_flash_attn: bool = False):
    if not torch.cuda.is_available():
        print('\n\tWARNING: cuda is not available on your device ... this will likely cause the eval run to crash!')
    
    # specify where HF downloads models to
    if (models_dir := os.getenv('VN_MODELS_DIR')) is None:
        models_dir = Path('models').resolve()
    
    model_name = 'Qwen/' + model_version
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f'\n\tINFO: Serving the model with {str(dtype)} weights.\n')
    
    if use_flash_attn:
        model = AutoModelForImageTextToText.from_pretrained(
	        model_name,
	        dtype=dtype, 
	        attn_implementation="flash_attention_2",
	        device_map="auto",
            cache_dir=models_dir
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
	        model_name, 
            dtype=dtype,
            device_map="auto",
            cache_dir=models_dir
        )
    
    processor = AutoProcessor.from_pretrained(model_name, cache_dir=models_dir)
    return model, processor

def prepare_content(question: list[dict], k: int, config: dict) -> list[dict]:
    QWEN_FPS = config['fps']
    in_context_text, in_context_videos, test_text, test_video = shared_utils.breakdown_question(question, k)
    in_context_video_absolute_paths: list[Path] = [shared_utils.video_to_path(ic).resolve() for ic in in_context_videos]
    test_video_absolute_path: Path = shared_utils.video_to_path(test_video).resolve()
    
    in_context_video_content = [
		{ "type": "video", "video": f"file://{abs_path}", "fps": QWEN_FPS } 
		for abs_path in in_context_video_absolute_paths
	]
    
    test_video_content = {
		"type": "video", 
		"video": f"file://{test_video_absolute_path}", 
        "fps": QWEN_FPS
	}
    
    if k == 0:
        messages = [{
            "role": "user",
            "content": [
                { "type": "text", "text": test_text },
                test_video_content
            ]
        }]
    else:
        messages = [{
			"role": "user",
			"content": [
				{ "type": "text", "text": in_context_text },
				*in_context_video_content,
				{ "type": "text", "text": test_text },
				test_video_content
			]
		}]
    
    return messages
        
def generate_response(key: str, question: dict, k: int, config: dict, model, processor, timeout=300) -> str | None:
    messages: list[dict] = prepare_content(question, k, config)
    
    # Implement timeout (default is 5 minutes per question)
    signal.signal(signal.SIGALRM, shared_utils.timeout_handler)
    signal.alarm(timeout)
    try:
        # prepare inputs w/ tokenizer
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, add_vision_id=True)
        images, videos, video_kwargs = process_vision_info(messages, image_patch_size=16, return_video_kwargs=True, return_video_metadata=True)
        if videos is not None:
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
        else:
            video_metadatas = None
        inputs = processor(text=text, images=images, videos=videos, video_metadata=video_metadatas, return_tensors="pt", do_resize=False, **video_kwargs)
        inputs = inputs.to(model.device)
    
        # generate model output & decode
        generated_ids = model.generate(**inputs, max_new_tokens=32768)
        generated_ids_trimmed = [ out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids) ]
        output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output_text
    finally:
        signal.alarm(0)