from transformers import AutoModelForImageTextToText, AutoProcessor
from . import shared_utils
from pathlib import Path
import torch, os, signal

def validate_config(model: str, max_frames: int, fps: int, reasoning_level: str, use_flash_attn: bool = False) -> dict[str, int]:
    if reasoning_level is not None:
        print('\n\tERROR: Molmo2 models do not support specifying the reasoning level. \
            Please remove the `--reasoning-level` flag.\n')
        exit(1)
    
    if max_frames is not None:
        print('\n\tERROR: we do not allow users to modify the max # of frames sampled for Molmo2 models.\n')
        exit(1)
    
    if use_flash_attn:
        print(f'\n\tERROR: Molmo2 models do not support Flash Attention.\n')
        exit(1)
        
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16
    print(f'\n\tINFO: Serving the model with {dtype} weights\n')
    
    return {'version': model, 'dtype': dtype}

def init_model(model_version: str, use_flash_attn: bool = False):
    if not torch.cuda.is_available():
        print('\n\tWARNING: cuda is not available on your device ... this will likely cause the eval run to crash!')
    
    # specify where HF downloads models to
    if (models_dir := os.getenv('VN_MODELS_DIR')) is None:
        models_dir = Path('models').resolve()
    
    model_name = 'allenai/' + model_version
    if use_flash_attn:
        model = AutoModelForImageTextToText.from_pretrained(
	        model_name,
            trust_remote_code=True,
	        dtype="auto", 
	        # attn_implementation="flash_attention_2",
	        device_map="auto",
            cache_dir=models_dir
        )
    else:
        model = AutoModelForImageTextToText.from_pretrained(
	        model_name, 
            trust_remote_code=True,
            dtype="auto",
            device_map="auto",
            cache_dir=models_dir
        )
    
    processor = AutoProcessor.from_pretrained(
        model_name, 
        trust_remote_code=True, 
        padding_size="left", 
        cache_dir=models_dir
    )
    
    return model, processor

def prepare_content(question: list[dict], k: int, config: dict) -> list[dict]:
    in_context_text, in_context_videos, test_text, test_video = shared_utils.breakdown_question(question, k)
    in_context_video_absolute_paths: list[Path] = [shared_utils.video_to_path(ic).resolve() for ic in in_context_videos]
    test_video_absolute_path: Path = shared_utils.video_to_path(test_video).resolve()
    
    in_context_video_content = [
		{ "type": "video", "video": str(abs_path) } 
		for abs_path in in_context_video_absolute_paths
	]
    
    test_video_content = {
		"type": "video", 
		"video": str(test_video_absolute_path)
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
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, padding=True)
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        
        # generate ouput
        with torch.inference_mode(), torch.autocast("cuda", dtype=config['dtype']):
            output = model.generate(**inputs, max_new_tokens=1024)
        
        # decode output
        generated_tokens = output[0, inputs['input_ids'].size(1):]
        output_text = processor.decode(generated_tokens, skip_special_tokens=True)
        return output_text
    finally:
        signal.alarm(0)