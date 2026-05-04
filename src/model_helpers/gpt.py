import cv2, base64, os, math, signal
from . import shared_utils
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI

def init_model(version, use_flash_attn=False):
    assert use_flash_attn == False, 'cannot specify Flash Attention on API models'
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        print('\n\tERROR: please set the "OPENAI_API_KEY" env variable.\n')
        exit(1)
    client = OpenAI(api_key=api_key)
    return client, None

def validate_config(model: str, max_frames: int, fps: int, reasoning_level: str, use_flash_attn: bool = False) -> dict[str, int]:
    if use_flash_attn:
        print(f'\n\tERROR: {model} is a proprietary (API-only) model. please remove the `--enable-flash-attn` flag.\n')
        exit(1)
        
    # GPT models have max context windows
    # this limits how many frames we can send before the API rejects our requests
    if model.startswith('gpt-5.4'):
        mandated_max = 256
    elif model.startswith('gpt-5'):
        mandated_max = 56
    else:
        print(f'\n\tERROR: {model} has not been properly setup. please edit the `validate_config` function in `gpt.py` to have a `mandated_max` for {model}. also update the `reasoning_level` checker in `validate_config`.')
        exit(1)
    
    if max_frames and max_frames > mandated_max:
        print(f'\n\tWARNING: you requested a `--max-frames` of {max_frames}. However, {model} can only reliably take {mandated_max} frames, so we are clamping `--max-frames` down to {mandated_max}.')    
    
    if max_frames is None or max_frames > mandated_max:
        max_frames = mandated_max

    # default to 1fps for GPT models, unless user specifies otherwise
    fps = 1 if fps is None else fps
    print(f'\n\tINFO: Will sample {fps} frames per second.')
    
    # reasoning level
    if reasoning_level is None:
        print(f'\n\tINFO: Defaulting to "medium" reasoning level. You can override this via the `--reasoning-level` flag.')
        reasoning_level = 'medium'
    
    if model.startswith('gpt-5.4'):
        if reasoning_level not in ['none', 'low', 'medium', 'high', 'xhigh']:
            print(f'\n\tERROR: Invalid reasoning level choice. See options at https://developers.openai.com/api/docs/models/gpt-5.4 \n')
            exit(1)
    elif model.startswith('gpt-5'):
        if reasoning_level not in ['minimal', 'low', 'medium', 'high']:
            print(f'\n\tERROR: Invalid reasoning level choice. See options at https://developers.openai.com/api/docs/models/gpt-5 \n')
            exit(1)

    return {'max_frames': max_frames, 'fps': fps, 'reasoning_level': reasoning_level, 'version': model}

def extract_frames(vid_paths: list[str]) -> list[tuple[list[str], float]]:
    """
    Extracts 2-tuple of frames (as list) and FPS (as float) from the videos located at `vid_paths` using OpenCV.
    """
    def extract(vid_path) -> tuple[list[str], float]:
        video = cv2.VideoCapture(vid_path)
        fps = video.get(cv2.CAP_PROP_FPS)
        if not (fps > 0):
            print(f'\n\tERROR: corrupt video at {vid_path}\t')
            exit(1)
        b64_frames = []
        while video.isOpened():
            success, frame = video.read()
            if not success: break
            _, buffer = cv2.imencode(".jpg", frame)
            b64_frames.append(base64.b64encode(buffer).decode('utf-8'))
        video.release()
        return (b64_frames, fps)

    results: list[tuple[list[str], float]] = []
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(extract, vid_paths))
    return results

def frames_to_inputs(idx: int, frames: list[str]) -> list:
    """
    Given a list of base64-encoded frames, prepares them for input to GPT.
    Prepends a text label "Frames from Video #{idx}" to help model understand the input.
    """
    frame_urls = [f"data:image/jpeg;base64,{frame}" for frame in frames]
    return [{"type": "input_text", "text": (f"Frames from Video #{idx}:")}] \
        + [{"type": "input_image", "image_url": frame_url} for frame_url in frame_urls]
        
def text_to_input(text: str) -> dict:
    return {
        "type": "input_text",
        "text": text
    }
        
def sample_relevant_frames_stride(vid_label: int, all_frames: list[str], sampling_rate: int) -> list[dict]:
    """
    Samples every `sampling_rate` frames from video whose frames are `all_frames` and whose UUID is `rel_uuid`.
    Returns prompt content for Video #n where n is `vid_label`.
    """
    rel_frames = all_frames[:: sampling_rate]
    return frames_to_inputs(vid_label, rel_frames)

def sample_relevant_frames_fixed(vid_label: int, all_frames: list[str], num_samples: int) -> list[dict]:
    """
    Samples `num_samples` frames uniformly from video whose frames are `all_frames` and whose UUID is `rel_uuid`.
    Returns prompt content for Video #n where n is `vid_label`.
    """
    if len(all_frames) <= num_samples:
        indices = list(range(len(all_frames)))
    else:
        indices = np.linspace(0, len(all_frames) - 1, num_samples, dtype=int)

    rel_frames = [all_frames[i] for i in indices] 
    return frames_to_inputs(vid_label, rel_frames)

def prepare_content(question: list[dict], k: int, config: dict) -> list[dict]:
    DEFAULT_FPS, MAX_FRAMES = config['fps'], config['max_frames']
    in_context_text, in_context_videos, test_text, test_video = shared_utils.breakdown_question(question, k)
    in_context_video_paths: list[Path] = [shared_utils.video_to_path(ic) for ic in in_context_videos]
    test_video_path: Path = shared_utils.video_to_path(test_video)
    
    # in code below, `ic` refers to the in-context clips; `tc` refers to the test clip
    # let's extract frames from the videos & get the videos' FPS. from this, we compute their duration
    ic_info: list[tuple[list[str], float]] = extract_frames(in_context_video_paths)
    total_ic_seconds: int = sum([ math.ceil(len(ic_info[i][0]) / ic_info[i][1]) for i in range(k) ])   # num. seconds = total frames / video's fps
    tc_info: tuple[list[str], float] = extract_frames([test_video_path])[0]
    tc_seconds: int = math.ceil(len(tc_info[0]) / tc_info[1])
    total_seconds = total_ic_seconds + tc_seconds
    
    # if fps sampling would allows us to stay in context length, just do that
    if total_seconds * DEFAULT_FPS <= MAX_FRAMES:
        test_video_content: list[dict] = sample_relevant_frames_stride(k + 1, tc_info[0], math.ceil(tc_info[1] / DEFAULT_FPS))
        in_context_video_content: list[dict] = []
        for j, icf in enumerate(ic_info):
            sample_every = math.ceil(icf[1] / DEFAULT_FPS)  # video FPS / our sampling rate; so if 30fps video and we sample 2 frames per second, `sample_every`=15
            in_context_video_content.extend(sample_relevant_frames_stride(j + 1, icf[0], sample_every))

    # otherwise, uniformly sample MAX_FRAMES / NUM_VIDEOS frames from each video in the prompt
    else:
        frames_per_video = math.ceil(MAX_FRAMES / (k + 1))
        test_video_content: list[dict] = sample_relevant_frames_fixed(k + 1, tc_info[0], frames_per_video)
        in_context_video_content = []
        for j, icf in enumerate(ic_info):
            in_context_video_content.extend(sample_relevant_frames_fixed(j + 1, icf[0], frames_per_video))
    # last (and easiest) step is to convert simple text strings into a "text_input" dict for OpenAI API
    if k > 0: in_context_text_content = text_to_input(in_context_text)
    test_text_content = text_to_input(test_text)
    
    if k == 0:
        return [test_text_content, *test_video_content]
    else:
        return [in_context_text_content, *in_context_video_content, test_text_content, *test_video_content]

def generate_response(key: str, question: dict, k: int, config: dict, client: OpenAI, processor=None, timeout=180) -> str | None:
    content: list[dict] = prepare_content(question, k, config)
    prompt = [{
        "role": "user",
        "content": content
    }]
    
    # Implement timeout (default is 5 minutes per question)
    signal.signal(signal.SIGALRM, shared_utils.timeout_handler)
    signal.alarm(timeout)
    try: 
        response = client.responses.create(
            model=config['version'],
            input=prompt,
            reasoning={"effort" : config['reasoning_level']}
        )
        return response.output_text
    finally:
        signal.alarm(0)