"""
Helper methods to evaluate Gemini models on VideoNet.

Please note that the Gemini API requires videos to first be uploaded to the "Gemini File API" before they can be used in a query.
The File API has a hard limit on the size of files stored in it at any time (20 GB total). Files also expire after 48 hours.
We do our best to obscure the difficulties of using the File API, primarily by creating and maintaining a "cache" of files already in the File API.
"""
from datetime import datetime, timedelta, timezone
from . import shared_utils
from pathlib import Path
from google import genai
from tqdm import tqdm
import pickle, os, time

GEMINI_CACHE_PATH = Path('results/gemini_cache.pkl')

def _load_cache() -> dict[str, genai.types.File]:
    """
    If a cache is found, loads it in and returns it.
    This cache maps video UUIDs to Gemini File objects.
    """
    cache = GEMINI_CACHE_PATH
    if cache.is_file():
        with open(cache, 'rb') as f:
            uuid_to_gfile: dict = pickle.load(f)
            return uuid_to_gfile
    else:
        print('\n\tINFO: no cache found of videos that were previously uploaded to the Gemini File API. will upload them as-we-go\n')
        return {}

def validate_config(model: str, max_frames: int, fps: int, reasoning_level: str, use_flash_attn: bool = False) -> dict[str, int]:
    if use_flash_attn:
        print(f'\n\tERROR: {model} is a proprietary (API-only) model. please remove the `--enable-flash-attn` flag.\n')
        exit(1)
        
    if max_frames is not None:
        print('\n\tERROR: we do not allow users to modify the max # of frames sampled for Gemini models.\n')
        exit(1)

    # default to 1fps for Gemini models, unless user specifies otherwise
    fps = 1 if fps is None else fps
    print(f'\n\tINFO: Will sample {fps} frames per second.')
    
    # reasoning level
    if reasoning_level is None:
        print(f'\n\tINFO: Defaulting to "high" reasoning level. You can override this via the `--reasoning-level` flag.')
        reasoning_level = 'high'
    
    if model.startswith('gemini-3.1-pro') and reasoning_level not in ['low', 'medium', 'high']:
        print(f'\n\tERROR: Invalid reasoning level choice. See options at https://ai.google.dev/gemini-api/docs/gemini-3#thinking_level \n')
        exit(1)
    if model.startswith('gemini-3-flash') and reasoning_level not in ['minimal', 'low', 'medium', 'high']:
        print(f'\n\tERROR: Invalid reasoning level choice. See options at https://ai.google.dev/gemini-api/docs/gemini-3#thinking_level \n')
        exit(1)

    return {'fps': fps, 'reasoning_level': reasoning_level, 'version': model}

def _verify_cache(uuid_to_gfile: dict[str, genai.types.File], client: genai.client.Client):
    """
    Modifies the provided dict in-place, deleting any entries associated with files that have expired (or are about to expire).
    """
    min_hours_til_expiration = 8
    initial_uuids: list[str] = list(uuid_to_gfile.keys())
    now = datetime.now(timezone.utc)
    print('\n\tINFO: Found a cache of videos that have already been uploaded to the Gemini File API; we will verify if they are stil active.\n')
    for unique in tqdm(initial_uuids, desc="Verifying Gemini File API cache"):
        gfile: genai.types.File = uuid_to_gfile[unique]
        
        # check if the file object (*as written to the cache*) is expired
        if gfile.expiration_time - timedelta(hours=min_hours_til_expiration) < now:
            del uuid_to_gfile[unique]
            continue
        
        # ensure that Gemini's copy of the file object still exists
        try:
            gf = client.files.get(name=gfile.name)
        except genai.errors.ClientError as e:
            del uuid_to_gfile[unique]
            continue
        except Exception as e:
            print('\n\tERROR: unexpected issue when surveying if files previously uploaded to Gemini File API still exist\n')
            raise e
        
        # ensure that Gemini's copy of the file object matches our cache's copy (in terms of expiration date) & update our cache's copy with Gemini's copy
        if gf.expiration_time - timedelta(hours=min_hours_til_expiration) < now or gf.state != genai.types.FileState.ACTIVE:
            del uuid_to_gfile[unique]
        else:
            uuid_to_gfile[unique] = gf
    print(f'\n\tINFO: Cache of pre-existing videos on the Gemini File API has {len(uuid_to_gfile)} valid entries.')

def init_model(version, use_flash_attn=False):
    assert use_flash_attn == False, 'cannot specify Flash Attention on API models'
    
    # establish connection to Gemini API
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key:
        print('\n\tERROR: please set the "GEMINI_API_KEY" env variable.\n')
        exit(1)
    client = genai.Client(api_key=api_key)

    # if applicable, load in & verify cache of existing videos in the Gemini File API
    if (cache := _load_cache()):
        _verify_cache(cache, client)
    
    return client, cache

def _get_file_obj(unique: str, abs_path: str, client: genai.client.Client, cache: dict[str, genai.types.File]) -> genai.types.File:
    """
    Given the UUID of a video (and absolute path to local copy of that video), 
    returns the corresponding File object in the Gemini Files API.
    """
    if unique in cache:
        return cache[unique]
    
    # upload file to Gemini Files API
    file_obj = client.files.upload(file=abs_path, config={'display_name': unique})
    gfile_name = file_obj.name
    
    # wait for increasing amounts of time, then check if file is in "ACTIVE" state
    for delay in (2, 5, 10, 30, 90):
        time.sleep(delay)
        file_obj = client.files.get(name=gfile_name)
        if file_obj.state == genai.types.FileState.ACTIVE:
            with open(GEMINI_CACHE_PATH, 'wb') as f:
                pickle.dump(cache, f)
            return file_obj
        if file_obj.state == genai.types.FileState.FAILED:
            raise RuntimeError(f'\tERROR: the Gemini Files API file for the video at {abs_path} has a FAILED state')

    raise TimeoutError(f'\tERROR: unable to upload video at {abs_path} to Gemini File API')

def _video_to_part(video: str, fps: int, client: genai.client.Client, cache: dict[str, genai.types.File]):
    unique = video.split('.')[0]
    gfile = _get_file_obj(unique, shared_utils.video_to_path(video), client, cache)
    if unique not in cache:
        cache[unique] = gfile
    return genai.types.Part(
        file_data=genai.types.FileData(
            file_uri=gfile.uri,
            mime_type='video/mp4'
        ),
        video_metadata=genai.types.VideoMetadata(fps=fps)
    )
    
def _text_to_part(text: str) -> genai.types.Part:
    return genai.types.Part(text=text)    

def prepare_parts(question: dict, k: int, fps: int, client: genai.client.Client, cache: dict) -> list[genai.types.Part]:
    in_context_text, in_context_videos, test_text, test_video = shared_utils.breakdown_question(question, k)
    
    test_text_part = _text_to_part(test_text)
    test_video_part = _video_to_part(test_video, fps, client, cache)
    
    if k == 0:
        return [ test_text_part, test_video_part ]
    else:
        in_context_text_part = _text_to_part(in_context_text)
        in_context_video_parts = [_video_to_part(v, fps, client, cache) for v in in_context_videos]
        return [ in_context_text_part, *in_context_video_parts, test_text_part, test_video_part]

def generate_response(key: str, question: dict, k: int, config: dict, client: genai.client.Client, cache: dict[str, genai.types.File]) -> str | None:
    model_version, reasoning_level, fps = config['version'], config['reasoning_level'], config['fps']
    parts: list[dict] = prepare_parts(question, k, fps, client, cache)
    
    response = client.models.generate_content(
        model=model_version,
        contents=genai.types.Content(
            parts=parts
        ),
        config=genai.types.GenerateContentConfig(
            thinking_config=genai.types.ThinkingConfig(
                thinking_level=reasoning_level
            )
        )
    )
    
    return response.text