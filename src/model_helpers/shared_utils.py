"""
"""
import os
from pathlib import Path

def timeout_handler(signum, frame):
	raise TimeoutError('inference timed out')

def video_to_path(video: str):
    """
    video filename --> pathlib Path object for that video
    """
    videos_dir = os.getenv('VN_VIDEOS_DIR', 'videos')
    return Path(videos_dir) / video

def breakdown_question(content, k) -> tuple[None | str, list, str, str]:
    """
    Returns a 4 tuple:
        1. a string (or None if k=0). looks like: "the following k videos show ..."
        2. a list of strings. each string is filename of an in-context video. looks like: ["abc.mp4", "def.mp4"] for k=2
        3. a string. looks like: " ... does it also contain the action ? ..."
        4. a string. this is the filename of the test video. looks like "xyz.mp4"
    """
    if k == 0:
        # in the 0-shot case, we have a list of length 2:
        #   1. a text dict with the text part of the question ("Does the following vidoe contain ...")
        #   2. a video dict with the test video
        assert len(content) == 2, 'you have a malformed benchmark'
        test_text = content[0]['text']
        test_video = content[1]['video']
        return None, [], test_text, test_video
    else:
        # in the k-shot case (for k != 0), we have a list with:
        #   1. one text dict of "in_context_text" saying "The following k videos show ..."
        #   2. k video dicts with the in-context videos
        #   3. one text dict of "test_text" saying "Now consider the following video. Does it also contain ..."
        #   4. one video dict of the test video
        assert len(content) == 3 + k, 'you have a malformed benchmark'
        in_context_text = content[0]['text']
        in_context_videos = [d['video'] for d in content[1:1+k]]
        test_text = content[-2]['text']
        test_video = content[-1]['video']
        return in_context_text, in_context_videos, test_text, test_video