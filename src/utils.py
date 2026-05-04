import json, os
from pathlib import Path

def is_empty(path: Path) -> bool:
    return not any(path.iterdir())

def confirm_videos_and_load_benchmark(k: int, mcq_test: bool, mcq_val: bool) -> list[dict]:
    benchmarks_dir = Path('benchmarks')
    if not benchmarks_dir.is_dir():
        print('\n\tERROR: your repository is missing the `benchmarks` directory!\n')
        exit(1)
    benchmark_file = benchmarks_dir / ('mcq_test.jsonl' if mcq_test else 'mcq_val.jsonl' if mcq_val else f'binary_{k}shot.jsonl')
    if not benchmark_file.is_file():
        print(f'\n\tERROR: your repistory is missing the {k}-shot benchmark JSONL file! it should be located at {benchmark_file}\n')
        exit(1)
    
    with open(benchmark_file, 'r') as f:
        lines = f.readlines()
    jsons = [json.loads(l) for l in lines]
    all_uuids = {Path(d['video']).stem for j in jsons for d in j['question'] if d['type'] == 'video'}
    
    videos_dir = Path(os.getenv('VN_VIDEOS_DIR', 'videos'))
    if not videos_dir.is_dir():
        print(f'\n\tERROR: expected the benchmark videos to be located in {videos_dir.resolve()}, but no directory existed there. please download the videos, or update the "VN_VIDEOS_DIR" environment variable in your shell to point to where the videos are stored.\n')
        exit(1)

    fail = False
    for unique in all_uuids:
        video_file = videos_dir / f'{unique}.mp4'
        if not video_file.is_file():
            print(f'\n\tERROR: expected video for {unique} at {video_file.resolve()} but could not find it!')
            fail = True
    if fail:
        exit(1)
    
    return jsons

def load_existing_eval(eval_file: str | None) -> dict:
    if not eval_file: return {}
    if not Path(eval_file).is_file():
        print(f'\n\tERROR: you specified an existing eval run at {eval_file}, but no file was found there\n')
        exit(1)
    try:
        with open(eval_file, 'r') as f:
            retval = json.load(f)
        print(f'\n\tINFO: Loading in existing eval at {eval_file} which has {len(retval)} of the benchmark questions completed.')
        return retval
    except Exception as e:
        print(f'\n\tERROR: unable to load in JSON file from previous eval run at {eval_file} ... see below')
        print(str(e))
        exit(1)
        
def extract_prediction(text: str, mcq: bool) -> str:
    """
    Parses model output and returns "yes" or "no" if possible.
    If model output cannot be parsed, returns the entire output (with some simplification / stripping).
    """
    pred = text.splitlines()[-1].strip().lower()
    pred = pred.replace('*', '').replace('#', '').replace(',', '').replace('.', '').replace(':', '')
    if not mcq:
        if pred == "yes" or pred == "no": return pred
        parts = pred.split(' ')
        if len(parts) > 1:
            first, last = parts[0], parts[-1]
            if first == 'yes' or first == 'no': return first
            elif last == 'yes' or last == 'no': return last
        if "boxed{yes}" in pred: return "yes"
        if "boxed{no}" in pred: return "no"
        if "does not show" in pred or "does not depict" in pred: return "no"
        if "the answer is yes" in pred: return "yes"
        if 'is "yes"' in pred: return "yes"
        if "is 'yes'" in pred: return "yes"
        if 'is "no"' in pred: return "no"
        if "is 'no'" in pred: return "no"
    else:
        pred = pred.upper()
        if pred in {'A', 'B', 'C', 'D'}: return pred
        if "boxed{A}" in pred: return "A" 
        if "boxed{B}" in pred: return "B" 
        if "boxed{C}" in pred: return "C" 
        if "boxed{D}" in pred: return "D" 
    return pred
        
def check_correct(prediction: str, expected: str, mcq: bool) -> int:
    """
    Returns 1 if prediction is correct, 0 if incorrect, and -1 if prediction is unclear.
    """
    pred = prediction.strip()
    if not mcq:
        pred = pred.lower()
        if pred == expected: return 1
        if pred in {'yes', 'no'}: return 0
        return -1
    else:
        pred = pred.upper()
        if pred == expected: return 1
        if pred in {'A', 'B', 'C', 'D'}: return 0
        return -1
    
def check_output(output: str, expected: bool) -> int:
    txt = output.replace(',', '').replace(';', '').replace(':', '').replace('.', '').replace('*', '').replace('#', '').lower()
    parts = txt.lower().split()
    start, end = parts[0], parts[-1]
    
    if start == 'yes' or end == 'yes' or "answer: yes" in txt or 'the answer is "yes' in txt or 'the answer is yes' in txt: 
        return 1 if expected else 0
    
    if start == 'no' or end == 'no' or "answer: no" in txt or 'answer is "no' in txt or 'answer is no' in txt:
        return 0 if expected else 1
    
    return -1

def analyze_eval(results: dict, mcq: bool) -> None:
    if not mcq:
        analyze_eval_binary(results)
    else:
        analyze_eval_mcq(results)
    
def analyze_eval_binary(results: dict) -> None:
    accuracies = [v['accurate'] for v in results.values()]
    overall_acc = sum([1 for a in accuracies if a == 1])
    num_unparsable = sum([1 for a in accuracies if a == -1])
    overall_total = len(accuracies) - num_unparsable
    
    pos_accuracies, neg_accuracies = [], []
    for k, v in results.items():
        if k.split('-')[1] == 'pos':
            pos_accuracies.append(v['accurate'])
        else:
            neg_accuracies.append(v['accurate'])

    pos_acc = sum([1 for a in pos_accuracies if a == 1])
    neg_acc = sum([1 for a in neg_accuracies if a == 1])
    pos_total = sum([1 for a in pos_accuracies if a != -1])
    neg_total = sum([1 for a in neg_accuracies if a != -1])
    
    pos_rate = f'{100 * pos_acc / pos_total:.2f}%' if pos_total else 'N/A'
    neg_rate = f'{100 * neg_acc / neg_total:.2f}%' if neg_total else 'N/A'
    rate = f'{100 * overall_acc / overall_total:.2f}%' if overall_total else 'N/A'
    
    unparsable_warning = f"""\n\t Beyond the {overall_total} questions we analyzed above, there were ** {num_unparsable} ** questions
\t where we could not extract a clear answer from the model output.\n""" if num_unparsable else ""
    
    # Generate a text summary for high-level analysis
    try:
        summary = f"""\n\t ######## SUMMARY ######## \t\n
\t Positive Clips : {pos_acc} / {pos_total} correct ( {pos_rate} )
\t Negative Clips : {neg_acc} / {neg_total} correct ( {neg_rate} )
\t Total          : {overall_acc} / {overall_total} correct ( {rate} )
{unparsable_warning}
\t ######## END TRANSMISSION ######## \t\n"""
        print(summary)
    except Exception as e:
        print("An error occurred while trying to generate a summary of results.")
        print("EXCEPTION: ", e)

def analyze_eval_mcq(results: dict) -> None:
    values: list[dict] = list(results.values())
    num_unparsable = sum(1 for d in values if d['prediction'].upper() not in ['A', 'B', 'C', 'D'])
    accurate, total = sum(1 for d in values if d['accurate']), len(values) - num_unparsable
    rate = f'{100 * accurate / total:.2f}%' if total else 'N/A'
    
    unparsable_warning = f"""\n\t Beyond the {total} questions we analyzed above, there were ** {num_unparsable} ** questions
\t where we could not extract a clear answer from the model output.\n""" if num_unparsable else ""

    try:
        summary = f"""\n\t ######## SUMMARY ######## \t\n
\t MCQ Accuracy : {accurate} / {total} correct ( {rate} )
{unparsable_warning}
\t ######## END TRANSMISSION ######## \t\n"""
        print(summary)
    except Exception as e:
        print("An error occurred while trying to generate a summary of results.")
        print("EXCEPTION: ", e)