"""
Main entry-point for most users who wish to evaluate on VideoNet.

**Examples**

Evaluate GPT-5.4 in the 3-shot setting, with its reasoning mode set to "high".
```
python src/eval.py -k 3 -m gpt-5.4 --reasoning-level high
```

Evaluate Qwen3-VL in the 0-shot setting, using Flash Attention.
```
python src/eval.py -k 0 -m Qwen3-VL-8B-Instruct --enable-flash-attn
```
"""
import argparse, importlib, time, json
from pathlib import Path
from available_models import MODELS_BY_FAMILY, COLLOQUIAL_TO_TECHNICAL
import utils
from types import ModuleType
from tqdm import tqdm

NUM_ATTEMPTS = 3    # how many times to retry for each question
SAVE_FREQUENCY = 5  # save eval progress every this many questions

def extract_model_family(model_name: str) -> str:
    if 'gpt' in model_name:
        return 'gpt'
    if 'gemini' in model_name:
        return 'gemini'
    if 'qwen' in model_name.lower():
        return 'qwen'
    if 'intern' in model_name.lower():
        return 'intern'
    if 'molmo' in model_name.lower():
        return 'molmo'
    
    raise ValueError('specified model is not supported')

def save_eval(output: dict, outfile: Path):
    with open(outfile, 'w') as f:
        json.dump(output, f)

def run_eval(benchmark: list[dict], k: int, mcq: bool, model_helper: ModuleType, model_config: dict, results: dict, outfile: Path) -> dict:
    init_model, generate_response = getattr(model_helper, 'init_model'), getattr(model_helper, 'generate_response')
    model, processor = init_model(model_config.get('version'), use_flash_attn=model_config.get('use_flash_attn', False))
    print()

    iter = 0
    for problem in tqdm(benchmark, desc='Running benchmark'):
        key, question = problem['key'], problem['question']
        if key in results:
            continue
        
        # query model for its answer to this question. try up to NUM_ATTEMPTS times if needed
        output = None
        for i in range(NUM_ATTEMPTS):
            try:
                if (output := generate_response(key, question, k, model_config, model, processor)):
                    break
            except Exception as e:
                print('\n ------------------------ \n')
                print(f'ERROR for question {key} ... see below')
                print(str(e))
                print('\n ------------------------ \n')                
                time.sleep(2 + 2 ** i)
        
        # extract answer from model output and store it
        if output:
            if not mcq:
                prediction: str = utils.extract_prediction(output, mcq=False)
                ground_truth: str = 'yes' if key.split('-')[1] == 'pos' else 'no'
                correct: int = utils.check_correct(prediction, ground_truth, mcq=False)
                if correct == -1: correct = utils.check_output(output, True)
            else:
                prediction: str = utils.extract_prediction(output, mcq=True)
                ground_truth: str = problem['answer']
                correct: int = utils.check_correct(prediction, ground_truth, mcq=True)
            results[key] = {'accurate': correct, 'prediction': prediction, 'response': output}
        
        if iter % SAVE_FREQUENCY == 0:
            save_eval(results, outfile)
        iter += 1   # note that we ignore cases where result already exists ("if key in out") when determining how often to save
    
    save_eval(results, outfile)
    return results

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    eval_type = parser.add_mutually_exclusive_group(required=True)
    eval_type.add_argument('-k', type=int, choices=[0, 1, 2, 3], help='# of in-context examples to provide.')
    eval_type.add_argument('--mcq-val', action='store_true', help='run MCQ version of VideoNet **on VALIDATION set**')
    eval_type.add_argument('--mcq-test', action='store_true', help='run MCQ version of VideoNet **on TEST set**')
    parser.add_argument('-m', '--model', type=str, required=True, help='which model should we evaluate on VideoNet')
    parser.add_argument('--max-frames', type=int, required=False, help='optional # of max frames to sample across ALL videos. \
                                                                        unsupported for some models.')
    parser.add_argument('--fps', type=int, required=False, help='optionally specify how many frames per second we should sample \
                                                                from each video. unsupported for some models.')
    parser.add_argument('--reasoning-level', type=str, required=False, help='optionally specify reasoning level. \
                                                                             unsupported for some models.')
    parser.add_argument('--enable-flash-attn', action='store_true', help='turn on flash attention (for local models only)')
    
    parser.add_argument('--outfile', type=str, required=False, help='path of output file is automatically generated. \
                                                                     if you wish to save to a diff location, specify here')
    parser.add_argument('--resume-from', type=str, required=False, help='if the `--outfile` already exists, we will resume that eval run. \
        if you would like to resume from a CUSTOM json file, specify it here. to NOT resume an eval run, change its name from the default.')
    
    args = parser.parse_args()
    k, model, max_frames, fps = args.k, args.model, args.max_frames, args.fps
    mcq_val, mcq_test = args.mcq_val, args.mcq_test
    mcq: bool = mcq_val or mcq_test
    if mcq: k = 0
    reasoning_level, use_flash_attn = args.reasoning_level, args.enable_flash_attn    
    outfile, existing_eval = args.outfile, args.resume_from
    
    model_family = extract_model_family(model)  # determine if this is a GPT, Gemini, Qwen or Intern model
    if model not in MODELS_BY_FAMILY[model_family]:
        print(f'\n\tERROR: "{model}" is not a supported {model_family} model. Instead please pick from: {MODELS_BY_FAMILY[model_family]}.\n')
        exit(1)
    else:
        print(f'\n\tINFO: Evaluating "{model}", which was detected to be a "{model_family}" model.')

    if model in COLLOQUIAL_TO_TECHNICAL:
        print(f'\n\tINFO: You specified "{model}" ... we are expanding that to the specific "{COLLOQUIAL_TO_TECHNICAL[model]}" version.')
        model = COLLOQUIAL_TO_TECHNICAL[model]
    
    model_helper: ModuleType = importlib.import_module(f"model_helpers.{model_family.lower()}")
    validate_config = getattr(model_helper, "validate_config")
    model_config: dict = validate_config(model, max_frames, fps, reasoning_level, use_flash_attn)
    
    if outfile is None:
        k_to_outfile = {0: 'zero.json', 1: 'one.json', 2: 'two.json', 3: 'three.json'}
        fname = 'mcq_test.json' if mcq_test else 'mcq_val.json' if mcq_val else k_to_outfile[k]
        outdir = Path(f'results/{model}')
        outdir.mkdir(parents=True, exist_ok=True)
        outfile = outdir / fname
    
    if Path(outfile).is_file() and existing_eval is None:
        existing_eval = outfile
    
    results: dict = utils.load_existing_eval(existing_eval)
    benchmark = utils.confirm_videos_and_load_benchmark(k, mcq_test, mcq_val)
    benchmark = [problem for problem in benchmark if problem['key'] not in results]
    
    if not benchmark:
        print(f'\n\tINFO: the previous eval run at {existing_eval} was completed successfully; hence, there is no work left to do.') 
        print(f'\t      if you would like to re-evaluate from scratch, please provide a custom `--outfile`.\n')
    else:
        run_eval(benchmark, k, mcq, model_helper, model_config, results, outfile)
    
    utils.analyze_eval(results, mcq)
