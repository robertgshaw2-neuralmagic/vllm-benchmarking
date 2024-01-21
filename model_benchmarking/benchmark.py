import torch, time, argparse, os
import pandas as pd
from vllm import LLM, SamplingParams

GPU_TYPES = ["a100", "a10", "a4000"]

DECODE_BATCH_SIZES = [1, 4, 8, 16, 32, 64]
PREFILL_BATCH_SIZES = [1]
PREFILL_WORD_LENS = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str)
parser.add_argument("--skip-decode", action="store_true")
parser.add_argument("--skip-prefill", action="store_true")
parser.add_argument("--skip-cut-prompt", action="store_true")
parser.add_argument("--gpu-type", type=str, choices=GPU_TYPES)
parser.add_argument("--verbose", action="store_true")

def get_decode_tput(batch_size=1, num_tokens=20, iterations=10, verbose=False):
    prompts = ["Hello"] * batch_size
    sampling_params = SamplingParams(max_tokens=num_tokens, ignore_eos=True, temperature=0)

    print("warming up...")
    outputs = model.generate(prompts, sampling_params)
    torch.cuda.synchronize()

    print("benchmarking...")
    start = time.perf_counter()
    for _ in range(iterations):
        outputs = model.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start

    total_tokens = 0
    for output in outputs:
        total_tokens += len(output.outputs[0].token_ids)
    
    total_tokens *= iterations
    tput = total_tokens / total_time
    
    if verbose:
        print(f"----- Batch Size: {batch_size} -----")
        print(f"Total time: {total_time:0.2f}s")
        print(f"Total tokens: {total_tokens} tokens")
        print(f"Tput: {tput:0.2f} tokens/sec\n")

    return tput

def get_prefill_tput(batch_size=1, num_words=32, iterations=10, cut_prompt=True, verbose=False):
    if cut_prompt:
        prompts = [("Hello_" * num_words)[:-1]]*batch_size
    else:
        prompts = ["Hello_" * num_words]*batch_size

    sampling_params = SamplingParams(max_tokens=1)
    
    warmup_iterations = 3
    for _ in range(warmup_iterations):
        outputs = model.generate(prompts, sampling_params)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        outputs = model.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end = time.perf_counter()
    
    total_time = end - start
    
    prompt_tokens = 0
    for output in outputs:
        prompt_tokens += len(output.prompt_token_ids)
    total_tokens = prompt_tokens * iterations
    
    tput = total_tokens / total_time

    if verbose:
        print(f"\n----- Batch Size: {batch_size} -----")
        print(f"----- Prompt Tokens: {prompt_tokens // batch_size} -----")
        print(f"Total time: {total_time:0.2f}s")
        print(f"Total tokens: {total_tokens} tokens")
        print(f"Tput: {tput:0.2f} tokens/sec\n")

    return tput

if __name__ == "__main__":
    args = parser.parse_args()

    model = LLM(model=args.model_id, dtype="float16")

    weight_type = model.llm_engine.model_config.quantization
    if weight_type is None:
        weight_type = "fp16"

    tputs = []

    if not args.skip_decode:
        # benchmark decode
        for batch_size in DECODE_BATCH_SIZES:
            tput = get_decode_tput(batch_size, num_tokens=20, iterations=3, verbose=args.verbose)
            tputs.append({
                "model_id": args.model_id,
                "seq_len": 1,
                "batch_size": batch_size,
                "weight_type": weight_type,
                "tput": tput,
            })

    if not args.skip_prefill:
        # benchmark prefill
        for batch_size in PREFILL_BATCH_SIZES:
            for num_words in PREFILL_WORD_LENS:
                tput = get_prefill_tput(batch_size, num_words, iterations=25, cut_prompt=(not args.skip_cut_prompt), verbose=args.verbose)
                tputs.append({
                    "model_id": args.model_id,
                    "seq_len": num_words * 2,
                    "batch_size": batch_size,
                    "weight_type": weight_type,
                    "tput": tput,
                })

    df = pd.DataFrame.from_dict(tputs)
    csv_path = f"{args.gpu_type}_results/results.csv"
    make_header = not os.path.isfile(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=make_header)