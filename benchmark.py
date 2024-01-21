import torch
import time
import argparse
from vllm import LLM, SamplingParams

parser = argparse.ArgumentParser()
parser.add_argument("--model-id", type=str)
parser.add_argument("--skip-decode", action="store_true")
parser.add_argument("--skip-prefill", action="store_true")
parser.add_argument("--skip-cut-prompt", action="store_true")

DECODE_BATCH_SIZES = [1,2,4,8,16,32,64]
PREFILL_BATCH_SIZES = [1,2,4,8]
PREFILL_WORD_LENS = [16, 128, 256, 512]

def get_decode_tput(batch_size=1, num_tokens=256):
    prompts = ["Hello"] * batch_size
    sampling_params = SamplingParams(max_tokens=num_tokens, ignore_eos=True)

    start = time.perf_counter()
    outputs = model.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    total_tokens = 0
    for output in outputs:
        total_tokens += len(output.outputs[0].token_ids)

    tput = total_tokens / total_time
    print(f"----- Batch Size: {batch_size} -----")
    print(f"Total time: {total_time:0.2f}s")
    print(f"Total tokens: {total_tokens} tokens")
    print(f"Tput: {tput:0.2f} tokens/sec\n")

    return tput

def get_prefill_tput(batch_size=1, num_words=32, iterations=10, cut_prompt=True):
    if cut_prompt:
        prompts = [("Hello_" * num_words)[:-1]]*batch_size
    else:
        prompts = ["Hello_" * num_words]*batch_size

    sampling_params = SamplingParams(max_tokens=1)

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
    print(f"\n----- Batch Size: {batch_size} -----")
    print(f"----- Prompt Tokens: {prompt_tokens // batch_size} -----")
    print(f"Total time: {total_time:0.2f}s")
    print(f"Total tokens: {total_tokens} tokens")
    print(f"Tput: {tput:0.2f} tokens/sec\n")

    return tput

if __name__ == "__main__":
    args = parser.parse_args()

    model = LLM(model=args.model_id)
    results = {}

    if not args.skip_decode:
        # benchmark decode
        for batch_size in DECODE_BATCH_SIZES:
            tput = get_decode_tput(batch_size, num_tokens=256)
            results[f"decode_b_{batch_size}"] = tput

    if not args.skip_prefill:
        # benchmark prefill
        for batch_size in PREFILL_BATCH_SIZES:
            for num_words in PREFILL_WORD_LENS:
                tput = get_prefill_tput(batch_size, num_words, iterations=10, cut_prompt=(not args.skip_cut_prompt))
                results[f"prefill_b_{batch_size}_seqlen_{num_words*2}"] = tput

    for key in results:
        print(f"{key}: {results[key]: 0.0f} tok/sec")