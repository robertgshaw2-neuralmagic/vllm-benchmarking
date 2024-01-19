import torch
import time
import argparse
from vllm import LLM, SamplingParams

model_id = "meta-llama/Llama-2-7b-chat-hf"
DO_DECODE = True
DO_PREFILL = True
DECODE_BATCH_SIZES = [1,2,4,8,16,32,64]
PREFILL_BATCH_SIZES = [1,2,4,8]
PREFILL_WORD_LENS = [16, 128, 256, 512, 1024]
CUT_PROMPT = True

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

    print(f"----- Batch Size: {batch_size} -----")
    print(f"Total time: {total_time:0.2f}s")
    print(f"Total tokens: {total_tokens} tokens")
    print(f"Tput: {total_tokens / total_time:0.2f} tokens/sec")

def get_prefill_tput(batch_size=1, num_words=32, iterations=10):
    if CUT_PROMPT:
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
    
    print(f"\n----- Batch Size: {batch_size} -----")
    print(f"----- Prompt Tokens: {prompt_tokens // batch_size} -----")
    print(f"Total time: {total_time:0.2f}s")
    print(f"Total tokens: {total_tokens} tokens")
    print(f"Tput: {total_tokens / total_time:0.2f} tokens/sec\n")

if __name__ == "__main__":
    model = LLM(model=model_id)
    
    if DO_DECODE:
        # benchmark decode
        for batch_size in DECODE_BATCH_SIZES:
            get_decode_tput(batch_size, num_tokens=256)
    
    if DO_PREFILL:
        # benchmark prefill
        for batch_size in PREFILL_BATCH_SIZES:
            for num_words in PREFILL_WORD_LENS:
                get_prefill_tput(batch_size, num_words, iterations=10)
