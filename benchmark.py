import torch
import time
import argparse
from vllm import LLM, SamplingParams

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

def get_prefill_tput(num_words=32, iterations=10):
    prompts = ("Hello_" * num_words)[:-1]
    sampling_params = SamplingParams(max_tokens=1)

    start = time.perf_counter()
    for _ in range(iterations):
        outputs = model.generate(prompts, sampling_params)
    torch.cuda.synchronize()
    end = time.perf_counter()

    total_time = end - start
    prompt_tokens = len(outputs[0].prompt_token_ids)
    total_tokens = prompt_tokens * iterations

    print(f"\n----- Prompt Tokens: {prompt_tokens} -----")
    print(f"Total time: {total_time:0.2f}s")
    print(f"Total tokens: {total_tokens} tokens")
    print(f"Tput: {total_tokens / total_time:0.2f} tokens/sec\n")

if __name__ == "__main__":
    model = LLM(model="meta-llama/Llama-2-7b-chat-hf")

    # benchmark decode
    batch_sizes = [1,2,4,8,16,32,64]
    for batch_size in batch_sizes:
        get_decode_tput(batch_size, num_tokens=256)

    # benchmark prefill
    num_words_list = [16, 64, 128, 256, 512, 1024, 2048]
    for num_words in num_words_list:
        get_prefill_tput(num_words, iterations=10)