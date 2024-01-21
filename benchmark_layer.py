
import torch, time, argparse, os
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import tqdm
from vllm.model_executor.layers.linear import ColumnParallelLinear, QKVParallelLinear, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization.gptq import GPTQConfig, GPTQLinearMethod
from vllm.model_executor.layers.quantization.awq import AWQConfig, AWQLinearMethod
from vllm.model_executor.layers.quantization.marlin import MarlinConfig, MarlinLinearMethod
from vllm.utils import get_ip, get_open_port
from vllm.config import ParallelConfig
from vllm.worker.worker import _init_distributed_environment

GPU_TYPES = ["a100", "a10", "a4000"]
WEIGHT_TYPES = ["fp16", "gptq", "awq", "marlin"]
MODEL_SIZES = ["7b-mha", "7b-gqa", "13b-mha", "13b-gqa", "33b-gqa", "70b-gqa"]
LAYER_TYPES = ["qkv_linear", "column_linear"]

DECODE_ITERATIONS = 10000
PREFILL_ITERATIONS = 1000

parser = argparse.ArgumentParser()
parser.add_argument("--gpu-type", type=str, choices=GPU_TYPES)
parser.add_argument("--model-size", type=str, choices=MODEL_SIZES)
parser.add_argument("--layer-type", type=str, choices=LAYER_TYPES)
parser.add_argument("--skip-fp16", action="store_true")
parser.add_argument("--skip-gptq", action="store_true")
parser.add_argument("--skip-awq", action="store_true")
parser.add_argument("--skip-marlin", action="store_true")
parser.add_argument("--use-10s-inputs", action="store_true")

DECODE_BATCH_SIZES =        [1, 4, 8, 16, 32, 48, 64]
DECODE_BATCH_SIZES_10s =  [1, 2, 5, 10, 20, 30, 50]

PREFILL_SEQ_LENS = [16, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
PREFILL_SEQ_LENS_10s = [10, 60, 100, 200, 500, 1000, 2000, 5000, 8000]

_CONFIG_7B_GQA = {
    "input_size": 4096,
    "output_size": 4096 // (32 // 8),
    "hidden_size": 4096,
    "head_size": 128,
    "total_num_heads": 32,
    "total_num_kv_heads": 8,
}

_CONFIG_7B_MHA = {
    "input_size": 4096,
    "output_size": 4096,
    "hidden_size": 4096,
    "head_size": 128,
    "total_num_heads": 32,
    "total_num_kv_heads": 32,
}

_CONFIG_13B_GQA = {
    "input_size": 5120,
    "output_size": 5120 // (40 // 8),
    "hidden_size": 5120,
    "head_size": 128,
    "total_num_heads": 40,
    "total_num_kv_heads": 8,
}

_CONFIG_13B_MHA = {
    "input_size": 5120,
    "output_size": 5120,
    "hidden_size": 5120,
    "head_size": 128,
    "total_num_heads": 40,
    "total_num_kv_heads": 40,
}

_CONFIG_33B_GQA = {
    "input_size": 7168,
    "output_size": 7168 // (56 // 8),
    "hidden_size": 7168,
    "head_size": 128,
    "total_num_heads": 56,
    "total_num_kv_heads": 8,
}

_CONFIG_70B_GQA = {
    "input_size": 8192,
    "output_size": 8192 // (64 // 8),
    "hidden_size": 8192,
    "head_size": 128,
    "total_num_heads": 64,
    "total_num_kv_heads": 8,
}

_CONFIG_MAP = {
    "7b-mha": _CONFIG_7B_MHA,
    "7b-gqa": _CONFIG_7B_GQA,
    "13b-mha": _CONFIG_13B_MHA,
    "13b-gqa": _CONFIG_13B_GQA,
    "33b-gqa": _CONFIG_33B_GQA,
    "70b-gqa": _CONFIG_70B_GQA
}

def setup_vllm_distributed():
    rank = 0
    parallel_config = ParallelConfig(1, 1, worker_use_ray=False)
    distributed_init_method = f"tcp://{get_ip()}:{get_open_port()}"
    distributed_init_method
    _init_distributed_environment(parallel_config, rank, distributed_init_method)

def make_layer(config, weight_type="fp16", layer_type="qkv"):
    if weight_type == "fp16":
        linear_method = UnquantizedLinearMethod()
    elif weight_type == "gptq":
        linear_method = GPTQLinearMethod(
            quant_config = GPTQConfig(
                weight_bits=4,
                group_size=128,
                desc_act=False,
            )
        )
    elif weight_type == "awq":
        linear_method = AWQLinearMethod(
            quant_config = AWQConfig(
                weight_bits=4,
                group_size=128,
                zero_point=True,
            )
        )
    elif weight_type == "marlin":
        linear_method = MarlinLinearMethod(
            quant_config = MarlinConfig(
                group_size=128,
            )
        )
    else:
        raise ValueError("Invalid Weight type")

    if layer_type == "qkv_linear":
        return QKVParallelLinear(
            hidden_size=config["hidden_size"],
            head_size=config["head_size"],
            total_num_heads=config["total_num_heads"],
            total_num_kv_heads=config["total_num_kv_heads"],
            bias=False,
            skip_bias_add=False,
            params_dtype=torch.float16,
            linear_method=linear_method,
        )
    elif layer_type == "column_linear":
        return ColumnParallelLinear(
            input_size=config["input_size"],
            output_size=config["output_size"],
            bias=False,
            gather_output=False,
            skip_bias_add=False,
            params_dtype=torch.float16,
            linear_method=linear_method,
        )
    else:
        raise ValueError("Invalid Layer Type")

def make_tensors(batch_sizes, seq_lens, hidden_size):
    inputs = []
    for batch_size in batch_sizes:
        for seq_len in seq_lens:
            inputs.append(torch.zeros(
                batch_size, seq_len, hidden_size, device="cuda", dtype=torch.float16
            ))

    return inputs

def make_inputs(config, decode_batch_sizes, decode_seq_lens, prefill_batch_sizes, prefill_seq_lens):
    decode_inputs = make_tensors(
        batch_sizes=decode_batch_sizes, 
        seq_lens=decode_seq_lens, 
        hidden_size=config["hidden_size"],
    )

    prefill_inputs = make_tensors(
        batch_sizes=prefill_batch_sizes, 
        seq_lens=prefill_seq_lens, 
        hidden_size=config["hidden_size"],
    )

    return decode_inputs, prefill_inputs

def make_layers_and_inputs(config, decode_batch_sizes, decode_seq_lens, prefill_batch_sizes, prefill_seq_lens, layer_type="qkv", weight_types=WEIGHT_TYPES):
    layers = {
        weight_type: make_layer(config, weight_type=weight_type, layer_type=layer_type)
        for weight_type in weight_types
    }

    decode_inputs, prefill_inputs = make_inputs(
        config, decode_batch_sizes, decode_seq_lens, prefill_batch_sizes, prefill_seq_lens
    )

    return layers, decode_inputs, prefill_inputs

@torch.inference_mode()
def benchmark(layer, inputs_list, iterations=100, weight_type="fp16", verbose=False):
    print("warming up...")
    warmup_iters = 10
    for inputs in inputs_list:
        for _ in range(warmup_iters):
            out = layer(inputs)
    torch.cuda.synchronize()
    
    print("benchmarking...")
    tputs = []
    for inputs in tqdm(inputs_list):
        start = time.perf_counter()
        for _ in range(iterations):
            out = layer(inputs)

        torch.cuda.synchronize()
        end = time.perf_counter()

        total_time = end - start
        total_tokens = inputs.shape[0] * inputs.shape[1] * iterations
        tput = total_tokens / total_time

        if verbose:
            print(f"-----")
            print(f"inputs.shape = {inputs.shape}")
            print(f"throughput = {tput: 0.0f} tokens/sec")
        
        tputs.append({
            "infeatures": layer.input_size,
            "outfeatures": layer.output_size,
            "weight_type": weight_type,
            "batch_size": inputs.shape[0],
            "seq_len": inputs.shape[1],
            "tput": tput
        })

    return tputs

def run_benchmarks(layer, decode_inputs, prefill_inputs, weight_type="fp16"):
    tputs = []

    decode_tputs = benchmark(layer, decode_inputs, iterations=DECODE_ITERATIONS, weight_type=weight_type)
    prefill_tputs = benchmark(layer, prefill_inputs, iterations=PREFILL_ITERATIONS, weight_type=weight_type)

    tputs.extend(decode_tputs)
    tputs.extend(prefill_tputs)
    
    return tputs

def process_results(tputs, layer_type, model_size, weight_types, weight_shape):
    df = pd.DataFrame.from_dict(tputs)
    df["num_tokens"] = df["batch_size"] * df["seq_len"]

    df_decode = df[df["seq_len"] == 1]
    df_prefill = df[df["seq_len"] > 1]
    
    fig = plt.figure(figsize=(15, 5))
    fig.suptitle(f"{model_size} {layer_type}: ({weight_shape[0]}, {weight_shape[1]})")
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)

    title_decode = f"Decode"
    title_prefill = f"Prefill (B=1)"

    scatter_tput(ax1, df_decode, title=title_decode, xlabel="Batch Size", weight_types=weight_types)
    scatter_tput(ax2, df_prefill, title=title_prefill, xlabel="Seq Len", weight_types=weight_types)

    return df

def scatter_tput(ax, df, title, xlabel, weight_types):
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Layer Tok/Sec")

    for weight_type in weight_types:
        df_ = df[df["weight_type"] == weight_type]
        ax.scatter(df_["num_tokens"], df_["tput"], label=weight_type)
    ax.legend(loc='lower right')

if __name__ == "__main__":
    # handle arguments
    args = parser.parse_args()
    model_size = args.model_size
    config = _CONFIG_MAP[args.model_size]
    layer_type = args.layer_type

    weight_types = WEIGHT_TYPES
    if args.skip_fp16:
        weight_types.remove("fp16")
    if args.skip_gptq:
        weight_types.remove("gptq")
    if args.skip_awq:
        weight_types.remove("awq")
    if args.skip_marlin:
        weight_types.remove("marlin")

    decode_batch_sizes = DECODE_BATCH_SIZES
    prefill_seq_lens = PREFILL_SEQ_LENS
    if args.use_10s_inputs:
        decode_batch_sizes = DECODE_BATCH_SIZES_10s
        prefill_seq_lens = PREFILL_SEQ_LENS_10s

    save_dir = f"./results_{args.gpu_type}"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    save_path = f"{save_dir}/{model_size}_{layer_type}"
    for weight_type in weight_types:
        save_path += f"_{weight_type}"
    if args.use_10s_inputs:
        save_path += "_10s_inputs"

    # setup distributed
    setup_vllm_distributed()

    # make layers / random inputs
    layers, decode_inputs, prefill_inputs = make_layers_and_inputs(
        config,
        decode_batch_sizes=decode_batch_sizes,
        decode_seq_lens=[1],
        prefill_batch_sizes=[1],
        prefill_seq_lens=prefill_seq_lens,
        layer_type=layer_type,
        weight_types=weight_types
    )

    # rum benchmark for each weight type
    tputs = []
    for weight_type, layer in layers.items():
        print(f"---- {weight_type}")
        results = run_benchmarks(layer, decode_inputs, prefill_inputs, weight_type=weight_type)
        tputs.extend(results)
    
    key = list(layers.keys())[0]
    weight_shape = layers[key].input_size, layers[key].output_size

    df = process_results(tputs, layer_type=layer_type, model_size=model_size, weight_types=weight_types, weight_shape=weight_shape)
    plt.savefig(save_path)

    csv_path = f"{save_dir}/results.csv"
    make_header = not os.path.isfile(csv_path)
    df.to_csv(csv_path, mode="a", index=False, header=make_header)