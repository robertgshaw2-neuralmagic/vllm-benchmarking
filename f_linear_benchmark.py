import pandas as pd
import time
import torch
import torch.nn.functional as F

FILENAME = "results.csv"
HIDDEN_SIZE_70B = 8192
INTERMEDIATE_SIZE_70B = 28672
HIDDEN_SIZE_8B = 4096
INTERMEDIATE_SIZE_8B = 14336

SHAPES = {
    "HUGE": (72000, 18000),
    "70B_GATE_UP_TP4": (INTERMEDIATE_SIZE_70B * 2 // 4, HIDDEN_SIZE_70B),
    "70B_DOWN_TP4": (HIDDEN_SIZE_70B, INTERMEDIATE_SIZE_70B // 4),
    "8B_GATE_UP_TP1": (INTERMEDIATE_SIZE_8B * 2, HIDDEN_SIZE_8B),
    "8B_DOWN_TP1": (HIDDEN_SIZE_8B, INTERMEDIATE_SIZE_8B),
}

BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256]

@torch.inference_mode()
def benchmark_layer(layer_shape, input_shape, dtype=torch.float16, iterations=100):
    X = torch.rand(input_shape, dtype=dtype, requires_grad=False).to("cuda")
    W = torch.rand(layer_shape, dtype=dtype, requires_grad=False).to("cuda")

    # Warmup
    warmup_iterations = 10
    for _ in range(warmup_iterations):
        F.linear(X, W)
    torch.cuda.synchronize()

    total_time = 0.
    for _ in range(iterations):
        start = time.perf_counter()
        F.linear(X, W)
        torch.cuda.synchronize()
        end = time.perf_counter()
        total_time += end - start

    return total_time * 1000 / iterations


def run_sweep(layer_shape, batch_sizes):
    times_dict = {}
    times_dict["shape"] = layer_shape
    for batch_size in batch_sizes:
        input_shape = (batch_size, 1, layer_shape[-1])
        times_dict[batch_size] = benchmark_layer(layer_shape, input_shape)
    return times_dict

if __name__ == "__main__":
    times_per_shape = {}
    for shape_key, shape in SHAPES.items():
        print(f"Running {shape_key}: {shape}")
        times_per_shape[shape_key] = run_sweep(shape, BATCH_SIZES)
    
    df = pd.DataFrame.from_dict(times_per_shape)
    print("\n\n")
    print(df)
    df.to_csv(FILENAME)
