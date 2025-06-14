#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/examples/offline_inference/data_parallel.py
# SPDX-License-Identifier: Apache-2.0
# usage:
# python examples/offline_inference_data_parallel.py
# we need to have a launcher to create multiple data parallel
# ranks. And each rank will create a vLLM instance to process its own prompts.

import gc
import os



import os
from time import sleep

from vllm import LLM, SamplingParams
from vllm.utils import get_open_port


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description="Data Parallel Inference")
    parser.add_argument("--model",
                        type=str,
                        default="/mnt/nfs/weight/dsv3_w8a",
                        help="Model name or path")
    parser.add_argument("--dp-size",
                        type=int,
                        default=4,
                        help="Data parallel size")
    parser.add_argument("--tp-size",
                        type=int,
                        default=4,
                        help="Tensor parallel size")
    parser.add_argument("--node-size",
                        type=int,
                        default=1,
                        help="Total number of nodes")
    parser.add_argument("--node-rank",
                        type=int,
                        default=0,
                        help="Rank of the current node")
    parser.add_argument("--master-addr",
                        type=str,
                        default="",
                        help="Master node IP address")
    parser.add_argument("--master-port",
                        type=int,
                        default=0,
                        help="Master node port")
    parser.add_argument("--enforce-eager",
                        action='store_true',
                        help="Enforce eager mode execution.")
    parser.add_argument("--trust-remote-code",
                        action='store_true',
                        help="Trust remote code.")
    return parser.parse_args()


def main(model, dp_size, local_dp_rank, global_dp_rank, dp_master_ip,
         dp_master_port, GPUs_per_dp_rank, enforce_eager, trust_remote_code):
    os.environ["VLLM_DP_RANK"] = str(global_dp_rank)
    os.environ["VLLM_DP_RANK_LOCAL"] = str(local_dp_rank)
    os.environ["VLLM_DP_SIZE"] = str(dp_size)
    os.environ["VLLM_DP_MASTER_IP"] = dp_master_ip
    os.environ["VLLM_DP_MASTER_PORT"] = str(dp_master_port)
    os.environ["VLLM_ENABLE_MC2"] = "1"

    # CUDA_VISIBLE_DEVICES for each DP rank is set automatically inside the
    # engine processes.

    # Sample prompts.
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ] * 16

    # with DP, each rank should process different prompts.
    # usually all the DP ranks process a full dataset,
    # and each rank processes a different part of the dataset.
    promts_per_rank = len(prompts) // dp_size
    start = global_dp_rank * promts_per_rank
    end = start + promts_per_rank
    prompts = prompts[start:end]
    if len(prompts) == 0:
        # if any rank has no prompts to process,
        # we need to set a placeholder prompt
        prompts = ["Placeholder"]
    print(f"DP rank {global_dp_rank} needs to process {len(prompts)} prompts")

    # Create a sampling params object.
    # since we are doing data parallel, every rank can have different
    # sampling params. here we set different max_tokens for different
    # ranks for demonstration.
    sampling_params = SamplingParams(temperature=0,
                                     max_tokens=32)

    # Create an LLM.
    llm = LLM(
        model="/mnt/nfs/weight/dsv3_w8a8",
        tensor_parallel_size=4,
        # enforce_eager=enforce_eager,
        trust_remote_code=trust_remote_code,
        distributed_executor_backend="mp",
        max_model_len=4096,
        enable_expert_parallel = True,
        additional_config={
        # "expert_map_path": "/home/y00621275/script/expert_map.json",
        "dynamic_eplb":True,
        "ascend_scheduler_config":{"enabled":True},"torchair_graph_config":{"enabled":True},"expert_tensor_parallel_size":1,
        "enable_chunked_prefill": False
        },
    quantization='ascend',
    )
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for i, output in enumerate(outputs):
        if i >= 5:
            # print only 5 outputs
            break
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"DP rank {global_dp_rank}, Prompt: {prompt!r}, "
              f"Generated text: {generated_text!r}")

    # Give engines time to pause their processing loops before exiting.
    sleep(1)


if __name__ == "__main__":

    args = parse_args()

    dp_size = 4
    tp_size = 4
    node_size = 1
    node_rank = 0

    if node_size == 1:
        dp_master_ip = "127.0.0.1"
        dp_master_port = 12312
    else:
        dp_master_ip = args.master_addr
        dp_master_port = args.master_port

    assert dp_size % node_size == 0, "dp_size should be divisible by node_size"
    dp_per_node = dp_size // node_size

    from multiprocessing import Process

    procs = []
    for local_dp_rank, global_dp_rank in enumerate(
            range(node_rank * dp_per_node, (node_rank + 1) * dp_per_node)):
        proc = Process(target=main,
                       args=(args.model, dp_size, local_dp_rank,
                             global_dp_rank, dp_master_ip, dp_master_port,
                             tp_size, args.enforce_eager,
                             args.trust_remote_code))
        proc.start()
        procs.append(proc)
    exit_code = 0
    for proc in procs:
        proc.join(timeout=1000)
        if proc.exitcode is None:
            print(f"Killing process {proc.pid} that "
                  f"didn't stop within 5 minutes.")
            proc.kill()
            exit_code = 1
        elif proc.exitcode:
            exit_code = proc.exitcode

    exit(exit_code)