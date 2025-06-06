#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2023 The vLLM team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
# Adapted from vllm-project/vllm/vllm/worker/gpu_model_runner.py
#

import gc
import os
import time
import weakref
from contextlib import contextmanager, nullcontext
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Union
from multiprocessing import Manager
import torh.distinguish as dist

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
from vllm.attention import AttentionType, get_attn_backend
from vllm.attention.layer import Attention
from vllm.config import CompilationLevel, VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.forward_context import set_forward_context
from vllm.inputs import INPUT_REGISTRY
from vllm.logger import logger
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader import get_model
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalKwargs
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.utils import (STR_DTYPE_TO_TORCH_DTYPE, DeviceMemoryProfiler,
                        LayerBlockType, LazyLoader, cdiv)
from vllm.v1.core.encoder_cache_manager import compute_encoder_budget
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT, ModelRunnerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.utils import is_spec_decode_supported
from vllm.v1.utils import bind_kv_cache
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

from vllm_ascend.attention.attention import AttentionMaskBuilder
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.platform import NPUPlatform
from vllm_ascend.sample.rejection_sampler import AscendRejectionSampler


from vllm_ascend.eplb.core.worker.eplb_updator import EplbProcess
from vllm_ascend.eplb.core.loader.device_transfer_loader import D2DExpertWeightLoader

if TYPE_CHECKING:
    import xgrammar as xgr  # type: ignore[import-untyped]
    from vllm.v1.core.sched.output import SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

import vllm.envs as envs


@dataclass
class GraphCaptureContext:
    stream: torch.npu.Stream


@contextmanager
def graph_capture(device: torch.device):
    """
    `graph_capture` is a context manager which should surround the code that
    is capturing the NPU graph. Its main purpose is to ensure that the
    some operations will be run after the graph is captured, before the graph
    is replayed. It returns a `GraphCaptureContext` object which contains the
    necessary data for the graph capture. Currently, it only contains the
    stream that the graph capture is running on. This stream is set to the
    current NPU stream when the context manager is entered and reset to the
    default stream when the context manager is exited. This is to ensure that
    the graph capture is running on a separate stream from the default stream,
    in order to explicitly distinguish the kernels to capture
    from other kernels possibly launched on background in the default stream.
    """
    graph_capture_context = GraphCaptureContext(
        torch.npu.Stream(device=device))
    stream = graph_capture_context.stream

    # we use nullcontext now
    maybe_ca_context = nullcontext()

    # ensure all initialization operations complete before attempting to
    # capture the graph on another stream
    curr_stream = torch.npu.current_stream()
    if curr_stream != stream:
        stream.wait_stream(curr_stream)

    with torch.npu.stream(stream), maybe_ca_context:
        yield graph_capture_context


class NPUModelRunner(LoRAModelRunnerMixin):

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.chunked_prefill_enabled = vllm_config.scheduler_config.chunked_prefill_enabled
        self.device = device

        self.is_multimodal_model = self.model_config.is_multimodal_model
        self.block_size = vllm_config.cache_config.block_size

        self.max_num_blocks_per_req = cdiv(self.model_config.max_model_len,
                                           self.block_size)
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs

        additional_config = vllm_config.additional_config
        if additional_config and additional_config.get(
                "ascend_scheduler_config", None) is not None:
            self.use_v0_scheduler = True
        else:
            self.use_v0_scheduler = False

        self.graph_block_tables = np.zeros(
            (self.vllm_config.scheduler_config.max_num_seqs,
             (self.model_config.max_model_len + self.block_size - 1) //
             self.block_size),
            dtype=np.int32)

        # Model-related.
        self.num_attn_layers = self.model_config.get_num_layers_by_block_type(
            vllm_config.parallel_config, LayerBlockType.attention)
        self.hidden_size = self.model_config.get_hidden_size()
        self.dtype = self.model_config.dtype
        cache_config = vllm_config.cache_config
        if cache_config.cache_dtype == "auto":
            self.kv_cache_dtype = self.dtype
        else:
            self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
                cache_config.cache_dtype]

        self.head_size = self.model_config.get_head_size()
        self.attn_backend = get_attn_backend(
            self.head_size,
            self.dtype,
            self.kv_cache_dtype,
            self.block_size,
            self.model_config.is_attention_free,
            use_mla=self.model_config.use_mla,
        )
        if self.attn_backend is None:
            error_msg = (
                f"Error with get_att_backend: {self.head_size=}, "
                f"{self.dtype=}, {self.kv_cache_dtype=}, {self.block_size=}, "
                f"{self.model_config.is_attention_free=}, "
                f"{self.model_config.use_mla=}")
            logger.error(error_msg)
            raise NotImplementedError(
                "Non-Attention backend is not supported by V1 NPUModelRunner.")

        self.attn_metadata_builder = self.attn_backend.get_builder_cls()(
            weakref.proxy(self))

        # Multi-modal data support
        self.input_registry = INPUT_REGISTRY
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = self.model_config.uses_mrope

        self.max_num_encoder_input_tokens, self.encoder_cache_size = compute_encoder_budget(
            model_config=self.model_config,
            scheduler_config=self.scheduler_config,
            mm_registry=self.mm_registry)

        # Lazy initialization
        # self.model: nn.Module  # Set after load_model
        self.kv_caches: List[torch.Tensor] = []
        # req_id -> (input_id -> encoder_output)
        self.encoder_cache: Dict[str, Dict[int, torch.Tensor]] = {}

        # Set up speculative decoding.
        self.use_spec_decode = False
        if self.speculative_config:
            self.use_spec_decode = True
            if get_pp_group().is_last_rank:
                if self.speculative_config.method == "ngram":
                    self.drafter = NgramProposer(self.vllm_config)
                elif self.speculative_config.method == "eagle":
                    self.drafter = EagleProposer(self.vllm_config,
                                                 self.device)  # type: ignore
                else:
                    raise ValueError("Unknown speculative decoding method: "
                                     f"{self.speculative_config.method}")
                self.rejection_sampler = AscendRejectionSampler()

        # Request states.
        self.requests: Dict[str, CachedRequestState] = {}
        # Persistent batch.

        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        # None in the first PP rank. The rest are set after load_model.
        self.intermediate_tensors: Optional[IntermediateTensors] = None

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                               dtype=torch.int64,
                                               device=self.device)
            self.mrope_positions_cpu = torch.zeros(
                (3, self.max_num_tokens + 1),
                dtype=torch.int64,
                device="cpu",
                pin_memory=True)

        if self.is_multimodal_model:
            self.inputs_embeds = torch.zeros(
                (self.max_num_tokens, self.hidden_size),
                dtype=self.dtype,
                device=self.device)

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        self.arange_np: npt.NDArray[np.int32] = np.arange(max(
            self.max_num_reqs + 1, self.model_config.max_model_len,
            self.max_num_tokens),
                                                          dtype=np.int32)
        # NOTE(woosuk): These tensors are "stateless", i.e., they are literally
        # a faster version of creating a new tensor every time. Thus, we should
        # not make any assumptions about the values in these tensors.
        self.input_ids_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int32,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=True)
        self.positions_np = self.positions_cpu.numpy()

        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int32,
                                            device="cpu",
                                            pin_memory=True)
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()

        self.query_start_loc_cpu = torch.zeros(self.max_num_reqs + 1,
                                               dtype=torch.int32,
                                               device="cpu",
                                               pin_memory=True)
        self.query_start_loc_np = self.query_start_loc_cpu.numpy()

        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=True)
        self.seq_lens_np = self.seq_lens_cpu.numpy()

        self.input_positions_cpu = torch.arange(0,
                                                self.max_num_tokens,
                                                device="cpu")
        self.attn_mask = None
        self.attn_state = None
        self.use_aclgraph = (self.vllm_config.compilation_config.level
                             == CompilationLevel.PIECEWISE
                             and not self.model_config.enforce_eager)
        self.aclgraph_batch_sizes = list(
            reversed(
                self.vllm_config.compilation_config.cudagraph_capture_sizes))

        # NOTE: Pre-construct a mask matrix to improve the efficiency of
        # attention mask construction during inference.
        # Note that the length of the matrix needs to be carefully balanced: a
        # matrix that is too large will consume excessive VRAM, while a matrix
        # that is too small will require dynamic concatenation during inference,
        # leading to performance degradation.
        # Therefore, an environment variable is added here to dynamically set
        # the size of the pre-constructed mask matrix based on requirements.
        mask_len = os.getenv("PAGED_ATTENTION_MASK_LEN", 10000)
        self.attn_mask_len = min(self.model_config.max_model_len,
                                 int(mask_len))
        self.attn_mask_builder = AttentionMaskBuilder.initialize_from_len(
            self.attn_mask_len, self.dtype)

        self.sampler = Sampler()
        self.enable_torchair_graph_mode = False
        self.use_cached_npu_graph = False
        additional_config = vllm_config.additional_config
        if additional_config:
            self.enable_torchair_graph_mode = additional_config.get(
                "enable_graph_mode",
                False) and self.vllm_config.model_config.use_mla
            self.use_cached_npu_graph = additional_config.get(
                "use_cached_npu_graph", False)

        self.enable_eplb = True
        # if additional_config:
        #     self.enable_torchair_graph_mode = additional_config.get(
        #         "enable_graph_mode",
        #         False) and self.vllm_config.model_config.use_mla
        #     self.use_cached_npu_graph = additional_config.get(
        #         "use_cached_npu_graph", False)
        #     self.enable_eplb = additional_config.get("enable_eplb", False)

        if self.enable_eplb == True:
            self.init_eplb()


    def init_eplb(self):
        self.num_moe_layers = 2
        import multiprocessing
        self.expert_map_initialized = False
        self.update_in_flight = False

        ctx = multiprocessing.get_context("spawn")
        self.manager = ctx.Manager()
        self.shared_dict = self.manager.dict({
            "expert_map": None,  #当前rank_id的专家表[num_layers,num_experts]
            "moe_load": None,    #热度负载信息 [num_layers,num_experts]
            "expert_maps": None  #所有的专家表[num_layers, world_size, num_experts]
        })
        self.eplb = EplbProcess(
            device_id=self.device,
            shared_dict=self.shared_dict,
            policy_type=1,
            enable_d2d=True
        )

        self.planner_block_queue, self.block_update_queue, self.eplb_process = \
            self.eplb._launch_process()

        logger.info(f"[ModelRunner] Launched EPLB process (pid={self.eplb_process.pid})")

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The SamplingMetadata is updated and copied to the NPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        removed_req_indices: List[int] = []
        for req_id in scheduler_output.finished_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            if req_index is not None:
                removed_req_indices.append(req_index)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            req_index = self.input_batch.remove_request(req_id)
            assert req_index is not None
            removed_req_indices.append(req_index)

        req_ids_to_add: List[str] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            if sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            self.requests[req_id] = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_inputs=new_req_data.mm_inputs,
                mm_positions=new_req_data.mm_positions,
                sampling_params=sampling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            req_ids_to_add.append(req_id)

        # Update the states of the running/resumed requests.
        for req_data in scheduler_output.scheduled_cached_reqs:
            req_id = req_data.req_id
            req_state = self.requests[req_id]

            # Update the cached states.
            num_computed_tokens = req_data.num_computed_tokens
            req_state.num_computed_tokens = num_computed_tokens
            # Add the sampled token(s) from the previous step (if any).
            # This doesn't include "unverified" tokens like spec decode tokens.
            num_new_tokens = (num_computed_tokens +
                              len(req_data.new_token_ids) -
                              req_state.num_tokens)
            if num_new_tokens == 1:
                # Avoid slicing list in most common case.
                req_state.output_token_ids.append(req_data.new_token_ids[-1])
            elif num_new_tokens > 0:
                req_state.output_token_ids.extend(
                    req_data.new_token_ids[-num_new_tokens:])
            # Update the block IDs.
            if not req_data.resumed_from_preemption:
                # Append the new blocks to the existing block IDs.
                req_state.block_ids.extend(req_data.new_block_ids)
            else:
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = req_data.new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                req_ids_to_add.append(req_id)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)

            start_index = (len(req_state.block_ids) -
                           len(req_data.new_block_ids))
            self.input_batch.block_table.append_row(req_data.new_block_ids,
                                                    req_index)
            # Add new_token_ids to token_ids_cpu.
            start_token_index = num_computed_tokens
            end_token_index = num_computed_tokens + len(req_data.new_token_ids)
            self.input_batch.token_ids_cpu[
                req_index,
                start_token_index:end_token_index] = req_data.new_token_ids
            self.input_batch.num_tokens_no_spec[req_index] = end_token_index
            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = scheduler_output.scheduled_spec_decode_tokens.get(
                req_id, ())
            if spec_token_ids:
                start_index = end_token_index
                end_token_index += len(spec_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
            # NOTE(woosuk): `num_tokens` here may include spec decode tokens.
            self.input_batch.num_tokens[req_index] = end_token_index

        # Check if the batch has changed. If not, we can skip copying the
        # sampling metadata from CPU to GPU.
        batch_changed = len(removed_req_indices) > 0 or len(req_ids_to_add) > 0

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        removed_req_indices = sorted(removed_req_indices, reverse=True)
        for req_id in req_ids_to_add:
            req_state = self.requests[req_id]
            if removed_req_indices:
                # Fill the empty index.
                req_index = removed_req_indices.pop()
            else:
                # Append to the end.
                req_index = None
            self.input_batch.add_request(req_state, req_index)

        # Condense the batched states if there are empty indices.
        if removed_req_indices:
            self.input_batch.condense(removed_req_indices)

        if batch_changed:
            self.input_batch.refresh_sampling_metadata()

    def get_model(self) -> nn.Module:
        return self.model

    def _make_attention_mask(self, seq_lens, query_lens, position,
                             attn_state) -> torch.Tensor:
        # Chunk Prefill situation.
        if attn_state == AscendAttentionState.ChunkedPrefill:
            return self.attn_mask_builder.get_splitfuse_attn_mask(
                seq_lens, query_lens, position, self.dtype, self.device)
        # Prefill without cache situation.
        elif attn_state == AscendAttentionState.PrefillNoCache:
            max_seq_len = max(seq_lens, default=0)
            return self.attn_mask_builder.get_attn_mask(
                max_seq_len, self.dtype, self.device)
        # Prefill with cache hit.
        elif attn_state == AscendAttentionState.PrefillCacheHit:
            return self.attn_mask_builder.get_attn_mask(
                128, self.dtype, self.device)
        # Decode-only situation.
        else:
            return None

    def _process_reqs(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> tuple[SpecDecodeMetadata, torch.Tensor, SpecDecodeMetadata,
               torch.Tensor, int, torch.Tensor]:
        # Check input valid
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0
        if (self.use_aclgraph and
                total_num_scheduled_tokens <= self.aclgraph_batch_sizes[-1]):
            # Add padding to the batch size.
            num_input_tokens = self.vllm_config.pad_for_cudagraph(
                total_num_scheduled_tokens)
        else:
            # Eager mode.
            num_input_tokens = total_num_scheduled_tokens

        modified_batch = self.attn_metadata_builder.reorder_batch(
            self.input_batch, scheduler_output)
        if modified_batch:
            self.input_batch.refresh_sampling_metadata()

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit(num_reqs)

        # Get the number of scheduled tokens for each request.
        # TODO: The Python loop can be slow. Optimize.
        num_scheduled_tokens = np.empty(num_reqs, dtype=np.int32)
        max_num_scheduled_tokens = 0
        for i, req_id in enumerate(self.input_batch.req_ids):
            num_tokens = scheduler_output.num_scheduled_tokens[req_id]
            num_scheduled_tokens[i] = num_tokens
            max_num_scheduled_tokens = max(max_num_scheduled_tokens,
                                           num_tokens)

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, num_scheduled_tokens)

        # Prepare positions
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)
        cu_num_tokens = np.cumsum(num_scheduled_tokens)
        cumsums_offsets = np.repeat(cu_num_tokens - num_scheduled_tokens,
                                    num_scheduled_tokens)
        sample_indices = cu_num_tokens - 1
        sample_indices = torch.from_numpy(sample_indices).to(self.device,
                                                             non_blocking=True)
        arange = self.arange_np[:total_num_scheduled_tokens] - cumsums_offsets

        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        self.positions[:total_num_scheduled_tokens].copy_(
            self.positions_cpu[:total_num_scheduled_tokens], non_blocking=True)
        positions = self.positions[:num_input_tokens]
        self.query_lens = torch.from_numpy(num_scheduled_tokens)

        self.seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        seq_lens = self.seq_lens_cpu[:num_reqs]

        block_table_indices = (req_indices * self.max_num_blocks_per_req +
                               positions_np // self.block_size)

        block_table_cpu = self.input_batch.block_table[0].get_cpu_tensor()
        block_numbers = block_table_cpu.flatten()[block_table_indices].numpy()
        block_offsets = positions_np % self.block_size
        np.add(block_numbers * self.block_size,
               block_offsets,
               out=self.slot_mapping_np[:total_num_scheduled_tokens])

        if np.array_equal(self.seq_lens_np[:num_reqs], num_scheduled_tokens):
            attn_state = AscendAttentionState.PrefillNoCache
        # We assume it is the decode stage, where prefill occurs but only one token is not hit in cache.
        elif np.all(num_scheduled_tokens == 1):
            attn_state = AscendAttentionState.DecodeOnly
        # splitfuse
        elif not self.use_v0_scheduler or self.chunked_prefill_enabled:
            attn_state = AscendAttentionState.ChunkedPrefill
        else:
            attn_state = AscendAttentionState.PrefillCacheHit

        attn_mask = self._make_attention_mask(seq_lens=seq_lens,
                                              query_lens=num_scheduled_tokens,
                                              position=positions,
                                              attn_state=attn_state)
        self.attn_mask = attn_mask
        self.attn_state = attn_state  # type: ignore

        extra_builder_kwargs = {}

        # Add graph_pad_size here
        if self.enable_torchair_graph_mode:
            graph_pad_size = self.scheduler_config.max_num_seqs - len(seq_lens)
            extra_builder_kwargs['graph_pad_size'] = graph_pad_size

        attn_metadata = self.attn_metadata_builder.build(  # type: ignore
            num_reqs=num_reqs,
            num_actual_tokens=total_num_scheduled_tokens,
            max_query_len=max_num_scheduled_tokens,
            common_prefix_len=None,
            **extra_builder_kwargs,
        )
        attn_metadata.num_input_tokens = num_input_tokens

        # Prepare input_ids
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids_cpu[:total_num_scheduled_tokens])
        # Copy the tensors to the NPU.
        self.input_ids[:total_num_scheduled_tokens].copy_(
            self.input_ids_cpu[:total_num_scheduled_tokens], non_blocking=True)
        input_ids = self.input_ids[:num_input_tokens]

        if self.enable_torchair_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
            padding = torch.zeros(graph_pad_size,
                                  dtype=input_ids.dtype,
                                  device=input_ids.device)
            input_ids = torch.cat([input_ids, padding])
            positions = torch.cat([positions, padding])

        # Run forward pass
        with set_forward_context(attn_metadata,
                                 self.vllm_config,
                                 num_tokens=num_input_tokens):
            model_kwargs = {}
            if self.enable_torchair_graph_mode:
                model_kwargs["kv_caches"] = self.kv_caches
                model_kwargs["attn_metadata"] = attn_metadata
            if self.enable_torchair_graph_mode and attn_metadata.attn_state == AscendAttentionState.DecodeOnly:
                torch._dynamo.mark_static(input_ids)
                torch._dynamo.mark_static(positions)
                torch._dynamo.mark_static(attn_metadata.decode.block_table)
                torch._dynamo.mark_static(attn_metadata.decode.input_positions)
                torch._dynamo.mark_static(attn_metadata.slot_mapping)
                for kv in self.kv_caches:
                    if isinstance(kv, tuple):
                        torch._dynamo.mark_static(kv[0])
                        torch._dynamo.mark_static(kv[1])
                hidden_states = self.compile_model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=None,
                    **model_kwargs,
                )
            else:
                assert self.model is not None
                hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=None,
                    **model_kwargs,
                )

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            sample_indices = spec_decode_metadata.logits_indices

        return (attn_metadata, hidden_states, spec_decode_metadata, positions,
                total_num_scheduled_tokens, sample_indices)

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1
        # Step 1. [4, 5, 8, 9, 11]
        cu_num_sampled_tokens = np.cumsum(num_sampled_tokens, dtype=np.int32)
        total_num_sampled_tokens = cu_num_sampled_tokens[-1]
        # Step 2. [0, 0, 0, 0, 4, 5, 5, 5, 8, 9, 9]
        cumsums_offsets = np.repeat(cu_num_sampled_tokens - num_sampled_tokens,
                                    num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        arange = self.arange_np[:total_num_sampled_tokens] - cumsums_offsets
        # Step 4. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 5. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # [3, 3, 5, 5, 6]
        cu_num_draft_tokens = np.cumsum(num_draft_tokens, dtype=np.int32)
        total_num_draft_tokens = cu_num_draft_tokens[-1]
        # [0, 0, 0, 3, 3, 5]
        cumsums_offsets = np.repeat(cu_num_draft_tokens - num_draft_tokens,
                                    num_draft_tokens)
        # [0, 1, 2, 0, 1, 0]
        arange = self.arange_np[:total_num_draft_tokens] - cumsums_offsets
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> NPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        metadata = SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )
        return metadata

    def apply_grammar_bitmask(
        self,
        scheduler_output: "SchedulerOutput",
        logits: torch.Tensor,
    ) -> torch.Tensor:
        # Serialization of np.ndarray is much more efficient than a tensor,
        # so we receive it in that format.
        grammar_bitmask = scheduler_output.grammar_bitmask
        if grammar_bitmask is None:
            return

        # We receive the structured output bitmask from the scheduler, but the
        # indices of the requests in the batch may not match the indices of
        # the bitmask since the scheduler doesn't know how the gpu runner is
        # ordering the requests in the batch. We need to sort the bitmask to
        # match the order of the requests used here.
        struct_out_req_batch_indices: dict[str, int] = {}
        indices_match = True
        for req_id in self.input_batch.req_ids:
            mask_index = scheduler_output.structured_output_request_ids.get(
                req_id)
            if mask_index is None:
                # not a structured output request
                continue
            batch_index = self.input_batch.req_id_to_index[req_id]
            if batch_index != mask_index:
                indices_match = False
            struct_out_req_batch_indices[req_id] = batch_index

        if not indices_match:
            # Sort the bitmask to match the order of the requests
            sorted_bitmask = np.zeros_like(grammar_bitmask)
            for req_id, batch_index in struct_out_req_batch_indices.items():
                orig_index = scheduler_output.structured_output_request_ids[
                    req_id]
                sorted_bitmask[batch_index] = grammar_bitmask[orig_index]
            grammar_bitmask = sorted_bitmask

        grammar_bitmask = torch.from_numpy(grammar_bitmask)

        # TODO: compatibility with spec decode.
        # NOTE:
        # 1. XGrammar bitmask applying only supports CPU and GPU.
        # 2. The logits and bitmask should be on the same device.
        # 3. XGrammar logits on CPU only supports float32 dtype.
        logits_dtype = logits.dtype
        logits = logits.to("cpu").float()
        xgr.apply_token_bitmask_inplace(
            logits,
            grammar_bitmask,
            indices=list(struct_out_req_batch_indices.values()),
        )
        return logits.to(self.device).to(logits_dtype)

    def _get_spec_token_ids(
        self,
        valid_sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
        scheduler_output: "SchedulerOutput",
        spec_decode_metadata: SpecDecodeMetadata,
        positions: torch.Tensor,
        num_scheduled_tokens: int,
        hidden_states: torch.Tensor,
        attn_metadata: SpecDecodeMetadata,
    ) -> Optional[list[list[int]]]:
        if not self.use_spec_decode:
            # Speculative decoding is not enabled.
            spec_token_ids = None
        elif self.speculative_config.method == "ngram":
            assert isinstance(self.drafter, NgramProposer)
            spec_token_ids = self._generate_draft_token_ids(
                valid_sampled_token_ids, sampling_metadata)
        elif self.speculative_config.method == "eagle":
            raise NotImplementedError(
                "eagle method for spec decode doesn't work on vllm-ascend currently"
            )
        return spec_token_ids

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, torch.Tensor]:
        self._update_states(scheduler_output)
        if not scheduler_output.total_num_scheduled_tokens:
            # Return empty ModelRunnerOuptut if there's no work to do.
            return EMPTY_MODEL_RUNNER_OUTPUT
        # TO DO: read shared memory from asyn process
        # self.eplb_loader.generate_expert_d2d_transfer_task
        # Run a demo for testing D2D transfering
        if self.eplb_loader.mock_flag:
            rank_id = torch.distributed.get_rank()
            (expert_transfer_info, expert_pull_info, updated_expert_map, layer_id) = \
                self.eplb_loader.generate_mock_update_info(rank_id)
            self.eplb_loader.generate_expert_d2d_transfer_task(expert_transfer_info,
                expert_pull_info, updated_expert_map, layer_id)
        reqs = []
        self.eplb_loader.asyn_expert_weight_transfer(reqs)
        (attn_metadata, hidden_states, spec_decode_metadata, positions,
         num_scheduled_tokens,
         sample_indices) = (self._process_reqs(scheduler_output,
                                               intermediate_tensors))
        logits = self.model.compute_logits(hidden_states[sample_indices], None)
        self.eplb_loader.update_expert_map_and_weight(reqs)

        # Apply structured output bitmasks if present
        if scheduler_output.grammar_bitmask is not None:
            logits = self.apply_grammar_bitmask(scheduler_output, logits)

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            # When indexing with a tensor (bonus_logits_indices), PyTorch
            # creates a new tensor with separate storage from the original
            # logits tensor. This means any in-place operations on bonus_logits
            # won't affect the original logits tensor.
            bonus_logits = logits[spec_decode_metadata.bonus_logits_indices]
            sampler_output = self.sampler(
                logits=bonus_logits,
                sampling_metadata=sampling_metadata,
            )
            bonus_token_ids = sampler_output.sampled_token_ids

            # Just like `bonus_logits`, `target_logits` is a new tensor with
            # separate storage from the original `logits` tensor. Therefore,
            # it is safe to update `target_logits` in place.
            target_logits = logits[spec_decode_metadata.target_logits_indices]
            output_token_ids = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                target_logits,
                bonus_token_ids,
                sampling_metadata,
            )
            sampler_output.sampled_token_ids = output_token_ids

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token.
                # Rewind the generator state as if the token was not sampled.
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)

        # NOTE: NPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Get the valid generated tokens.
        sampled_token_ids = sampler_output.sampled_token_ids
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids = sampled_token_ids.tolist()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids = self.rejection_sampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
            )

        spec_token_ids = self._get_spec_token_ids(
            valid_sampled_token_ids,
            sampling_metadata,
            scheduler_output,
            spec_decode_metadata,
            positions,
            num_scheduled_tokens,
            hidden_states,
            attn_metadata,
        )

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=valid_sampled_token_ids,
            spec_token_ids=spec_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict={},
        )

        if self.enable_eplb:
            self.do_eplb()

        return model_runner_output

    def do_eplb(self):
        if not self.update_in_flight:
            try:
                if not self.expert_map_initialized:
                    self.get_expert_map()
                    self.expert_map_initialized = True

                moe_load = self.compute_and_set_moe_load()

                self.planner_block_queue.put(1)
                self.update_in_flight = True

            except Exception as e:
                logger.warning(f"[ModelRunner] Failed to wake EPLB process: {e}", exc_info=True)

        if  self.update_in_flight:
                if not self.block_update_queue.empty():
                    self.block_update_queue.get()
                    rank_id = dist.get_rank()
                    new_expert_map = self.shared_dict["expert_maps"][:, rank_id, :]
                    self.model.update_all_expert_map(new_expert_map, self.num_moe_layers)
                    #加载权重
                    self.update_in_flight = False

    def compute_and_set_moe_load(self):
        moe_load = self.model.get_all_moe_loads(self.num_moe_layers)

        import torch.distributed as dist
        if dist.is_initialized():
            dist.all_reduce(moe_load, op=dist.ReduceOp.SUM)
        self.shared_dict["moe_load"] = moe_load.to(torch.device("cpu"))

        logger.debug(f"[ModelRunner] Updated shared_dict['moe_load'] = {moe_load}")

        return moe_load


    def get_expert_map(self):
        expert_map = self.model.get_all_expert_map(self.num_moe_layers)
        if dist.is_initialized():
                    world_size = dist.get_world_size()
        rank = dist.get_rank()

        tensor_list = [
            torch.zeros_like(expert_map) for _ in range(world_size)
        ]

        dist.all_gather(tensor_list, expert_map)
        gathered = torch.stack(tensor_list, dim=0)
        all_maps = gathered.permute(1, 0, 2).contiguous()

        all_expert_maps = all_maps.to(torch.device("cpu"))
        self.shared_dict["expert_maps"] = all_expert_maps
        logger.debug(f"[ModelRunner] Updated shared_dict['expert_map'] = {expert_map}")
        return all_expert_maps

    def shutdown(self):
        """
        Clean up the EPLB process.
        """
        if self.eplb_process.is_alive():
            self.eplb_process.terminate()
            self.eplb_process.join()
            logger.info("[ModelRunner] EPLB process terminated")

    def _profile_multimodal(self) -> None:
        # TODO: handle encoder-decoder models once we support them.
        # NOTE: Currently model is profiled with a single non-text
        # modality with the max possible input tokens even when
        # it supports multiple.

        if (not self.is_multimodal_model
                or self.max_num_encoder_input_tokens <= 0
                or self.encoder_cache_size <= 0):
            return

        max_tokens_by_modality_dict = (
            MULTIMODAL_REGISTRY.get_max_tokens_per_item_by_nonzero_modality(
                self.model_config))
        dummy_data_modality, max_tokens_per_mm_item = max(
            max_tokens_by_modality_dict.items(), key=lambda item: item[1])

        # Check how many items of this modality can be supported by
        # the encoder budget.
        encoder_budget = min(self.max_num_encoder_input_tokens,
                             self.encoder_cache_size)

        max_num_mm_items_encoder_budget = cdiv(encoder_budget,
                                               max_tokens_per_mm_item)

        # Check how many items of this modality can be supported by
        # the decoder budget.
        max_mm_items_per_req = self.mm_registry.get_mm_limits_per_prompt(
            self.model_config)[dummy_data_modality]

        # NOTE: We do not consider max_num_batched_tokens on purpose
        # because the multimodal embeddings can be generated in advance
        # and chunked prefilled.
        max_num_mm_items_decoder_budget = self.max_num_reqs * \
            max_mm_items_per_req

        max_num_mm_items = min(max_num_mm_items_encoder_budget,
                               max_num_mm_items_decoder_budget)

        logger.info(
            "Encoder cache will be initialized with a budget of %s tokens,"
            " and profiled with %s %s items of the maximum feature size.",
            encoder_budget, max_num_mm_items, dummy_data_modality)

        # Create dummy batch of multimodal inputs.
        dummy_request_data = self.input_registry.dummy_data_for_profiling(
            model_config=self.model_config,
            seq_len=self.max_num_tokens,
            mm_registry=self.mm_registry,
        )
        dummy_mm_data = dummy_request_data.multi_modal_data

        if not isinstance(dummy_mm_data, MultiModalKwargs):
            # TODO: Delete this check once input mapper is fully removed.
            raise RuntimeError("Legacy input mapper is not supported in V1")

        # Dummy data definition in V0 may contain multiple multimodal items
        # (e.g, multiple images) for a single request, therefore here we
        # always replicate first item by max_num_mm_items times since in V1
        # they are scheduled to be processed separately.

        dummy_mm_item = dummy_mm_data.get_item(modality=dummy_data_modality,
                                               item_index=0)
        dummy_mm_kwargs = MultiModalKwargs.from_items([dummy_mm_item])

        batched_dummy_mm_inputs = MultiModalKwargs.batch([dummy_mm_kwargs] *
                                                         max_num_mm_items)
        batched_dummy_mm_inputs = MultiModalKwargs.as_kwargs(
            batched_dummy_mm_inputs, device=self.device)

        # Run multimodal encoder.
        dummy_encoder_outputs = self.model.get_multimodal_embeddings(
            **batched_dummy_mm_inputs)
        assert len(dummy_encoder_outputs) == max_num_mm_items, (
            "Expected dimension 0 of encoder outputs to match the number "
            f"of multimodal data items: {max_num_mm_items}, got "
            f"{len(dummy_encoder_outputs)=} instead. This is most likely "
            "due to the 'get_multimodal_embeddings' method of the model "
            "not implemented correctly.")

        # Cache the dummy encoder outputs.
        self.encoder_cache["tmp"] = dict(enumerate(dummy_encoder_outputs))

    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int) -> torch.Tensor:
        # Set num_scheduled_tokens based on num_tokens and max_num_seqs
        # for dummy run with LoRA so that the num_reqs collectively
        # has num_tokens in total.
        assert num_tokens <= self.scheduler_config.max_num_batched_tokens
        max_num_reqs = self.scheduler_config.max_num_seqs
        num_reqs = max_num_reqs if num_tokens >= max_num_reqs else num_tokens
        min_tokens_per_req = num_tokens // num_reqs
        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        with self.maybe_dummy_run_with_lora(self.lora_config,
                                            num_scheduled_tokens):
            model = self.model
            if self.is_multimodal_model:
                input_ids = None
                inputs_embeds = self.inputs_embeds[:num_tokens]
            else:
                input_ids = self.input_ids[:num_tokens]
                inputs_embeds = None

            if self.uses_mrope:
                positions = self.mrope_positions[:, :num_tokens]
            else:
                positions = self.positions[:num_tokens]

            if get_pp_group().is_first_rank:
                intermediate_tensors = None
            else:
                if self.intermediate_tensors is None:
                    self.intermediate_tensors = (
                        self.model.make_empty_intermediate_tensors(
                            batch_size=num_tokens,
                            dtype=self.dtype,
                            device=self.device))
                intermediate_tensors = IntermediateTensors({
                    k: v[:num_tokens]
                    for k, v in self.intermediate_tensors.items()
                })

            with set_forward_context(None, self.vllm_config):
                hidden_states = model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds)
            return hidden_states

    def profile_run(self) -> None:
        # Profile with multimodal encoder & encoder cache.
        self._profile_multimodal()

        # For profile, have maximum num_reqs and that collectively have
        # maximum num_tokens.
        num_reqs = self.scheduler_config.max_num_seqs
        num_tokens = self.max_num_tokens
        min_tokens_per_req = num_tokens // num_reqs

        num_scheduled_tokens_list = [min_tokens_per_req] * num_reqs
        num_scheduled_tokens_list[-1] += num_tokens % num_reqs
        assert sum(num_scheduled_tokens_list) == num_tokens
        assert len(num_scheduled_tokens_list) == num_reqs

        num_scheduled_tokens = np.array(num_scheduled_tokens_list,
                                        dtype=np.int32)
        logit_indices = np.cumsum(num_scheduled_tokens) - 1

        # assert self.lora_manager is not None, "LoRA is not enabled"
        # TODO: call maybe_profile_with_lora()

        dummy_kv_caches = [
            torch.tensor((), dtype=torch.float32, device=self.device)
            for _ in range(self.num_attn_layers)
        ]

        # Trigger compilation for general shape.
        hidden_states = self._dummy_run(self.max_num_tokens)

        if get_pp_group().is_last_rank:
            hidden_states = hidden_states[logit_indices]
            logits = self.model.compute_logits(hidden_states, None)
        else:
            logits = None

        NPUPlatform.synchronize()
        del hidden_states, logits, dummy_kv_caches
        self.encoder_cache.clear()
        gc.collect()

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)

        with DeviceMemoryProfiler() as m:  # noqa: SIM117
            self.model = get_model(vllm_config=self.vllm_config)
            self.eplb_loader = D2DExpertWeightLoader(model=self.model)
            if hasattr(self, "drafter"):
                logger.info("Loading drafter model...")
                self.drafter.load_model(self.model)
            if self.lora_config:
                self.model = self.load_lora_model(self.model,
                                                  self.model_config,
                                                  self.scheduler_config,
                                                  self.lora_config,
                                                  self.device)
        logger.info("Loading model weights took %.4f GB",
                    m.consumed_memory / float(2**30))

        # adapter torch compile with npu_backend
        if self.enable_torchair_graph_mode:
            import torchair  # type: ignore
            from torchair import patch_for_hcom  # type: ignore

            patch_for_hcom()
            config = torchair.CompilerConfig()
            config.experimental_config.frozen_parameter = True
            config.experimental_config.tiling_schedule_optimize = True
            torch.npu.set_compile_mode(jit_compile=False)
            if not self.use_cached_npu_graph:
                npu_backend = torchair.get_npu_backend(compiler_config=config)
                self.compile_model = torch.compile(
                    self.model,
                    dynamic=True,
                    fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    backend=npu_backend)
            else:
                self.compile_model = torchair.inference.cache_compile(
                    self.model.forward,
                    dynamic=True,
                    fullgraph=envs.VLLM_TEST_DYNAMO_FULLGRAPH_CAPTURE,
                    config=config,
                    ge_cache=False)

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        import torch_npu
        kv_caches: Dict[str, torch.Tensor] = {}

        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.model_config.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=True,
            vocab_size=self.model_config.get_vocab_size(),
            block_size=self.cache_config.block_size,
        )

        for kv_cache_group in kv_cache_config.kv_cache_groups:
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                tensor_config = kv_cache_config.tensors[layer_name]
                assert tensor_config.size % kv_cache_spec.page_size_bytes == 0
                num_blocks = tensor_config.size // kv_cache_spec.page_size_bytes
                # `num_blocks` is the number of blocks the model runner can use.
                # `kv_cache_config.num_blocks` is the number of blocks that
                # KVCacheManager may allocate.
                # Since different GPUs may have different number of layers and
                # different memory capacities, `num_blocks` can be different on
                # different GPUs, and `kv_cache_config.num_blocks` is set to
                # the min of all `num_blocks`. Verify it here.
                assert num_blocks >= kv_cache_config.num_blocks
                # TODO: remove this after the OOM issue is located and fixed, otherwise, some model may
                # encounter OOM issue
                if isinstance(kv_cache_spec, FullAttentionSpec):
                    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
                        num_blocks, kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, kv_cache_spec.head_size)
                    dtype = kv_cache_spec.dtype
                    if self.enable_torchair_graph_mode:
                        layer_kv_cache_nope = torch.zeros(
                            kv_cache_shape[:-1] +
                            (self.model_config.hf_text_config.kv_lora_rank, ),
                            dtype=self.dtype,
                            pin_memory=True,
                            device=self.device)
                        layer_kv_cache_pe = torch.zeros(
                            kv_cache_shape[:-1] +
                            (self.model_config.hf_text_config.qk_rope_head_dim,
                             ),
                            dtype=self.dtype,
                            pin_memory=True,
                            device=self.device)
                        kv_caches[layer_name] = (layer_kv_cache_nope,
                                                 layer_kv_cache_pe)
                        torch_npu.npu_format_cast(kv_caches[layer_name][0], 2)
                        torch_npu.npu_format_cast(kv_caches[layer_name][1], 2)
                    else:
                        kv_caches[layer_name] = torch.zeros(kv_cache_shape,
                                                            dtype=dtype,
                                                            device=self.device)
                        torch_npu.npu_format_cast(kv_caches[layer_name], 2)
                else:
                    # TODO: add new branches when introducing more types of
                    # KV cache specs.
                    raise ValueError("Unknown KV cache spec type.")

        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        forward_ctx = self.vllm_config.compilation_config.static_forward_context
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in forward_ctx.items():
            if isinstance(attn_module, FusedMoE):
                continue

            # TODO: Support other attention modules, e.g., sliding window,
            # cross-attention
            assert isinstance(attn_module, Attention)
            if attn_module.attn_type == AttentionType.DECODER:
                kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=attn_module.dtype,
                    use_mla=use_mla)
            elif attn_module.attn_type in (AttentionType.ENCODER,
                                           AttentionType.ENCODER_ONLY):
                # encoder-only attention does not need KV cache.
                continue
            elif attn_module.attn_type == AttentionType.ENCODER_DECODER:
                raise NotImplementedError
            else:
                raise ValueError(
                    f"Unknown attention type: {attn_module.attn_type}")

        return kv_cache_spec

    def capture_model(self) -> None:
        if not self.use_aclgraph:
            logger.warning(
                "Skipping NPU graph capture. Please add "
                "-O %s to use NPU graphs.", CompilationLevel.PIECEWISE)
            return

        start_time = time.perf_counter()
        start_free_npu_memory = torch.npu.mem_get_info()[0]

        # Trigger ACL graph capture for specific shapes.
        # Capture the large shapes first so that the smaller shapes
        # can reuse the memory pool allocated for the large shapes.
        with graph_capture(device=self.device):
            for num_tokens in reversed(self.aclgraph_batch_sizes):
                for _ in range(self.vllm_config.compilation_config.
                               cudagraph_num_of_warmups):
                    self._dummy_run(num_tokens)
                self._dummy_run(num_tokens)

        end_time = time.perf_counter()
        end_free_npu_memory = torch.npu.mem_get_info()[0]
        elapsed_time = end_time - start_time
        npu_graph_size = start_free_npu_memory - end_free_npu_memory
        # This usually takes 5~20 seconds.
        logger.info("Graph capturing finished in %.0f secs, took %.2f GiB",
                    elapsed_time, npu_graph_size / (1 << 30))

    def _generate_draft_token_ids(
        self,
        sampled_token_ids: list[list[int]],
        sampling_metadata: SamplingMetadata,
    ) -> list[list[int]]:
        # TODO(woosuk): Optimize.
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            num_sampled_ids = len(sampled_ids)
            if not num_sampled_ids:
                # Skip speculative decoding.
                draft_token_ids.append([])
                continue

            # Skip requests that require top-p, top-k, etc.
            req_id = self.input_batch.req_ids[i]
            if not is_spec_decode_supported(req_id, self.input_batch):
                draft_token_ids.append([])
                continue

            # Add sampled_token_ids to token_ids_cpu.
            start_idx = self.input_batch.num_tokens_no_spec[i]
            end_idx = start_idx + num_sampled_ids
            self.input_batch.token_ids_cpu[i, start_idx:end_idx] = sampled_ids
            drafter_output = self.drafter.propose(
                self.input_batch.token_ids_cpu[i, :end_idx])
            if drafter_output is None or len(drafter_output) == 0:
                draft_token_ids.append([])
            else:
                draft_token_ids.append(drafter_output.tolist())
        return draft_token_ids
