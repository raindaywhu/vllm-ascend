#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
#

import time
import numpy as np
import networkx as nx
import torch
import torch_npu
import logging
import torch.distributed as dist
from multiprocessing import Process, Queue, Manager
from abc import ABC, abstractmethod
from vllm.logger import logger
from collections import deque

from vllm_ascend.eplb.core.policy.policy_factory import PolicyFactory, DynamicConfig
from vllm_ascend.eplb.tool.eplb_utils import ExpertMapUtils


class EplbWorker:

    def __init__(self, shared_dict, policy_type, enable_d2d: bool = True, redundant_enable=0):
        self.policy_type = policy_type
        self.policy = PolicyFactory.generate_policy(policy_type, DynamicConfig())
        self.shared_dict = shared_dict
        self.old_expert_maps = None
        self.enable_d2d = enable_d2d
        self.redundant_enable = redundant_enable
        self.rank_id  = dist.get_rank()

    def do_update(self):
        # put data in to queue
        # in process self.policy.generate_policy()
        # get epxert table && tensor

        # async stream
        # D2D
        # H2D

        # Get initial expert_map
        if self.old_expert_maps is None:
            self.old_expert_maps = self.get_init_expert_maps()
            self.num_local_experts = self.old_expert_maps.max() + 1

        # Get MOE load information
        load_info = self.fetch_and_sum_load_info()
        if load_info is None:
            return

        #根据负载信息，获取更新后的专家表
        old_placement = self.global2local(self.old_expert_maps, self.num_local_experts)
        changed, priority, new_placement = self.calculate_rebalance_experts(load_info, old_placement)

        if not torch.is_tensor(new_placement):
            new_placement = torch.tensor(new_placement)
        self.check_expert_placement(old_placement, new_placement)
        new_expert_maps = self.local2global(new_placement)
        self.update_expert_map(new_expert_maps)
        logger.debug(f"[EPLB Process  new_map differs, performing D2D")

        update_info = self.compose_expert_update_info_bipartite(new_placement, old_placement, new_expert_maps)\
            if self.policy_type <= 2 else self.compose_expert_update_info_greedy(new_expert_maps, self.old_expert_maps)
        self.old_expert_maps = new_expert_maps
        logger.info("EPLB Process compute complete")

        packed_update_info = self.pack_update_info(update_info)

        return packed_update_info

    def check_expert_placement(self, old_placement, new_placement):
        num_layers = old_placement.shape[0]
        num_ranks = old_placement.shape[1]

        for layer_id in range(num_layers):
            # check if any logical expert is not placed on any rank
            if torch.unique(new_placement[layer_id]).numel() < torch.unique(old_placement[layer_id]).numel():
                logger.error(f"There exists expert not placed on any rank in layer {layer_id}")
                new_placement[layer_id] = old_placement[layer_id]
                continue

            for rank_id in range(num_ranks):
                new_placement_check = new_placement[layer_id][rank_id]
                old_placement_check = old_placement[layer_id][rank_id]

                # check if same logical experts are placed on the same NPU
                if new_placement_check.numel() != torch.unique(new_placement_check).numel():
                    logger.error(f"Replicated experts are placed on the same NPU, expert placement on layer {layer_id}, rank {rank_id} is invalid")
                    new_placement[layer_id] = old_placement[layer_id]
                    break

                # check if there is any experts movement inside one NPU
                expert_not_move = torch.isin(new_placement_check, old_placement_check)
                if not torch.equal(new_placement_check[expert_not_move], old_placement_check[expert_not_move]):
                    logger.error(f"There exists expert movement inside NPU, expert placement on layer {layer_id}, rank {rank_id} is invalid")
                    new_placement[layer_id] = old_placement[layer_id]
                    break

    def dfs(self, u, visited, match, graph):
        visited[u] = True
        for v in graph[u]:
            if not visited[v]:
                visited[v] = True
                if match[v] == -1 or self.dfs(match[v], visited, match, graph):
                    match[v] = u
                    match[u] = v
                    return True
        return False

    def compose_expert_update_info_bipartite(self, updated_expert_maps_org, current_expert_maps_org, new_expert_maps):

        updated_expert_maps = updated_expert_maps_org.clone()
        current_expert_maps = current_expert_maps_org.clone()
        updated_expert_maps = np.array(updated_expert_maps)
        current_expert_maps = np.array(current_expert_maps)

        num_layers = current_expert_maps.shape[0]
        num_ranks = updated_expert_maps.shape[1]

        for layer_id in range(num_layers):

            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]
            updated_expert_maps_this_layer_org = new_expert_maps[layer_id]

            expert_send_info_this_layer = dict()
            expert_recv_info_this_layer = dict()

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if (np.equal(updated_expert_maps_this_layer,
                         current_expert_maps_this_layer)).all():
                yield (expert_send_info_this_layer, expert_recv_info_this_layer,
                       updated_expert_maps_this_layer_org, layer_id)

            # 解析每个rank需要接收的专家ID
            dst_ranks_needed_experts = []
            for rank_id in range(updated_expert_maps_this_layer.shape[0]):
                for index in range(updated_expert_maps_this_layer.shape[1]):
                    if current_expert_maps_this_layer[rank_id][index] != updated_expert_maps_this_layer[rank_id][index]:
                        # 接收卡id为 rank索引
                        dst_ranks_needed_experts.append((rank_id, updated_expert_maps_this_layer[rank_id][index]))

            # 为每个专家构建源rank集合
            src_ranks_set = {}
            for dst_ranks_info in dst_ranks_needed_experts:
                expert_id = dst_ranks_info[1]
                if expert_id not in src_ranks_set:
                    src_ranks_set[expert_id] = np.where(np.any(current_expert_maps_this_layer == expert_id, axis=1))[0]

            # 处理直到没有需要接收的专家
            while dst_ranks_needed_experts:
                # 初始化二分图
                graph = [[] for _ in range(2 * num_ranks)]
                edge_added = [[False] * num_ranks for _ in range(num_ranks)]

                # 构建二分图
                for dst_info in dst_ranks_needed_experts:
                    d = dst_info[0]  # 目标rank
                    e = dst_info[1]  # 专家ID
                    if e in src_ranks_set:
                        for s in src_ranks_set[e]:
                            if not edge_added[s][d]:
                                graph[s].append(d + num_ranks)
                                graph[d + num_ranks].append(s)
                                edge_added[s][d] = True

                # 查找连通分量
                visited_comp = [False] * (2 * num_ranks)
                components = []
                for i in range(2 * num_ranks):
                    if not visited_comp[i] and graph[i]:
                        comp = []
                        q = deque([i])
                        visited_comp[i] = True
                        while q:
                            u = q.popleft()
                            comp.append(u)
                            for v in graph[u]:
                                if not visited_comp[v]:
                                    visited_comp[v] = True
                                    q.append(v)
                        components.append(comp)

                # 为每个连通分量查找最大匹配
                match = [-1] * (2 * num_ranks)
                for comp in components:
                    comp_set = set(comp)
                    comp_graph = [[] for _ in range(2 * num_ranks)]
                    for u in comp:
                        for v in graph[u]:
                            if v in comp_set:
                                comp_graph[u].append(v)

                    # 对左侧节点运行DFS（匈牙利算法）
                    for u in comp:
                        if u < num_ranks:  # 左侧节点
                            visited_dfs = [False] * (2 * num_ranks)
                            self.dfs(u, visited_dfs, match, comp_graph)

                # 收集所有匹配
                all_matches = {}
                for v in range(num_ranks, 2 * num_ranks):
                    if match[v] != -1:
                        s = match[v]  # 源rank
                        d = v - num_ranks  # 目标rank
                        all_matches[s] = d

                # 处理匹配并更新专家信息
                for source_rank_id, target_rank_id in all_matches.items():
                    source_rank_placement = current_expert_maps_this_layer[source_rank_id]

                    # 获取目标rank需要的专家
                    needed_experts = []
                    for dst_info in dst_ranks_needed_experts:
                        if dst_info[0] == target_rank_id:
                            needed_experts.append(dst_info[1])

                    # 找到源rank中目标rank需要的专家
                    expert_id = np.intersect1d(needed_experts, source_rank_placement)
                    if expert_id.size > 0:
                        recv_expert_id = expert_id[0]
                        source_rank_id = int(source_rank_id)
                        target_rank_id = int(target_rank_id)
                        recv_expert_id = int(recv_expert_id)
                        # record send/rcv pairs
                        if source_rank_id not in expert_send_info_this_layer:
                            expert_send_info_this_layer[source_rank_id] = []
                        if target_rank_id not in expert_recv_info_this_layer:
                            expert_recv_info_this_layer[target_rank_id] = []
                        #local_recv_expert_id = np.where(current_expert_maps_this_layer[source_rank_id] == recv_expert_id)[0][0]
                        expert_send_info_this_layer[source_rank_id].append((target_rank_id, recv_expert_id))
                        expert_recv_info_this_layer[target_rank_id].append((source_rank_id, recv_expert_id))

                        # 从需求列表中删除已处理的专家
                        idx_to_remove = None
                        for idx, dst_info in enumerate(dst_ranks_needed_experts):
                            if dst_info[0] == target_rank_id and dst_info[1] == recv_expert_id:
                                idx_to_remove = idx
                                break
                        if idx_to_remove is not None:
                            dst_ranks_needed_experts.pop(idx_to_remove)
            yield (expert_send_info_this_layer, expert_recv_info_this_layer,
            updated_expert_maps_this_layer_org, layer_id)

    # TODO: Here only expert weight exchange is considered, need to be extended to cover other weight update cases
    def compose_expert_update_info_greedy(self, updated_expert_maps, current_expert_maps):
        num_layers = current_expert_maps.shape[0]
        num_ranks = current_expert_maps.shape[1]
        num_experts = current_expert_maps.shape[2]

        for layer_id in range(num_layers):
            updated_expert_maps_this_layer = updated_expert_maps[layer_id]
            current_expert_maps_this_layer = current_expert_maps[layer_id]

            expert_send_info_this_layer = dict()
            expert_recv_info_this_layer = dict()

            # Guard Clause: if there is no expert weight update, avoid subsequent processing
            if torch.equal(updated_expert_maps_this_layer, current_expert_maps_this_layer):
                yield (expert_send_info_this_layer, expert_recv_info_this_layer, updated_expert_maps_this_layer, layer_id)

            # Parse expert_ids each rank needs to receive from other ranks
            dst_rank_indices, experts_to_recv = torch.where((current_expert_maps_this_layer == -1) \
                & (updated_expert_maps_this_layer != -1))

            # Parse expert_ids each rank needs to send to other ranks
            src_rank_indices, experts_to_send = torch.where((current_expert_maps_this_layer != -1) \
                & (updated_expert_maps_this_layer == -1))

            for idx in range(len(dst_rank_indices)):
                dst_rank_id = dst_rank_indices[idx].item()
                expert_id = experts_to_recv[idx].item()
                if dst_rank_id not in expert_recv_info_this_layer:
                    expert_recv_info_this_layer[dst_rank_id] = []

                if not torch.isin(torch.tensor(expert_id), experts_to_send).any():
                    # if expert_id are not sent out from any npu, it will be copied from one npu holding this expert
                    candidate_src_rank_indices = torch.where(current_expert_maps_this_layer[:, expert_id] != -1)[0]
                else:
                    candidate_src_rank_indices = src_rank_indices[experts_to_send == expert_id]

                #TODO: improve selection criterion of npu sending expert_id considering such as intra-node or inter-node...
                src_rank_id = candidate_src_rank_indices[0].item()
                if src_rank_id not in expert_send_info_this_layer:
                    expert_send_info_this_layer[src_rank_id] = []

                expert_send_info_this_layer[src_rank_id].append((dst_rank_id, expert_id))
                expert_recv_info_this_layer[dst_rank_id].append((src_rank_id, expert_id))

            yield (expert_send_info_this_layer, expert_recv_info_this_layer, updated_expert_maps_this_layer, layer_id)


    def calculate_rebalance_experts(self, load_info, old_placement):
        """
        通过 policy 实例的 rebalance_experts 方法计算 new_map。
        """
        if self.old_expert_maps is None:
            return False, None, None

        changed, priority, new_map = self.policy.rebalance_experts(old_placement, load_info)
        return changed, priority, new_map

    def get_init_expert_maps(self):
        """
        Read the initial expert_map from shared_dict.
        """
        return self.shared_dict.get("expert_maps", None)

    def fetch_and_sum_load_info(self):
        """
        Each time the subprocess is awakened, read the latest moe_load
        (shape: [num_moe_layers, num_experts_per_layer]) from shared_dict.
        """
        return self.shared_dict.get("moe_load", None)

    def update_expert_map(self, expert_maps):

        self.shared_dict["expert_maps"] = expert_maps

    def global2local(self,
        placement: torch.Tensor,
        E_local: int
    ) -> tuple[torch.Tensor, torch.Tensor]:

        L, G, _ = placement.shape
        device = placement.device

        pt_local = torch.full((L, G, E_local),
                              fill_value=-1,
                              dtype=torch.long,
                              device=device)

        valid = placement >= 0
        l_idx, g_idx, k_idx = valid.nonzero(as_tuple=True)

        slot_idx = placement[l_idx, g_idx, k_idx]

        pt_local[l_idx, g_idx, slot_idx] = k_idx

        return pt_local


    def local2global(self,
        placement_local: torch.Tensor
    ) -> torch.Tensor:

        L, G, E_local = placement_local.shape
        device = placement_local.device

        max_id = torch.max(placement_local)
        E_global = (max_id + 1).item() if max_id >= 0 else 0

        if E_global == 0:
            return torch.empty((L, G, 0), dtype=torch.long, device=device)

        placement_global = torch.full((L, G, E_global),
                                      fill_value=-1,
                                      dtype=torch.long,
                                      device=device)

        valid = placement_local >= 0
        l_idx, g_idx, slot_idx = valid.nonzero(as_tuple=True)
        gid_idx = placement_local[l_idx, g_idx, slot_idx]

        placement_global[l_idx, g_idx, gid_idx] = slot_idx

        return placement_global

    def pack_update_info(self, update_info_generator):
        """
        Pack a list of update info tuples for efficient IPC.
        """
        send_all = []
        recv_all = []
        maps = []
        log2phy_all = []
        layer_ids = []

        for send_info, recv_info, new_expert_map, layer_id in update_info_generator:

            send_info_this_rank = send_info[self.rank_id] if self.rank_id in send_info else []
            recv_info_this_rank = recv_info[self.rank_id] if self.rank_id in recv_info else []
            send_all.append(send_info_this_rank)
            recv_all.append(recv_info_this_rank)

            maps.append(new_expert_map[self.rank_id].numpy().tolist())

            if self.redundant_enable:
                log2phy_map = ExpertMapUtils.generate_log2phy_map(new_expert_map)
                log2phy_all.append(log2phy_map[self.rank_id].numpy().tolist())
            else:
                log2phy_all.append([])

            layer_ids.append(layer_id)

        return list(zip(send_all, recv_all, maps, log2phy_all, layer_ids))

class EplbProcess:
    def __init__(self, shared_dict, planner_q, block_update_q, redundant_enable, policy_type: int = 0, enable_d2d: bool = True):
        """
        Args:
            shared_dict: Cross-process shared dict returned by Manager().dict()
            policy_type: Integer passed to PolicyFactory.generate_policy
            enable_d2d: Whether to enable D2D loading
        """
        self.shared_dict = shared_dict
        self.policy_type = policy_type
        self.enable_d2d = enable_d2d
        self.planner_q = planner_q
        self.block_update_q = block_update_q
        self.redundant_enable = redundant_enable

        # Create EplbWorker instance
        self.worker = EplbWorker(self.shared_dict, self.policy_type, self.enable_d2d, self.redundant_enable)


    def worker_process(self, planner_q, block_update_q):
        """
        Subprocess entry: bind to specified NPU, loop waiting for planner_q to wake up, call do_update, then notify main process update is complete.
        """
        while True:
            try:

                planner_q.get()

                packed_update_info = self.worker.do_update()

                while True:
                    if not block_update_q.empty():
                        continue
                    block_update_q.put(packed_update_info)
                    break

            except Exception as e:
                logger.warning(f"[EPLB subprocess Exiting due to error: {e}", exc_info=True)
                break

    def _launch_process(self):
        """
        Use spawn method to launch subprocess and return (planner_q, block_update_q, proc).
        """
        proc = Process(
            target=self.worker_process,
            args=(self.planner_q,self.block_update_q),
            daemon=True
        )

        proc.start()
        return  proc

