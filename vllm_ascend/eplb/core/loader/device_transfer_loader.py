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
import torch
import torch.distributed as dist
from enum import Enum

from vllm.logger import logger
from vllm_ascend.eplb.core.loader.abstract_loader import ExpertWeightLoader

class ExpertWeightUpdateState(Enum):
    WAITING = 0      # waiting for updated expert_map by EplbWorker
    READY = 1        # ready for d2d expert weights updating
    TRANSFERING = 2  # d2d finished and waiting for updating expert_map into model

class D2DExpertWeightLoader(ExpertWeightLoader):

    def __init__(self, eplb_adaptor):
        self.comm_op_list = None
        self.eplb_adaptor = eplb_adaptor

        self.updated_expert_map = None
        self.updated_log2phy_map = None
        self.layer_id = -1             # layer id to be updated
        self.state = ExpertWeightUpdateState.WAITING
        self.recv_expert_list = []
        self.mock_flag = True

    def generate_expert_d2d_transfer_task(self, expert_send_info, expert_recv_info,
        updated_expert_map, layer_id):
        # When current send/recv and weight.expert_map update tasks are not finished, cannot accept new d2d task
        if self.state != ExpertWeightUpdateState.WAITING:
            logger.error("current d2d weight update tasks are on-going, cannot accept new weight update task")
            return

        # If neither send nor receive task is needed for this layer on this rank, return
        if not (expert_send_info or expert_recv_info):
            return

        self.updated_expert_map = updated_expert_map

        self.layer_id = layer_id
        self.comm_op_list = []
        for send_info in expert_send_info:
            dst_rank, global_expert_id_to_send = send_info
            for src_tensor in self.eplb_adaptor.get_expert_tensor(layer_id, global_expert_id_to_send):
                self.comm_op_list.append(dist.P2POp(dist.isend, src_tensor, dst_rank))

        buffer_tensor_id = 0
        for recv_info in expert_recv_info:
            recv_rank, global_expert_id_to_recv = recv_info
            for buffer_tensor in self.eplb_adaptor.get_buffer_tensor(buffer_tensor_id):
                self.comm_op_list.append(dist.P2POp(dist.irecv, buffer_tensor, recv_rank))
            self.recv_expert_list.append((global_expert_id_to_recv, buffer_tensor_id))
            buffer_tensor_id += 1

        self.state = ExpertWeightUpdateState.READY

    def generate_log2phy_map(self,expert_map):
        self.updated_log2phy_map = self.eplb_adaptor.generate_log2phy_map(expert_map)

    def asyn_expert_weight_transfer(self, reqs):
        # Only when send/recv tasks are parsed into self.comm_op_list, d2d send/recv tasks can be luanched
        if self.state != ExpertWeightUpdateState.READY:
            return

        # set asynchronous stream for d2d expert weight transfer
        if self.comm_op_list:
            reqs = dist.batch_isend_irecv(self.comm_op_list)

        self.state = ExpertWeightUpdateState.TRANSFERING

    def update_expert_map_and_weight(self, reqs):
        # Only after send/recv tasks have been luanched, expert_map and weight can be updated
        if self.state != ExpertWeightUpdateState.TRANSFERING:
            return

        # Waiting for send/recv tasks finish
        for req in reqs:
            req.wait()

        if self.comm_op_list is not None:
            self.comm_op_list = None

        # update expert_map
        self.eplb_adaptor.do_update_expert_map(self.layer_id, self.updated_expert_map)

        #update log2phy_map
        self.eplb_adaptor.do_update_log2phy_map(self.layer_id, self.updated_log2phy_map)

        # update expert weight
        buffer_tensor_id = 0
        for recv_expert_info in self.recv_expert_list:
            global_expert_id_to_recv, buffer_tensor_id = recv_expert_info
            self.eplb_adaptor.do_update_expert_weight(self.layer_id, global_expert_id_to_recv, buffer_tensor_id)

        self.recv_expert_list = []
        self.updated_expert_map = None
        self.layer_id = -1
        self.state = ExpertWeightUpdateState.WAITING

    def generate_mock_update_info(self, rank_id):
        if rank_id == 0:
            expert_send_info = [(1, 0)]
            expert_recv_info = [(1, 64)]
            updated_expert_map_list = [-1] + [i for i in range(1, 64)] + [0] + [j for j in [-1] * 191]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        if rank_id == 1:
            expert_send_info = [(0, 64)]
            expert_recv_info = [(0, 0)]
            updated_expert_map_list = [0] + [k for k in [-1] * 63] + [i for i in range(1, 64)] + [j for j in [-1] * 129]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        if rank_id == 2:
            expert_send_info = [(3, 128)]
            expert_recv_info = [(3, 192)]
            updated_expert_map_list = [k for k in [-1] * 129] + [i for i in range(1, 64)] + [0] + [j for j in [-1] * 63]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        if rank_id == 3:
            expert_send_info = [(2, 192)]
            expert_recv_info = [(2, 128)]
            updated_expert_map_list = [k for k in [-1] * 128] + [0] + [k for k in [-1] * 64] + [i for i in range(1, 64)]
            updated_expert_map = torch.tensor(updated_expert_map_list)
            layer_id = 3

        self.mock_flag = False
        return (expert_send_info, expert_recv_info, updated_expert_map, layer_id)

    def load_impl(self, old_expert_table, new_expert_table):
        raise NotImplementedError

    