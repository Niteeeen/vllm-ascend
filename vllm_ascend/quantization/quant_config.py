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
#
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional

import torch
from vllm.config import get_current_vllm_config
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.model_executor.layers.fused_moe import (FusedMoE, FusedMoEMethodBase,
                                                  FusedMoeWeightScaleSupported)
from vllm.model_executor.layers.linear import (LinearBase, LinearMethodBase,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization import \
    register_quantization_config
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod
from vllm.model_executor.layers.vocab_parallel_embedding import (
    UnquantizedEmbeddingMethod, VocabParallelEmbedding)
from vllm.model_executor.models.utils import WeightsMapper
from vllm.model_executor.parameter import PerTensorScaleParameter
from vllm.model_executor.utils import set_weight_attrs

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import (get_flashcomm2_otp_group,
                                                    get_mlp_tp_group,
                                                    get_otp_group)
from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod
from vllm_ascend.ops.linear import AscendUnquantizedLinearMethod
from vllm_ascend.utils import (ASCEND_QUANTIZATION_METHOD, flashcomm2_enable,
                               mlp_tp_enable, oproj_tp_enable)

from .utils import get_quant_method, is_mx_quant_type


@register_quantization_config(ASCEND_QUANTIZATION_METHOD)
class AscendQuantConfig(QuantizationConfig):
    """Config class for Ascend

    This class is a general class that parse quantization configs
    that are supported on ascend hardware.
    """

    def __init__(self, quant_config: Dict[str, Any]):
        super().__init__()
        self.quant_description = quant_config
        # TODO(whx): remove this adaptation after adding "shared_head"
        # to prefix of DeepSeekShareHead in vLLM.
        extra_quant_dict = {}
        for k in self.quant_description.keys():
            if "shared_head" in k:
                new_k = k.replace(".shared_head.", ".")
                extra_quant_dict[new_k] = self.quant_description[k]
            if "weight_packed" in k:
                new_k = k.replace("weight_packed", "weight")
                extra_quant_dict[new_k] = self.quant_description[k]
        self.quant_description.update(extra_quant_dict)
        
        # 标记：量化配置映射是否已执行
        self._quant_mapping_done = False
        
        # 尝试立即执行映射（如果 vllm_config 可用，否则延迟到第一次使用）
        # 这与 deepseek_v2.py 中的权重数据映射逻辑保持一致
        self._map_quant_description_for_dense_to_moe()

    def _map_quant_description_for_dense_to_moe(self):
        """
        当 first_k_dense_replace=0 时，将第一个 MoE 层的量化配置复制到前面的 Dense 层。
        
        这个方法与 deepseek_v2.py 中的权重映射逻辑配套使用：
        - deepseek_v2.py 负责复制权重数据
        - 本方法负责复制量化元数据
        
        适用场景：
        - 原始模型: first_k_dense_replace=1 (Layer 0 Dense, Layer 1+ MoE)
        - 原始模型: first_k_dense_replace=3 (Layer 0-2 Dense, Layer 3+ MoE)
        - 修改后: first_k_dense_replace=0 (所有层都是 MoE)
        
        支持延迟执行：如果 vllm_config 在 __init__ 时未完全初始化，
        会在第一次使用量化配置时自动重试。
        """
        # 如果已经执行过，直接返回
        if self._quant_mapping_done:
            return
        
        try:
            vllm_config = get_current_vllm_config()
            
            # 检查 vllm_config 是否完全初始化
            if not vllm_config or not vllm_config.model_config:
                # 未完全初始化，延迟执行（不标记为已完成，允许后续重试）
                return
            
            config = vllm_config.model_config.hf_config
            
            # 只在 first_k_dense_replace=0 时执行映射
            if not hasattr(config, 'first_k_dense_replace') or config.first_k_dense_replace != 0:
                self._quant_mapping_done = True  # 标记为已完成（不需要映射）
                return
            
            # 1. 找到第一个包含 MoE 量化配置的层索引
            import re
            first_moe_layer = None
            
            for key in self.quant_description.keys():
                # 查找包含 experts 或 shared_experts 的层
                if "model.layers." in key and (".mlp.experts." in key or ".mlp.shared_experts." in key):
                    # 提取层索引
                    match = re.search(r"model\.layers\.(\d+)\.", key)
                    if match:
                        layer_idx = int(match.group(1))
                        if first_moe_layer is None or layer_idx < first_moe_layer:
                            first_moe_layer = layer_idx
            
            # 2. 如果没有找到 MoE 层，或者第一个 MoE 层就是 Layer 0，则无需映射
            if first_moe_layer is None or first_moe_layer == 0:
                self._quant_mapping_done = True
                return
            
            # 3. 复制第一个 MoE 层的所有 MoE 相关配置到所有前面的层
            # 需要复制的配置包括：
            # - model.layers.{N}.mlp.experts.*
            # - model.layers.{N}.mlp.shared_experts.*
            # - model.layers.{N}.mlp.gate.weight
            # - model.layers.{N}.mlp.gate.e_score_correction_bias
            new_entries = {}
            for key, value in self.quant_description.items():
                # 检查是否是第一个 MoE 层的 MoE 相关配置
                is_moe_config = (
                    f"model.layers.{first_moe_layer}.mlp.experts." in key or
                    f"model.layers.{first_moe_layer}.mlp.shared_experts." in key or
                    f"model.layers.{first_moe_layer}.mlp.gate." in key
                )
                
                if is_moe_config:
                    # 复制到所有前面的层
                    for target_layer in range(first_moe_layer):
                        new_key = key.replace(
                            f"model.layers.{first_moe_layer}.",
                            f"model.layers.{target_layer}."
                        )
                        new_entries[new_key] = value
            
            # 4. 更新量化配置字典
            if new_entries:
                self.quant_description.update(new_entries)
                print(f"[AscendQuantConfig] Mapped {len(new_entries)} quantization config entries from Layer {first_moe_layer} "
                      f"to Layers 0-{first_moe_layer-1} for first_k_dense_replace=0 compatibility")
            else:
                print(f"[AscendQuantConfig] Warning: No MoE quantization config found to map for first_k_dense_replace=0")
            
            # 标记为已完成
            self._quant_mapping_done = True
        
        except Exception as e:
            # 如果获取配置失败，不标记为已完成，允许后续重试
            import logging
            logger = logging.getLogger(__name__)
            logger.debug(f"Defer quant_description mapping: {e}")


    def __repr__(self) -> str:
        return "AscendQuantConfig:\n" + super().__repr__()

    @classmethod
    def get_name(cls) -> str:
        return ASCEND_QUANTIZATION_METHOD

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.int8, torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        raise NotImplementedError(
            "Ascend hardware dose not support \"get_min_capability\" feature.")

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["quant_model_description.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "AscendQuantConfig":
        return cls(config)

    @classmethod
    def override_quantization_method(cls, hf_quant_cfg,
                                     user_quant) -> Optional[str]:
        if hf_quant_cfg is not None:
            quant_method = hf_quant_cfg.get("quant_method", None)
            if not quant_method and torch.npu.is_available():
                return ASCEND_QUANTIZATION_METHOD
        return None

    def quant_prefix_mapper(self, model_type: str, prefix: str) -> str:
        # TODO (Levi-JQ): will be removed when QuantizationConfig.apply_vllm_mapper is implemented
        prefix_mapping = QUANT_MODEL_PREFIX_MAPPINGS.get(model_type)
        if prefix_mapping:
            hf_to_vllm_mapper = WeightsMapper(
                orig_to_new_prefix=prefix_mapping)
            return hf_to_vllm_mapper._map_name(prefix)
        return prefix

    def get_quant_method(self, layer: torch.nn.Module,
                         prefix: str) -> Optional["QuantizeMethodBase"]:
        vllm_config = get_current_vllm_config()
        model_type = vllm_config.model_config.hf_text_config.model_type

        if model_type in ["minimax", "minimax_m2"]:
            prefix = prefix.replace("mlp", "block_sparse_moe")

            #To adapt to minimax, modify the prefix of the model layer name
            parts = prefix.split('.')
            if "experts" in parts and len(parts) > 2:
                exp_idx = parts.index("experts")
                if exp_idx + 1 < len(parts) and parts[exp_idx + 1].isdigit():
                    parts = parts[:exp_idx + 1]
                    prefix = ".".join(parts)

        if model_type in packed_modules_model_mapping:
            self.packed_modules_mapping = packed_modules_model_mapping[
                model_type]
        prefix = self.quant_prefix_mapper(model_type, prefix)
        from vllm.attention.layer import Attention
        if prefix.startswith("language_model"):
            prefix = prefix.split('.', 1)[-1]
        if isinstance(layer, LinearBase):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return AscendUnquantizedLinearMethod()
            return AscendLinearMethod(self, prefix,
                                      self.packed_modules_mapping, layer)
        elif isinstance(layer, Attention) and \
            'fa_quant_type' in self.quant_description.keys() and \
            self.quant_description['fa_quant_type'] is not None:
            return AscendKVCacheMethod(self, prefix)
        elif isinstance(layer, FusedMoE):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return AscendUnquantizedFusedMoEMethod(layer.moe_config)
            return AscendFusedMoEMethod(self, prefix,
                                        self.packed_modules_mapping, layer)
        elif isinstance(layer, VocabParallelEmbedding):
            if self.is_layer_skipped_ascend(prefix,
                                            self.packed_modules_mapping):
                return UnquantizedEmbeddingMethod()
            return AscendEmbeddingMethod(self, prefix,
                                         self.packed_modules_mapping, layer)
        return None

    def is_layer_skipped_ascend(
        self,
        prefix: str,
        fused_mapping: Mapping[str, List[str]] = MappingProxyType({})):
        # 在使用前确保映射已执行（支持延迟执行）
        if not self._quant_mapping_done:
            self._map_quant_description_for_dense_to_moe()
        
        # adapted from vllm.model_executor.layers.quantization.utils.quant_utils.is_layer_skipped
        proj_name = prefix.split(".")[-1]
        if proj_name in fused_mapping:
            shard_prefixes = [
                prefix.replace(proj_name, shard_proj_name)
                for shard_proj_name in fused_mapping[proj_name]
            ]

            is_skipped = None
            for shard_prefix in shard_prefixes:
                is_shard_skipped = self.quant_description[shard_prefix +
                                                          '.weight'] == "FLOAT"

                if is_skipped is None:
                    is_skipped = is_shard_skipped
                elif is_shard_skipped != is_skipped:
                    raise ValueError(
                        f"Detected some but not all shards of {prefix} "
                        "are quantized. All shards of fused layers "
                        "to have the same precision.")
        else:
            # NOTE: In GLM4.6, the MTP draft model shares the same LM head weigthts
            # with the main model. Therefore, before `load_weights()` runs, some parameter
            # names may not include the expected prefix and may appear only with the
            # ".head" suffix. This can trigger a load-time error, so here we replace the
            # key with "lm_head.weight".
            key = prefix + '.weight'
            if key not in self.quant_description and ".head" in prefix:
                key = 'lm_head.weight'
            is_skipped = self.quant_description[key] == "FLOAT"

        assert is_skipped is not None
        return is_skipped

    def get_scaled_act_names(self) -> List[str]:
        return []


# key: model_type
# value: orig_to_new_prefix
QUANT_MODEL_PREFIX_MAPPINGS = {
    "qwen3_vl_moe_text": {
        "visual.": "model.visual.",
        "language_model.lm_head.": "lm_head.",
        "language_model.model.": "model.language_model.",
    },
    "qwen3_vl_text": {
        "visual.": "model.visual.",
        "language_model.lm_head.": "lm_head.",
        "language_model.model.": "model.language_model.",
    },
}

packed_modules_model_mapping = {
    "qwen3_moe": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    },
    "deepseek_v2": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    },
    "deepseek_v3": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    },
    "pangu_ultra_moe": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    },
    "kimi_k2": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    },
    "deepseek_v32": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    },
    # NOTE 1.The quantized MTP layer of deepseek on the NPU is not quantized;
    # NOTE 2.The description file generated by the current msmodelslim tool does not have
    # MTP layer info. Please manually add it and set the value to FLOAT.
    "deepseek_mtp": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    },
    "pangu_ultra_moe_mtp": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    },
    "qwen3_next": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj": ["in_proj_qkvz", "in_proj_ba"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    },
    "qwen2_5_vl": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    },
    "qwen3_vl_moe_text": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
    },
    "glm4_moe": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"]
    },
    "longcat_flash": {
        "gate_up_proj": ["gate_proj", "up_proj"],
        "experts":
        ["experts.0.gate_proj", "experts.0.up_proj", "experts.0.down_proj"],
        "fused_qkv_a_proj": ["q_a_proj", "kv_a_proj_with_mqa"]
    },
    "minimax_m2": {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "experts": ["experts.0.w1", "experts.0.w2", "experts.0.w3"]
    }
}


class AscendLinearMethod(LinearMethodBase):
    """Linear method for Ascend quantization.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self,
                 quant_config: AscendQuantConfig,
                 prefix: str,
                 packed_modules_mapping: Dict[str, Any] | None,
                 layer: torch.nn.Module = None) -> None:
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix,
                                             "linear",
                                             packed_modules_mapping,
                                             layer=layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: List[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        output_size_per_partition = sum(output_partition_sizes)
        weight_loader = extra_weight_attrs.get("weight_loader")

        weight_dict = self.quant_method.get_weight(input_size_per_partition,
                                                   output_size_per_partition,
                                                   params_dtype)

        # Extract packing information (if present)
        packed_dim = weight_dict.pop("_packed_dim", None)
        packed_factor = weight_dict.pop("_packed_factor", None)

        for weight_name, weight_param in weight_dict.items():
            param = torch.nn.Parameter(weight_param, requires_grad=False)
            set_weight_attrs(param, {"input_dim": 1, "output_dim": 0})

            # Set packing attributes if the weight is packed
            if packed_dim is not None and packed_factor is not None:
                set_weight_attrs(param, {
                    "packed_dim": packed_dim,
                    "packed_factor": packed_factor
                })

            layer.register_parameter(weight_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        pertensor_dict = self.quant_method.get_pertensor_param(params_dtype)
        for pertensor_name, pertensor_param in pertensor_dict.items():
            param = PerTensorScaleParameter(data=pertensor_param,
                                            weight_loader=weight_loader)
            # disable warning
            param.ignore_warning = True
            layer.register_parameter(pertensor_name, param)
            param.weight_loader = extra_weight_attrs.get("weight_loader")

        perchannel_dict = self.quant_method.get_perchannel_param(
            output_size_per_partition, params_dtype)
        for perchannel_name, perchannel_param in perchannel_dict.items():
            param = torch.nn.Parameter(perchannel_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(perchannel_name, param)
            set_weight_attrs(param, extra_weight_attrs)

        # NOTE: In w4a8 quantization implementation,
        # for down_proj and o_proj scale_bias shape is [output_size, 16],
        # others are [output_size, 1]
        layer_type = "row" if isinstance(layer,
                                         RowParallelLinear) else "others"

        pergroup_dict = self.quant_method.get_pergroup_param(
            input_size_per_partition,
            output_size_per_partition,
            params_dtype,
            layer_type=layer_type)
        for pergroup_name, pergroup_param in pergroup_dict.items():
            param = torch.nn.Parameter(pergroup_param, requires_grad=False)
            set_weight_attrs(param, {"output_dim": 0})
            layer.register_parameter(pergroup_name, param)
            set_weight_attrs(param, extra_weight_attrs)
            if "weight_scale_second" in pergroup_name or "weight_offset_second" in pergroup_name \
                or is_mx_quant_type(self.quant_method):
                setattr(param, "input_dim", 1)
                param.input_dim = 1

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if isinstance(layer, RowParallelLinear):
            if layer.prefix.find("o_proj") != -1 and oproj_tp_enable():
                tp_rank = get_otp_group().rank_in_group
            elif layer.prefix.find("down_proj") != -1 and mlp_tp_enable():
                tp_rank = get_mlp_tp_group().rank_in_group
            elif (layer.prefix.find("o_proj") != -1 or
                  layer.prefix.find("out_proj") != -1) and flashcomm2_enable():
                if get_ascend_config(
                ).flashcomm2_oproj_tensor_parallel_size == 1:
                    tp_rank = 0
                else:
                    tp_rank = get_flashcomm2_otp_group().rank_in_group
            else:
                tp_rank = get_tensor_model_parallel_rank()
        else:
            tp_rank = 0
        return self.quant_method.apply(layer, x, bias, tp_rank)


class AscendKVCacheMethod(BaseKVCacheMethod):
    """KVCache method for Ascend quantization.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str) -> None:
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix, "attention")

    def create_weights(self, layer: torch.nn.Module) -> None:
        # Different from linear method, there are no weight processing/slicing
        # steps for attention in vllm. So the whole process of create weights
        # is hidden into the specific quant method.
        self.quant_method.create_weights(layer)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def apply(self, layer: torch.nn.Module, query: torch.Tensor,
              key: torch.Tensor, value: torch.Tensor, kv_cache, attn_metadata,
              attn_type, scale, output) -> torch.Tensor:
        return self.quant_method.apply(layer, query, key, value, kv_cache,
                                       attn_metadata, attn_type, scale, output)


class AscendFusedMoEMethod(FusedMoEMethodBase):
    """FusedMoE method for Ascend quantization.

    Args:
        quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str,
                                              Any], layer: torch.nn.Module):
        super().__init__(layer.moe_config)
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix,
                                             "moe",
                                             packed_modules_mapping,
                                             layer=layer)

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        weight_param = self.quant_method.get_weight(
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in weight_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)

        extra_weight_attrs.update(
            {"quant_method": FusedMoeWeightScaleSupported.CHANNEL.value})
        per_group_param = [
            "weight_scale_second", "weight_offset_second", "scale_bias"
        ] + ["weight_scale", "weight_offset"] if hasattr(
            self.quant_method,
            "group_size") and self.quant_method.group_size > 0 else []
        dynamic_quant_param = self.quant_method.get_dynamic_quant_param(
            num_experts, intermediate_size_per_partition, hidden_size,
            params_dtype)
        for param_key, param_value in dynamic_quant_param.items():
            param = torch.nn.Parameter(param_value, requires_grad=False)
            layer.register_parameter(param_key, param)
            set_weight_attrs(param, extra_weight_attrs)
            if any(fields in param_key for fields in per_group_param):
                setattr(param, "quant_method",
                        FusedMoeWeightScaleSupported.GROUP.value)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        routed_scaling_factor: float = 1.0,
        e_score_correction_bias: Optional[torch.Tensor] = None,
        is_prefill: bool = True,
        enable_force_load_balance: bool = False,
        log2phy: torch.Tensor = None,
        global_redundant_expert_num=0,
        **kwargs,
    ) -> torch.Tensor:
        return self.quant_method.apply(
            layer, x, router_logits, top_k, renormalize, use_grouped_topk,
            global_num_experts, expert_map, topk_group, num_expert_group,
            custom_routing_function, scoring_func, routed_scaling_factor,
            e_score_correction_bias, is_prefill, enable_force_load_balance,
            log2phy, global_redundant_expert_num, **kwargs)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if hasattr(self.quant_method, "process_weights_after_loading"):
            self.quant_method.process_weights_after_loading(layer)

    def get_fused_moe_quant_config(self, layer: torch.nn.Module):
        # TODO: implement this function
        pass

    @property
    def supports_eplb(self):
        supports_eplb = getattr(self.quant_method, "supports_eplb", False)
        return supports_eplb


class AscendEmbeddingMethod(AscendLinearMethod):
    """Embedding method for Ascend quantization.
    
      Args:
          quant_config: The Ascend quantization config.
    """

    def __init__(self, quant_config: AscendQuantConfig, prefix: str,
                 packed_modules_mapping: Dict[str, Any],
                 layer: torch.nn.Module) -> None:
        self.quant_method = get_quant_method(quant_config.quant_description,
                                             prefix,
                                             "linear",
                                             packed_modules_mapping,
                                             layer=layer)
