# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SM120 implementation variant for ``FLASHINFER_MLA_SPARSE_SM120``."""

from typing import TYPE_CHECKING, cast

import torch

from vllm.v1.attention.backend import (
    AttentionLayer,
    AttentionType,
    MLAAttentionImpl,
)
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseMetadata,
    _get_workspace_buffer,
)
from vllm.v1.attention.backends.mla.sparse_utils import (
    triton_convert_req_index_to_global_index,
)

if TYPE_CHECKING:
    from vllm.model_executor.models.deepseek_v2 import Indexer


def _kv_scale_format_for_model(model_type: str | None) -> str:
    if model_type is not None and model_type.startswith("glm"):
        return "arbitrary_fp32"
    return "pow2_fp32"


class FlashInferMLASparseSM120Impl(MLAAttentionImpl[FlashInferMLASparseMetadata]):
    """SM120 FlashInfer sparse-MLA implementation."""

    is_sparse = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        indexer: "Indexer | None" = None,
        **mla_args,
    ) -> None:
        if any([alibi_slopes, sliding_window, logits_soft_cap]):
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 does not support alibi_slopes / "
                "sliding_window / logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 only supports decoder self-attention"
            )

        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise NotImplementedError(
                "FLASHINFER_MLA_SPARSE_SM120 requires the packed fp8_ds_mla "
                f"KV cache layout; got kv_cache_dtype={kv_cache_dtype!r}."
            )

        self.kv_lora_rank: int = mla_args["kv_lora_rank"]
        self.qk_nope_head_dim: int = mla_args["qk_nope_head_dim"]
        self.qk_rope_head_dim: int = mla_args["qk_rope_head_dim"]
        from vllm.config import get_current_vllm_config

        vllm_config = get_current_vllm_config()
        model_type = None
        if vllm_config.model_config is not None:
            model_type = getattr(
                vllm_config.model_config.hf_text_config, "model_type", None
            )
        self.kv_scale_format = _kv_scale_format_for_model(model_type)

        # Native selected-token budget (GLM index_topk). The physical top-k page
        # table is kpool-widened past this and rounded up to a multiple of 128
        # (2048 -> 2176), but the SM120 GLM_NSA decode kernels only instantiate
        # the (num_heads, topk) configs at topk == index_topk (2048). Pass the
        # logical budget as the page-table capacity and slice the widened table
        # to match; the triton index compaction packs all valid slots into the
        # first ``seq_lens`` (<= index_topk) columns, so nothing valid is lost.
        self.index_topk = 2048
        if vllm_config.model_config is not None:
            self.index_topk = int(
                getattr(vllm_config.model_config.hf_text_config, "index_topk", 2048)
            )

        # Skip-topk layers are built with indexer=None and get the shared
        # buffer via mla_args instead (cf. FLASHMLA_SPARSE).
        self.topk_indices_buffer: torch.Tensor | None = (
            indexer.topk_indices_buffer
            if indexer is not None
            else mla_args.get("topk_indices_buffer")
        )
        from vllm.utils.flashinfer import has_flashinfer_sparse_mla_sm120

        if not has_flashinfer_sparse_mla_sm120():
            raise RuntimeError(
                "FLASHINFER_MLA_SPARSE_SM120 requires FlashInfer's "
                "sparse MLA decode API."
            )
        assert self.topk_indices_buffer is not None

        self.supports_quant_query_input = False
        self._workspace_buffer: torch.Tensor | None = None

    def forward_mqa(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: FlashInferMLASparseMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if isinstance(q, tuple):
            q = torch.cat(q, dim=-1)

        # GLM-5.3-Flash NoPE-MLA overlay for the packed SM120 sparse kernel.
        # Two independent bugs in the shipped SM120 forward_mqa are fixed here:
        #
        #  1) Zero rope tail. The model is NoPE (qk_rope_head_dim==0) so the
        #     absorbed query arrives as [B, N, kv_lora_rank(512)] with no rope
        #     tail. The packed SM120 kernel HARD-requires kv_lora_rank=512 +
        #     qk_rope_head_dim=64 + query head dim 576 (unlike the SM100 no-rope
        #     kernel, it has no native NoPE path). Pad the query with a zero 64
        #     tail and pass qk_rope_head_dim=64. The cache rope slot is likewise
        #     zero-filled on write (see do_kv_cache_update), so zero*zero == 0
        #     contribution to QK^T: bit-for-bit equivalent to NoPE.
        #
        #  2) Sparse page-table width. GLM's kpool indexer widens the physical
        #     top-k table past index_topk and rounds up to a multiple of 128
        #     (2048 -> 2176). The shipped code hard-coded sparse_mla_top_k=2048
        #     (attn_metadata.topk_tokens), mismatching the 2176-wide page table.
        #     Use the ACTUAL buffer width as the capacity and pass the compacted
        #     per-token valid counts (seq_lens) so the -1 padding slots past each
        #     token's valid length are never attended to.
        if self.qk_rope_head_dim == 0 and q.shape[-1] == self.kv_lora_rank:
            q = torch.nn.functional.pad(q, (0, 64))
        decode_qk_rope_head_dim = 64 if self.qk_rope_head_dim == 0 else (
            self.qk_rope_head_dim
        )

        num_actual_toks = q.shape[0]

        assert self.topk_indices_buffer is not None
        topk_indices = self.topk_indices_buffer[:num_actual_toks]

        topk_indices_physical, seq_lens = triton_convert_req_index_to_global_index(
            attn_metadata.req_id_per_token[:num_actual_toks],
            attn_metadata.block_table,
            topk_indices,
            BLOCK_SIZE=attn_metadata.block_size,
            NUM_TOPK_TOKENS=topk_indices.shape[1],
            return_valid_counts=True,
        )

        # Zero-length rows (indexer selected nothing) crash the kernel: point them
        # at a valid dummy slot with length 1 and zero their output afterwards.
        empty_rows = seq_lens == 0
        topk_indices_physical[:, 0] = topk_indices_physical[:, 0].masked_fill(
            empty_rows, 0
        )
        seq_lens = seq_lens.clamp(min=1)

        #  3) Logical top-k capacity. The kpool-widened physical page table is
        #     2176 wide, but the SM120 GLM_NSA decode kernels only instantiate
        #     (num_heads, topk) configs at topk == index_topk (2048); passing the
        #     widened 2176 fails the config lookup ("Unsupported sparse-MLA
        #     configuration ... topk=2176"). Slice the page table to index_topk
        #     and pass that as both the capacity and max_seq_len. The triton
        #     compaction already packs valid slots into the first ``seq_lens``
        #     (<= index_topk) columns, so the sliced-off tail is pure padding.
        sparse_topk_capacity = self.index_topk
        if topk_indices_physical.shape[1] > sparse_topk_capacity:
            # The kernel requires contiguous indices; a column slice of the
            # 2176-wide buffer is strided, so materialize it.
            topk_indices_physical = topk_indices_physical[
                :, :sparse_topk_capacity
            ].contiguous()

        output = q.new_empty(
            (num_actual_toks, self.num_heads, self.kv_lora_rank),
            dtype=q.dtype,
        )

        if self._workspace_buffer is None:
            self._workspace_buffer = _get_workspace_buffer(q.device)

        from vllm.utils.flashinfer import (
            flashinfer_trtllm_batch_decode_with_kv_cache_mla,
        )

        flashinfer_trtllm_batch_decode_with_kv_cache_mla(
            query=q.unsqueeze(1),
            kv_cache=kv_c_and_k_pe_cache.view(torch.uint8).unsqueeze(1),
            workspace_buffer=self._workspace_buffer,
            qk_nope_head_dim=self.qk_nope_head_dim,
            kv_lora_rank=self.kv_lora_rank,
            qk_rope_head_dim=decode_qk_rope_head_dim,
            block_tables=topk_indices_physical.unsqueeze(1),
            seq_lens=seq_lens,
            max_seq_len=sparse_topk_capacity,
            out=output.unsqueeze(1),
            bmm1_scale=self.scale,
            bmm2_scale=1.0,
            sparse_mla_top_k=sparse_topk_capacity,
            kv_scale_format=self.kv_scale_format,
        )

        output.masked_fill_(empty_rows.view(-1, 1, 1), 0.0)
        return output, None
