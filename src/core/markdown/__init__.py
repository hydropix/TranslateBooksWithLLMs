"""Markdown support: structure-preserving segmentation and chunk codec."""

from .segmenter import Segment, segment_markdown, rebuild, segments_of
from .chunks import ChunkPlan, plan_chunk, decode_chunk, group_chunks

__all__ = [
    "Segment", "segment_markdown", "rebuild", "segments_of",
    "ChunkPlan", "plan_chunk", "decode_chunk", "group_chunks",
]
