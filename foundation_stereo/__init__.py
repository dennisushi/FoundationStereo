# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
FoundationStereo: Zero-Shot Stereo Matching

A foundation model for stereo depth estimation designed to achieve strong zero-shot generalization.
"""

__version__ = "0.1.0"

# Re-export main classes and utilities for convenience
from foundation_stereo.foundation_stereo import FoundationStereo
from foundation_stereo.utils.utils import InputPadder

__all__ = [
    "FoundationStereo",
    "InputPadder",
]

