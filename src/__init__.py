"""
EEP 595 Clock Skew-Based Intrusion Detection System

This package implements a clock skew-based IDS for CAN bus networks,
featuring both state-of-the-art and NTP-based detection approaches.

Author: EEP 595 Project
"""

from .ids import IDS
from .simulation import CANSimulation

__version__ = "1.0.0"
__all__ = ["IDS", "CANSimulation"]
