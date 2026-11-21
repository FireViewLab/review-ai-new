"""Isolated P_text v3 training and evaluation pipeline.

Nothing in this package is imported by the production runtime.  The package
exists so v3 experiments can evolve without changing the current P_text model,
API, or score contract.
"""

from .data_contract import LABEL_MAPPING, SPLITS, ContractError, FixedDataset

__all__ = ["LABEL_MAPPING", "SPLITS", "ContractError", "FixedDataset"]
