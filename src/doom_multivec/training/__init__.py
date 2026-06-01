"""DOOM MultiVec training data pipeline."""

from .action_mapping import BASE_ACTIONS, GAMANGEN_ACTIONS, action_id_to_scores
from .teacher import compute_teacher_scores
from .dataset import DoomKDDatasetBuilder, ACTION_QUERY_TEXTS
from .dpo_dataset import (
    DPOPreferenceSample,
    DPOPreferenceDataset,
    DPODataBuilder,
)
from .dpo_trainer import DPOTrainer, create_reference_policy

__all__ = [
    'BASE_ACTIONS',
    'GAMANGEN_ACTIONS',
    'action_id_to_scores',
    'compute_teacher_scores',
    'DoomKDDatasetBuilder',
    'ACTION_QUERY_TEXTS',
    'DPOPreferenceSample',
    'DPOPreferenceDataset',
    'DPODataBuilder',
    'DPOTrainer',
    'create_reference_policy',
]
