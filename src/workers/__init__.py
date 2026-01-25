"""
Echo Brain Workers
Task workers for autonomous LORA training and other operations
"""

from src.workers.lora_generation_worker import LoraGenerationWorker, handle_lora_generation_task
from src.workers.lora_tagging_worker import LoraTaggingWorker, handle_lora_tagging_task
from src.workers.lora_training_worker import LoraTrainingWorker, handle_lora_training_task
from src.workers.anime_generation_worker import AnimeGenerationWorker, anime_worker

__all__ = [
    'LoraGenerationWorker',
    'LoraTaggingWorker',
    'LoraTrainingWorker',
    'AnimeGenerationWorker',
    'anime_worker',
    'handle_lora_generation_task',
    'handle_lora_tagging_task',
    'handle_lora_training_task',
]
