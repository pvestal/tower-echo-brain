#!/usr/bin/env python3
"""
Abstract interface for vision/image processing models in Echo Brain.
Provides standardized image analysis, processing, and generation capabilities.
Patrick Vestal - December 9, 2025
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np

from .ml_model_interface import MLModelInterface, ModelType, ModelPrediction


class VisionTask(Enum):
    """Types of vision tasks."""
    CLASSIFICATION = "classification"
    DETECTION = "detection"
    SEGMENTATION = "segmentation"
    FEATURE_EXTRACTION = "feature_extraction"
    QUALITY_ASSESSMENT = "quality_assessment"
    GENERATION = "generation"
    ENHANCEMENT = "enhancement"
    ANALYSIS = "analysis"


@dataclass
class ImageInput:
    """Standardized image input format."""
    image_data: Union[np.ndarray, str, bytes]  # numpy array, file path, or bytes
    format: str  # jpg, png, webp, etc.
    width: Optional[int] = None
    height: Optional[int] = None
    channels: Optional[int] = None
    metadata: Dict[str, Any] = None


@dataclass
class VisionResult:
    """Result from vision model processing."""
    predictions: Any
    confidence_scores: List[float]
    processing_time: float
    model_name: str
    task_type: VisionTask
    input_metadata: Dict[str, Any]
    output_metadata: Dict[str, Any]


@dataclass
class BoundingBox:
    """Bounding box for object detection."""
    x: float
    y: float
    width: float
    height: float
    confidence: float
    class_label: Optional[str] = None


@dataclass
class DetectionResult:
    """Object detection result."""
    bounding_boxes: List[BoundingBox]
    class_labels: List[str]
    confidence_scores: List[float]
    image_width: int
    image_height: int


class VisionInterface(MLModelInterface):
    """Abstract interface for vision/image processing models."""

    def __init__(self, model_name: str, task_type: VisionTask):
        """Initialize vision interface."""
        super().__init__(model_name, ModelType.VISION)
        self._task_type = task_type

    @property
    def task_type(self) -> VisionTask:
        """Get vision task type."""
        return self._task_type

    @abstractmethod
    async def process_image(
        self,
        image_input: ImageInput,
        context: Optional[Dict[str, Any]] = None
    ) -> VisionResult:
        """Process a single image.

        Args:
            image_input: ImageInput object with image data
            context: Optional context information

        Returns:
            VisionResult with processing results
        """
        pass

    @abstractmethod
    async def process_batch(
        self,
        images: List[ImageInput],
        batch_size: int = 8,
        context: Optional[Dict[str, Any]] = None
    ) -> List[VisionResult]:
        """Process batch of images.

        Args:
            images: List of ImageInput objects
            batch_size: Batch size for processing
            context: Optional context information

        Returns:
            List of VisionResult objects
        """
        pass

    @abstractmethod
    def validate_image_input(self, image_input: ImageInput) -> bool:
        """Validate image input format and size.

        Args:
            image_input: ImageInput to validate

        Returns:
            True if valid, False otherwise
        """
        pass

    @abstractmethod
    def get_supported_formats(self) -> List[str]:
        """Get supported image formats.

        Returns:
            List of supported format strings
        """
        pass

    @abstractmethod
    def get_input_requirements(self) -> Dict[str, Any]:
        """Get input requirements (resolution, channels, etc.).

        Returns:
            Dictionary with requirement specifications
        """
        pass


class ImageClassificationInterface(VisionInterface):
    """Interface for image classification models."""

    def __init__(self, model_name: str):
        """Initialize image classification interface."""
        super().__init__(model_name, VisionTask.CLASSIFICATION)

    @abstractmethod
    async def classify_image(
        self,
        image_input: ImageInput,
        top_k: int = 5,
        confidence_threshold: float = 0.1,
        context: Optional[Dict[str, Any]] = None
    ) -> VisionResult:
        """Classify image into categories.

        Args:
            image_input: ImageInput object
            top_k: Number of top predictions to return
            confidence_threshold: Minimum confidence threshold
            context: Optional context information

        Returns:
            VisionResult with classification predictions
        """
        pass

    @abstractmethod
    def get_class_labels(self) -> List[str]:
        """Get list of possible class labels.

        Returns:
            List of class label strings
        """
        pass

    @abstractmethod
    async def get_class_probabilities(
        self,
        image_input: ImageInput,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Get probabilities for all classes.

        Args:
            image_input: ImageInput object
            context: Optional context information

        Returns:
            Dictionary mapping class labels to probabilities
        """
        pass


class ObjectDetectionInterface(VisionInterface):
    """Interface for object detection models."""

    def __init__(self, model_name: str):
        """Initialize object detection interface."""
        super().__init__(model_name, VisionTask.DETECTION)

    @abstractmethod
    async def detect_objects(
        self,
        image_input: ImageInput,
        confidence_threshold: float = 0.5,
        nms_threshold: float = 0.4,
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """Detect objects in image.

        Args:
            image_input: ImageInput object
            confidence_threshold: Minimum confidence threshold
            nms_threshold: Non-maximum suppression threshold
            context: Optional context information

        Returns:
            DetectionResult with detected objects
        """
        pass

    @abstractmethod
    async def detect_specific_class(
        self,
        image_input: ImageInput,
        target_class: str,
        confidence_threshold: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """Detect objects of specific class.

        Args:
            image_input: ImageInput object
            target_class: Class to detect
            confidence_threshold: Minimum confidence threshold
            context: Optional context information

        Returns:
            DetectionResult with detected objects of target class
        """
        pass

    @abstractmethod
    def get_detectable_classes(self) -> List[str]:
        """Get list of detectable object classes.

        Returns:
            List of class names
        """
        pass


class ImageQualityInterface(VisionInterface):
    """Interface for image quality assessment models."""

    def __init__(self, model_name: str):
        """Initialize image quality interface."""
        super().__init__(model_name, VisionTask.QUALITY_ASSESSMENT)

    @abstractmethod
    async def assess_technical_quality(
        self,
        image_input: ImageInput,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Assess technical quality metrics.

        Args:
            image_input: ImageInput object
            context: Optional context information

        Returns:
            Dictionary with quality metrics (sharpness, noise, etc.)
        """
        pass

    @abstractmethod
    async def assess_artistic_quality(
        self,
        image_input: ImageInput,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, float]:
        """Assess artistic quality metrics.

        Args:
            image_input: ImageInput object
            context: Optional context information

        Returns:
            Dictionary with artistic metrics (composition, color harmony, etc.)
        """
        pass

    @abstractmethod
    async def calculate_overall_score(
        self,
        image_input: ImageInput,
        weights: Optional[Dict[str, float]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> float:
        """Calculate overall quality score.

        Args:
            image_input: ImageInput object
            weights: Optional weights for different quality aspects
            context: Optional context information

        Returns:
            Overall quality score (0.0 to 1.0)
        """
        pass

    @abstractmethod
    async def detect_artifacts(
        self,
        image_input: ImageInput,
        artifact_types: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Detect image artifacts.

        Args:
            image_input: ImageInput object
            artifact_types: Optional list of artifact types to detect
            context: Optional context information

        Returns:
            Dictionary with detected artifacts and severity
        """
        pass


class ImageGenerationInterface(VisionInterface):
    """Interface for image generation models."""

    def __init__(self, model_name: str):
        """Initialize image generation interface."""
        super().__init__(model_name, VisionTask.GENERATION)

    @abstractmethod
    async def generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        width: int = 512,
        height: int = 512,
        steps: int = 20,
        guidance_scale: float = 7.5,
        seed: Optional[int] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> ImageInput:
        """Generate image from text prompt.

        Args:
            prompt: Text description of desired image
            negative_prompt: Optional negative prompt
            width: Output image width
            height: Output image height
            steps: Number of generation steps
            guidance_scale: Guidance scale for generation
            seed: Optional random seed
            context: Optional context information

        Returns:
            ImageInput with generated image
        """
        pass

    @abstractmethod
    async def generate_batch(
        self,
        prompts: List[str],
        batch_size: int = 4,
        **generation_params
    ) -> List[ImageInput]:
        """Generate batch of images.

        Args:
            prompts: List of text prompts
            batch_size: Batch size for generation
            **generation_params: Additional generation parameters

        Returns:
            List of ImageInput objects with generated images
        """
        pass

    @abstractmethod
    async def image_to_image(
        self,
        source_image: ImageInput,
        prompt: str,
        strength: float = 0.7,
        context: Optional[Dict[str, Any]] = None
    ) -> ImageInput:
        """Transform existing image based on prompt.

        Args:
            source_image: Source ImageInput object
            prompt: Transformation prompt
            strength: Transformation strength (0.0 to 1.0)
            context: Optional context information

        Returns:
            ImageInput with transformed image
        """
        pass

    @abstractmethod
    def get_generation_parameters(self) -> Dict[str, Any]:
        """Get available generation parameters and their ranges.

        Returns:
            Dictionary with parameter specifications
        """
        pass


class ImageEnhancementInterface(VisionInterface):
    """Interface for image enhancement models."""

    def __init__(self, model_name: str):
        """Initialize image enhancement interface."""
        super().__init__(model_name, VisionTask.ENHANCEMENT)

    @abstractmethod
    async def upscale_image(
        self,
        image_input: ImageInput,
        scale_factor: float = 2.0,
        method: str = "esrgan",
        context: Optional[Dict[str, Any]] = None
    ) -> ImageInput:
        """Upscale image resolution.

        Args:
            image_input: Source ImageInput object
            scale_factor: Scaling factor (1.0 to 8.0)
            method: Upscaling method
            context: Optional context information

        Returns:
            ImageInput with upscaled image
        """
        pass

    @abstractmethod
    async def denoise_image(
        self,
        image_input: ImageInput,
        noise_level: str = "auto",
        context: Optional[Dict[str, Any]] = None
    ) -> ImageInput:
        """Remove noise from image.

        Args:
            image_input: Source ImageInput object
            noise_level: Noise level (low, medium, high, auto)
            context: Optional context information

        Returns:
            ImageInput with denoised image
        """
        pass

    @abstractmethod
    async def enhance_colors(
        self,
        image_input: ImageInput,
        enhancement_type: str = "auto",
        strength: float = 0.5,
        context: Optional[Dict[str, Any]] = None
    ) -> ImageInput:
        """Enhance image colors.

        Args:
            image_input: Source ImageInput object
            enhancement_type: Enhancement type (auto, vibrant, natural)
            strength: Enhancement strength (0.0 to 1.0)
            context: Optional context information

        Returns:
            ImageInput with color-enhanced image
        """
        pass

    @abstractmethod
    def get_enhancement_options(self) -> Dict[str, List[str]]:
        """Get available enhancement options.

        Returns:
            Dictionary with enhancement types and options
        """
        pass