"""
Echo Brain Board of Directors System

This module provides the Board of Directors framework for Echo Brain,
enabling specialized domain experts to evaluate tasks, provide guidance,
and suggest improvements across different areas of expertise.

The Board of Directors system consists of:
- DirectorBase: Abstract base class for all directors
- DirectorRegistry: Central registry for managing all directors
- Specialized Directors: Domain-specific director implementations

Usage:
    from directors import DirectorBase, DirectorRegistry
    from directors import SecurityDirector, QualityDirector, PerformanceDirector, EthicsDirector, UXDirector

    # Create and register directors
    registry = DirectorRegistry()
    security_director = SecurityDirector()
    quality_director = QualityDirector()
    performance_director = PerformanceDirector()
    ethics_director = EthicsDirector()
    ux_director = UXDirector()

    registry.register_director(security_director)
    registry.register_director(quality_director)
    registry.register_director(performance_director)
    registry.register_director(ethics_director)
    registry.register_director(ux_director)

    # Evaluate a task
    task = {"type": "code_review", "description": "Review authentication system"}
    context = {"user": "developer", "priority": "high"}

    evaluations = registry.evaluate_task(task, context)

Author: Echo Brain Board of Directors System
Created: 2025-09-16
Version: 1.0.0
"""

from .base_director import DirectorBase

# Version information
__version__ = "1.0.0"
__author__ = "Echo Brain Board of Directors System"
__email__ = "echo@tower.local"

# Module-level exports
__all__ = [
    "DirectorBase"
]

# Import registry
from .director_registry import DirectorRegistry
__all__.append("DirectorRegistry")

# Import specialized directors
from .security_director import SecurityDirector
from .quality_director import QualityDirector
from .performance_director import PerformanceDirector
from .ethics_director import EthicsDirector
from .ux_director import UXDirector
__all__.extend(["SecurityDirector", "QualityDirector", "PerformanceDirector", "EthicsDirector", "UXDirector"])

# Module metadata
__title__ = "Echo Brain Directors"
__description__ = "Board of Directors framework for Echo Brain domain expertise system"
__url__ = "https://tower.local/api/echo"
__license__ = "Proprietary - Tower System"
__copyright__ = "2025 Tower System - Echo Brain"

def get_version():
    """Return the version string."""
    return __version__

def get_info():
    """Return module information."""
    return {
        "name": __title__,
        "version": __version__,
        "description": __description__,
        "author": __author__,
        "url": __url__,
        "license": __license__
    }