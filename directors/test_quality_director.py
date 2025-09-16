#!/usr/bin/env python3
"""
Test script for QualityDirector functionality

This script demonstrates the capabilities of the QualityDirector class
including code quality evaluation, code smell detection, and recommendations.

Usage:
    python3 test_quality_director.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from directors.quality_director import QualityDirector


def test_quality_director():
    """Test the QualityDirector with various code examples."""

    print("=" * 60)
    print("Testing QualityDirector - Code Quality Evaluation")
    print("=" * 60)

    # Initialize the director
    director = QualityDirector()

    print(f"Director: {director.name}")
    print(f"Expertise: {director.expertise}")
    print(f"Knowledge Base Size: {sum(len(items) for items in director.knowledge_base.values())} items")
    print()

    # Test Case 1: Poor quality code
    print("Test Case 1: Poor Quality Code")
    print("-" * 30)

    poor_code = '''
def calculate(a, b, c, d, e, f, g):  # Too many parameters
    # This function does too much and has poor structure
    result = 0
    if a > 10:
        if b > 5:
            if c > 3:
                if d > 2:  # Deep nesting
                    result = a * b * c * d + 100  # Magic number

    # Duplicate-like code blocks
    temp = result * 2
    temp = temp + 1
    final1 = temp * 2

    temp2 = result * 2
    temp2 = temp2 + 1
    final2 = temp2 * 2

    for i in range(50):  # Magic number
        if i % 2 == 0:
            result += i * 3  # Magic number

    return result

class massive_class:  # Wrong naming, no docs
    def method1(self): pass
    def method2(self): pass
    def method3(self): pass
    def method4(self): pass
    def method5(self): pass
    def method6(self): pass
    def method7(self): pass
    def method8(self): pass
    def method9(self): pass
    def method10(self): pass
    def method11(self): pass
    def method12(self): pass
    def method13(self): pass
    def method14(self): pass
    def method15(self): pass
    def method16(self): pass
    def method17(self): pass
    def method18(self): pass
    def method19(self): pass
    def method20(self): pass
    def method21(self): pass
    def method22(self): pass  # Too many methods
'''

    task1 = {
        'code': poor_code,
        'type': 'code_review',
        'description': 'Legacy code quality assessment',
        'language': 'python'
    }
    context1 = {
        'requirements': ['maintainability', 'readability'],
        'constraints': ['minimize breaking changes']
    }

    result1 = director.evaluate(task1, context1)

    print(f"Quality Score: {result1['quality_score']:.1f}/10")
    print(f"Confidence: {result1['confidence']:.2f}")
    print(f"Code Smells: {len(result1['code_smells'])}")
    print(f"Naming Issues: {len(result1['naming_issues'])}")
    print(f"Documentation Coverage: {result1['documentation_assessment']['coverage_percentage']:.1f}%")
    print(f"Refactoring Effort: {result1['estimated_effort']}")

    print("\nTop Code Smells:")
    for i, smell in enumerate(result1['code_smells'][:5], 1):
        print(f"  {i}. {smell['type']}: {smell['description']}")

    print("\nTop Recommendations:")
    for i, rec in enumerate(result1['recommendations'][:3], 1):
        print(f"  {i}. {rec}")

    print("\n" + "=" * 60)

    # Test Case 2: Good quality code
    print("Test Case 2: High Quality Code")
    print("-" * 30)

    good_code = '''
"""
A well-structured module for mathematical operations.

This module provides clean, documented functions for common calculations.
"""

from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

# Constants
DEFAULT_PRECISION = 2
MAX_INPUT_SIZE = 1000


class Calculator:
    """
    A calculator class that performs mathematical operations.

    This class provides methods for basic arithmetic operations
    with proper error handling and logging.
    """

    def __init__(self, precision: int = DEFAULT_PRECISION):
        """
        Initialize the calculator.

        Args:
            precision (int): Number of decimal places for results
        """
        self.precision = precision
        logger.info(f"Calculator initialized with precision {precision}")

    def add(self, first_number: float, second_number: float) -> float:
        """
        Add two numbers together.

        Args:
            first_number (float): The first number
            second_number (float): The second number

        Returns:
            float: The sum of the two numbers

        Raises:
            ValueError: If inputs are not valid numbers
        """
        try:
            result = first_number + second_number
            return round(result, self.precision)
        except (TypeError, ValueError) as error:
            logger.error(f"Addition failed: {error}")
            raise ValueError("Invalid input for addition") from error

    def multiply(self, numbers: List[float]) -> float:
        """
        Multiply a list of numbers.

        Args:
            numbers (List[float]): List of numbers to multiply

        Returns:
            float: The product of all numbers

        Raises:
            ValueError: If input list is empty or contains invalid numbers
        """
        if not numbers:
            raise ValueError("Cannot multiply empty list")

        if len(numbers) > MAX_INPUT_SIZE:
            raise ValueError(f"Input list too large (max {MAX_INPUT_SIZE})")

        result = 1.0
        for number in numbers:
            if not isinstance(number, (int, float)):
                raise ValueError(f"Invalid number type: {type(number)}")
            result *= number

        return round(result, self.precision)


def validate_input(value: float, minimum: Optional[float] = None) -> bool:
    """
    Validate numeric input against constraints.

    Args:
        value (float): The value to validate
        minimum (Optional[float]): Minimum allowed value

    Returns:
        bool: True if value is valid
    """
    if not isinstance(value, (int, float)):
        return False

    if minimum is not None and value < minimum:
        return False

    return True
'''

    task2 = {
        'code': good_code,
        'type': 'code_review',
        'description': 'Well-written calculator module',
        'language': 'python'
    }
    context2 = {
        'requirements': ['production-ready', 'well-documented'],
        'assumptions': ['Python 3.8+', 'Type hints required']
    }

    result2 = director.evaluate(task2, context2)

    print(f"Quality Score: {result2['quality_score']:.1f}/10")
    print(f"Confidence: {result2['confidence']:.2f}")
    print(f"Code Smells: {len(result2['code_smells'])}")
    print(f"Naming Issues: {len(result2['naming_issues'])}")
    print(f"Documentation Coverage: {result2['documentation_assessment']['coverage_percentage']:.1f}%")
    print(f"Refactoring Effort: {result2['estimated_effort']}")

    if result2['code_smells']:
        print("\nCode Smells:")
        for i, smell in enumerate(result2['code_smells'], 1):
            print(f"  {i}. {smell['type']}: {smell['description']}")
    else:
        print("\n✅ No significant code smells detected!")

    if result2['recommendations']:
        print("\nRecommendations:")
        for i, rec in enumerate(result2['recommendations'], 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✅ No major improvements needed!")

    print("\n" + "=" * 60)

    # Test specialized methods
    print("Testing Specialized Methods")
    print("-" * 30)

    test_code = '''
def badlyNamed(x, y):
    """Function with bad naming."""
    return x + y

def goodFunction():
    # No docstring
    pass

class BadClassName:
    def method(self):
        """Documented method."""
        if True:
            if True:
                if True:
                    return 42  # Magic number
'''

    print("Complexity Analysis:")
    complexity = director.analyze_complexity(test_code)
    print(f"  Average Complexity: {complexity['average_complexity']:.1f}")
    print(f"  Functions Analyzed: {len(complexity['functions'])}")

    print("\nNaming Convention Issues:")
    naming = director.check_naming_conventions(test_code)
    for issue in naming:
        print(f"  {issue['type']}: {issue['item']} (line {issue['line']})")

    print("\nDocumentation Assessment:")
    docs = director.assess_documentation(test_code)
    print(f"  Coverage: {docs['coverage_percentage']:.1f}%")
    print(f"  Quality Issues: {len(docs['quality_issues'])}")

    print("\nCode Smells:")
    smells = director.detect_code_smells(test_code)
    for smell in smells:
        print(f"  {smell['type']}: {smell['description']}")

    print("\n✅ QualityDirector testing completed successfully!")

    # Final summary
    print("\n" + "=" * 60)
    print("QualityDirector Summary")
    print("=" * 60)
    summary = director.get_performance_summary()
    print(f"Director: {summary['director_name']}")
    print(f"Version: {summary['version']}")
    print(f"Evaluations Performed: {summary['evaluations_performed']}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    print(f"Knowledge Base Categories: {len(summary['knowledge_base_size'])}")

    total_knowledge = sum(summary['knowledge_base_size'].values())
    print(f"Total Knowledge Items: {total_knowledge}")

    print("\nKnowledge Base Breakdown:")
    for category, count in summary['knowledge_base_size'].items():
        print(f"  {category}: {count} items")


if __name__ == "__main__":
    test_quality_director()