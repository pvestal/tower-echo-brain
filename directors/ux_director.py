"""
UX Director for Echo Brain Board of Directors System

This module provides comprehensive user experience evaluation capabilities including
accessibility compliance, usability assessment, responsive design evaluation,
user flow analysis, and error handling UX evaluation.

Author: Echo Brain Board of Directors System
Created: 2025-09-16
Version: 1.0.0
"""

import logging
import re
import json
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from .base_director import DirectorBase

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class UXDirector(DirectorBase):
    """
    UX Director specializing in user experience and interface design evaluation.

    This director provides comprehensive UX analysis including:
    - Accessibility compliance (WCAG 2.1 AA/AAA)
    - Usability assessment and heuristic evaluation
    - Responsive design evaluation
    - User flow and journey analysis
    - Error handling and feedback UX
    - Information architecture assessment
    - Interaction design evaluation
    - Mobile-first design compliance
    """

    def __init__(self):
        """Initialize the UX Director with user experience expertise."""
        super().__init__(
            name="UXDirector",
            expertise="User Experience Design, Accessibility Compliance, Usability Assessment, Responsive Design",
            version="1.0.0"
        )

        # Initialize UX-specific tracking
        self.accessibility_patterns = self._load_accessibility_patterns()
        self.ux_weights = {
            "critical": 1.0,    # Major UX violations
            "high": 0.8,        # Significant UX issues
            "medium": 0.5,      # Moderate UX problems
            "low": 0.2          # Minor UX considerations
        }

        logger.info(f"UXDirector initialized with {len(self.knowledge_base.get('ux_best_practices', []))} UX best practices")

    def evaluate(self, task: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate a task from a user experience perspective.

        Args:
            task (Dict[str, Any]): Task information including code, description, requirements
            context (Dict[str, Any]): Additional context including user info, system state

        Returns:
            Dict[str, Any]: Comprehensive UX evaluation result
        """
        try:
            # Extract code and relevant information
            code_content = task.get("code", "")
            task_type = task.get("type", "unknown")
            description = task.get("description", "")

            # Perform comprehensive UX analysis
            ux_findings = self._perform_ux_analysis(code_content, task_type, description)

            # Calculate overall UX score
            ux_score = self._calculate_ux_score(ux_findings)

            # Generate usability assessment
            usability_assessment = self._assess_usability_heuristics(ux_findings)

            # Determine confidence based on analysis completeness
            confidence_factors = {
                "code_coverage": 0.9 if code_content else 0.3,
                "accessibility_analysis": 0.8 if ux_findings.get("accessibility_issues") else 0.5,
                "pattern_matching": 0.7,
                "context_completeness": 0.8 if context.get("requirements") else 0.4
            }
            confidence = self.calculate_confidence(confidence_factors)

            # Generate recommendations
            recommendations_dict = self._generate_ux_recommendations(ux_findings, task_type)
            # Convert to string format expected by registry
            recommendations = [f"{rec['description']} ({rec['implementation']})" for rec in recommendations_dict]

            # Create detailed reasoning
            reasoning_factors = [
                f"Detected {len(ux_findings.get('issues', []))} UX issues",
                f"Overall UX score: {ux_score:.2f}/10",
                f"Accessibility compliance: {ux_findings.get('accessibility_score', 0):.1f}/10",
                f"Usability score: {ux_findings.get('usability_score', 0):.1f}/10",
                f"Responsive design score: {ux_findings.get('responsive_score', 0):.1f}/10"
            ]

            reasoning = self.generate_reasoning(
                assessment=f"UX evaluation completed with {len(ux_findings.get('issues', []))} UX issues identified",
                factors=reasoning_factors + [
                    f"User flow clarity: {ux_findings.get('flow_score', 0):.1f}/10",
                    f"Error handling UX: {ux_findings.get('error_handling_score', 0):.1f}/10",
                    f"Mobile experience: {ux_findings.get('mobile_score', 0):.1f}/10",
                    f"UX risk level: {self._calculate_ux_risk_level(ux_findings)}"
                ],
                context=context
            )

            # Record evaluation
            evaluation_result = {
                "director": self.name,
                "task_type": task_type,
                "timestamp": datetime.now().isoformat(),
                "ux_score": ux_score,
                "confidence": confidence,
                "findings": ux_findings,
                "recommendations": recommendations,
                "usability_assessment": usability_assessment,
                "reasoning": reasoning,
                "metadata": {
                    "evaluation_duration": "computed",
                    "best_practices_checked": len(self.knowledge_base["ux_best_practices"]),
                    "accessibility_guidelines_verified": len(self.knowledge_base["accessibility_guidelines"])
                }
            }

            self.evaluation_history.append(evaluation_result)
            logger.info(f"UX evaluation completed with score: {ux_score:.2f}")

            return evaluation_result

        except Exception as e:
            logger.error(f"Error during UX evaluation: {str(e)}")
            return self._create_error_response(e, task_type)

    def check_accessibility(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Check accessibility compliance (WCAG 2.1).

        Args:
            code (str): Code to analyze for accessibility
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: Accessibility compliance assessment
        """
        accessibility_issues = []
        accessibility_score = 8.0  # Start optimistic

        # WCAG 2.1 compliance patterns
        accessibility_violations = {
            "missing_alt_text": [
                r"<img(?![^>]*alt=)", r"img.*src.*(?!.*alt)",
                r"image.*without.*alt", r"<img[^>]*>"
            ],
            "insufficient_color_contrast": [
                r"color:\s*#[a-f0-9]{3,6}.*background:\s*#[a-f0-9]{3,6}",
                r"low.*contrast", r"color.*contrast.*issue"
            ],
            "missing_semantic_html": [
                r"<div.*button", r"<span.*clickable", r"onclick.*div",
                r"<div.*href", r"non.*semantic.*element"
            ],
            "keyboard_navigation_issues": [
                r"tabindex.*-1", r"no.*focus.*indicator", r"keyboard.*trap",
                r"focus.*outline.*none", r"accessibility.*keyboard"
            ],
            "missing_form_labels": [
                r"<input(?![^>]*id=.*<label[^>]*for=)", r"form.*input.*no.*label",
                r"<input[^>]*>.*(?!<label)"
            ],
            "missing_aria_attributes": [
                r"role.*button.*(?!.*aria-)", r"dropdown.*(?!.*aria-expanded)",
                r"modal.*(?!.*aria-labelledby)", r"interactive.*(?!.*aria-)"
            ],
            "insufficient_heading_structure": [
                r"<h[4-6]>.*<h[1-3]>", r"heading.*structure.*invalid",
                r"skip.*heading.*level"
            ]
        }

        for violation_type, patterns in accessibility_violations.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE | re.MULTILINE):
                    severity = "high" if violation_type in [
                        "missing_alt_text", "missing_form_labels", "keyboard_navigation_issues"
                    ] else "medium"
                    accessibility_issues.append({
                        "type": violation_type,
                        "severity": severity,
                        "description": f"WCAG violation: {violation_type.replace('_', ' ')}",
                        "wcag_guideline": self._get_wcag_guideline(violation_type)
                    })
                    accessibility_score -= 2.0 if severity == "high" else 1.0

        # Check for positive accessibility practices
        accessibility_practices = {
            "semantic_html": [
                r"<button", r"<nav", r"<main", r"<header", r"<footer",
                r"<section", r"<article", r"<aside"
            ],
            "aria_attributes": [
                r"aria-label", r"aria-labelledby", r"aria-describedby",
                r"aria-expanded", r"aria-hidden", r"role="
            ],
            "alt_text": [
                r"alt=", r"alt\s*=\s*['\"].*['\"]"
            ],
            "form_labels": [
                r"<label.*for=", r"aria-label.*input"
            ],
            "focus_management": [
                r"focus\(\)", r"tabindex.*[0-9]", r":focus",
                r"focus.*visible", r"focus.*indicator"
            ]
        }

        for practice_type, patterns in accessibility_practices.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    accessibility_score += 0.3
                    break

        return {
            "accessibility_score": max(0, min(10, accessibility_score)),
            "accessibility_issues": accessibility_issues,
            "wcag_compliance_level": self._assess_wcag_compliance_level(accessibility_score),
            "recommendations": self._get_accessibility_recommendations(accessibility_issues)
        }

    def assess_usability(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Assess usability based on Nielsen's heuristics.

        Args:
            code (str): Code to analyze for usability
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: Usability assessment results
        """
        usability_issues = []
        usability_score = 7.0  # Neutral starting point

        # Nielsen's 10 Usability Heuristics evaluation
        usability_patterns = {
            "visibility_of_system_status": [
                r"loading", r"progress", r"status", r"indicator",
                r"feedback", r"spinner", r"loader"
            ],
            "match_system_real_world": [
                r"intuitive", r"familiar", r"conventional", r"metaphor",
                r"natural.*language", r"user.*friendly"
            ],
            "user_control_and_freedom": [
                r"undo", r"redo", r"back", r"cancel", r"exit",
                r"escape", r"close", r"clear"
            ],
            "consistency_and_standards": [
                r"consistent", r"standard", r"pattern", r"convention",
                r"design.*system", r"style.*guide"
            ],
            "error_prevention": [
                r"validate", r"validation", r"prevent", r"constraint",
                r"format.*check", r"input.*validation"
            ],
            "recognition_rather_than_recall": [
                r"tooltip", r"hint", r"help.*text", r"placeholder",
                r"visible.*options", r"autocomplete"
            ],
            "flexibility_and_efficiency": [
                r"shortcut", r"keyboard", r"hotkey", r"quick.*action",
                r"batch.*operation", r"bulk.*edit"
            ],
            "aesthetic_and_minimalist_design": [
                r"minimal", r"clean", r"simple", r"clutter.*free",
                r"white.*space", r"visual.*hierarchy"
            ],
            "help_users_recognize_diagnose_recover": [
                r"error.*message", r"error.*handling", r"recovery",
                r"help.*text", r"troubleshooting"
            ],
            "help_and_documentation": [
                r"help", r"documentation", r"tutorial", r"guide",
                r"faq", r"support", r"manual"
            ]
        }

        heuristic_scores = {}
        for heuristic, patterns in usability_patterns.items():
            heuristic_score = 0
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    heuristic_score += 1

            heuristic_scores[heuristic] = min(10, heuristic_score * 2)
            if heuristic_score == 0:
                usability_issues.append({
                    "type": f"missing_{heuristic}",
                    "severity": "medium",
                    "description": f"No evidence of {heuristic.replace('_', ' ')} implementation"
                })

        # Calculate average usability score
        usability_score = sum(heuristic_scores.values()) / len(heuristic_scores) if heuristic_scores else 5.0

        # Check for usability anti-patterns
        anti_patterns = {
            "unclear_navigation": [
                r"confusing.*menu", r"unclear.*navigation", r"lost.*user",
                r"no.*breadcrumb", r"missing.*nav"
            ],
            "poor_form_design": [
                r"long.*form", r"required.*field.*not.*marked",
                r"unclear.*error", r"no.*field.*label"
            ],
            "slow_performance": [
                r"slow.*load", r"performance.*issue", r"timeout",
                r"delayed.*response", r"blocking.*operation"
            ]
        }

        for anti_pattern_type, patterns in anti_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    usability_issues.append({
                        "type": anti_pattern_type,
                        "severity": "high",
                        "description": f"Usability anti-pattern: {anti_pattern_type.replace('_', ' ')}"
                    })
                    usability_score -= 1.0

        return {
            "usability_score": max(0, min(10, usability_score)),
            "usability_issues": usability_issues,
            "heuristic_scores": heuristic_scores,
            "recommendations": self._get_usability_recommendations(usability_issues, heuristic_scores)
        }

    def verify_responsive_design(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Verify responsive design implementation.

        Args:
            code (str): Code to analyze for responsive design
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: Responsive design evaluation results
        """
        responsive_issues = []
        responsive_score = 6.0

        # Responsive design patterns
        responsive_patterns = {
            "media_queries": [
                r"@media", r"media.*query", r"breakpoint",
                r"screen.*and.*max-width", r"screen.*and.*min-width"
            ],
            "flexible_grid": [
                r"grid", r"flexbox", r"flex", r"grid-template",
                r"display:\s*grid", r"display:\s*flex"
            ],
            "relative_units": [
                r"em", r"rem", r"%", r"vw", r"vh", r"vmin", r"vmax",
                r"fr", r"auto"
            ],
            "responsive_images": [
                r"max-width:\s*100%", r"height:\s*auto", r"srcset",
                r"picture", r"responsive.*image"
            ],
            "mobile_first": [
                r"mobile.*first", r"progressive.*enhancement",
                r"min-width.*media.*query"
            ],
            "touch_friendly": [
                r"touch-action", r"pointer-events", r"tap-highlight",
                r"user-select", r"touch.*target"
            ]
        }

        responsive_implementations = 0
        for pattern_type, patterns in responsive_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    responsive_score += 0.7
                    responsive_implementations += 1
                    break

        # Check for responsive design issues
        responsive_violations = {
            "fixed_widths": [
                r"width:\s*\d+px", r"min-width:\s*\d+px", r"max-width:\s*\d+px"
            ],
            "small_touch_targets": [
                r"button.*width:\s*[12]?\d+px", r"link.*height:\s*[12]?\d+px",
                r"touch.*target.*small"
            ],
            "horizontal_scrolling": [
                r"overflow-x:\s*scroll", r"overflow-x:\s*auto",
                r"horizontal.*scroll"
            ],
            "viewport_issues": [
                r"viewport.*not.*set", r"missing.*viewport.*meta",
                r"user-scalable.*no"
            ]
        }

        for violation_type, patterns in responsive_violations.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    responsive_issues.append({
                        "type": violation_type,
                        "severity": "medium",
                        "description": f"Responsive design issue: {violation_type.replace('_', ' ')}"
                    })
                    responsive_score -= 1.0

        return {
            "responsive_score": max(0, min(10, responsive_score)),
            "responsive_issues": responsive_issues,
            "implementations": responsive_implementations,
            "mobile_readiness": self._assess_mobile_readiness(responsive_score),
            "recommendations": self._get_responsive_recommendations(responsive_issues)
        }

    def analyze_user_flow(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Analyze user flow and journey clarity.

        Args:
            code (str): Code to analyze for user flow
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: User flow analysis results
        """
        flow_issues = []
        flow_score = 7.0

        # User flow patterns
        flow_patterns = {
            "clear_navigation": [
                r"navigation", r"nav", r"menu", r"breadcrumb",
                r"page.*hierarchy", r"site.*map"
            ],
            "progress_indication": [
                r"step.*indicator", r"progress.*bar", r"wizard",
                r"current.*step", r"step.*\d+.*of.*\d+"
            ],
            "call_to_action": [
                r"cta", r"call.*to.*action", r"primary.*button",
                r"action.*button", r"submit.*button"
            ],
            "information_hierarchy": [
                r"heading", r"title", r"subtitle", r"section",
                r"h1", r"h2", r"h3", r"hierarchy"
            ],
            "search_functionality": [
                r"search", r"find", r"filter", r"sort",
                r"search.*box", r"search.*input"
            ]
        }

        flow_implementations = 0
        for pattern_type, patterns in flow_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    flow_score += 0.5
                    flow_implementations += 1
                    break

        # Check for flow disruption issues
        flow_disruptions = {
            "dead_ends": [
                r"no.*back.*button", r"no.*return.*link", r"dead.*end",
                r"cannot.*go.*back"
            ],
            "confusing_navigation": [
                r"unclear.*path", r"confusing.*menu", r"ambiguous.*link",
                r"misleading.*label"
            ],
            "forced_registration": [
                r"must.*register", r"required.*signup", r"forced.*login",
                r"gate.*content"
            ],
            "complex_checkout": [
                r"too.*many.*steps", r"complex.*checkout", r"abandoned.*cart",
                r"checkout.*issues"
            ]
        }

        for disruption_type, patterns in flow_disruptions.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    flow_issues.append({
                        "type": disruption_type,
                        "severity": "medium",
                        "description": f"User flow issue: {disruption_type.replace('_', ' ')}"
                    })
                    flow_score -= 1.0

        return {
            "flow_score": max(0, min(10, flow_score)),
            "flow_issues": flow_issues,
            "implementations": flow_implementations,
            "journey_clarity": self._assess_journey_clarity(flow_score),
            "recommendations": self._get_flow_recommendations(flow_issues)
        }

    def evaluate_error_handling(self, code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Evaluate error handling user experience.

        Args:
            code (str): Code to analyze for error handling UX
            context (Dict[str, Any]): Additional context

        Returns:
            Dict[str, Any]: Error handling UX evaluation results
        """
        error_handling_issues = []
        error_handling_score = 6.0

        # Good error handling patterns
        good_patterns = {
            "user_friendly_messages": [
                r"user.*friendly.*error", r"helpful.*error.*message",
                r"clear.*error.*text", r"human.*readable.*error"
            ],
            "error_recovery": [
                r"try.*again", r"retry", r"alternative.*action",
                r"recovery.*option", r"fallback"
            ],
            "inline_validation": [
                r"inline.*validation", r"real.*time.*validation",
                r"immediate.*feedback", r"field.*validation"
            ],
            "error_prevention": [
                r"input.*constraint", r"format.*validation",
                r"prevent.*error", r"validate.*before.*submit"
            ],
            "contextual_help": [
                r"contextual.*help", r"field.*help", r"tooltip.*error",
                r"help.*text", r"error.*explanation"
            ]
        }

        for pattern_type, patterns in good_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    error_handling_score += 0.8
                    break

        # Poor error handling patterns
        poor_patterns = {
            "generic_errors": [
                r"error.*occurred", r"something.*went.*wrong",
                r"generic.*error", r"system.*error"
            ],
            "technical_jargon": [
                r"stack.*trace.*user", r"technical.*error.*message",
                r"500.*error", r"internal.*server.*error.*display"
            ],
            "no_error_handling": [
                r"no.*error.*handling", r"silent.*failure",
                r"ignore.*error", r"suppress.*exception"
            ],
            "blame_user": [
                r"invalid.*input", r"wrong.*format", r"user.*error",
                r"incorrect.*data.*entered"
            ]
        }

        for pattern_type, patterns in poor_patterns.items():
            for pattern in patterns:
                if re.search(pattern, code, re.IGNORECASE):
                    severity = "high" if pattern_type in ["no_error_handling", "technical_jargon"] else "medium"
                    error_handling_issues.append({
                        "type": pattern_type,
                        "severity": severity,
                        "description": f"Poor error handling: {pattern_type.replace('_', ' ')}"
                    })
                    error_handling_score -= 1.5 if severity == "high" else 1.0

        return {
            "error_handling_score": max(0, min(10, error_handling_score)),
            "error_handling_issues": error_handling_issues,
            "user_experience_level": self._assess_error_ux_level(error_handling_score),
            "recommendations": self._get_error_handling_recommendations(error_handling_issues)
        }

    def load_knowledge(self) -> Dict[str, List[str]]:
        """Load UX-specific knowledge base."""
        return {
            "ux_best_practices": [
                "Follow accessibility guidelines (WCAG 2.1 AA minimum)",
                "Implement responsive design for all screen sizes",
                "Use semantic HTML elements for better structure",
                "Provide clear visual feedback for user actions",
                "Ensure sufficient color contrast (4.5:1 for normal text)",
                "Make touch targets at least 44x44 pixels",
                "Use consistent design patterns throughout the interface",
                "Provide clear error messages with recovery options",
                "Implement keyboard navigation support",
                "Use loading indicators for operations taking >1 second",
                "Follow the principle of least surprise in interactions",
                "Provide undo functionality for destructive actions",
                "Use progressive disclosure to avoid cognitive overload",
                "Implement search functionality for content-heavy interfaces",
                "Provide clear navigation and site hierarchy",
                "Use familiar icons and conventions",
                "Implement form validation with helpful error messages",
                "Ensure fast page load times (< 3 seconds)",
                "Use white space effectively for visual breathing room",
                "Provide help and documentation when needed",
                "Implement user testing and feedback collection",
                "Follow mobile-first design principles",
                "Use appropriate typography and readability standards",
                "Implement proper focus management for accessibility",
                "Provide alternative text for images and media",
                "Use consistent labeling and terminology",
                "Implement proper heading hierarchy (h1-h6)",
                "Provide skip links for keyboard users",
                "Use ARIA attributes for complex interactions",
                "Test with real users and assistive technologies"
            ],
            "ux_anti_patterns": [
                "Using placeholder text as labels",
                "Implementing unclear or misleading navigation",
                "Creating forms without proper validation",
                "Using insufficient color contrast",
                "Implementing non-responsive design",
                "Creating inaccessible content for screen readers",
                "Using auto-playing videos or audio",
                "Implementing complex CAPTCHA systems",
                "Creating unclear error messages",
                "Using small touch targets on mobile",
                "Implementing modal dialogs without proper focus management",
                "Creating content that relies solely on color to convey meaning",
                "Using flashing or strobing content",
                "Implementing unclear or hidden navigation",
                "Creating forms without clear progress indication",
                "Using technical jargon in user-facing text",
                "Implementing forced registration walls",
                "Creating inconsistent interaction patterns",
                "Using invisible or hard-to-find buttons",
                "Implementing poor search functionality"
            ],
            "ux_risks": [
                "Poor accessibility leading to legal compliance issues",
                "Low user engagement due to usability problems",
                "High bounce rates from unclear navigation",
                "User frustration from poor error handling",
                "Reduced conversion rates from poor form design",
                "Negative brand perception from poor user experience",
                "Loss of users due to non-responsive design",
                "Decreased productivity from inefficient workflows",
                "User abandonment from slow performance",
                "Increased support costs from confusing interfaces",
                "SEO penalties from poor mobile experience",
                "Competitive disadvantage from outdated UX",
                "User safety issues from unclear critical actions",
                "Data loss from poor form validation",
                "User privacy concerns from unclear data practices"
            ],
            "improvement_strategies": [
                "Conduct regular usability testing with real users",
                "Implement accessibility audits and automated testing",
                "Use analytics to identify problem areas",
                "Create and maintain a design system",
                "Implement user feedback collection mechanisms",
                "Conduct competitor analysis for UX benchmarking",
                "Use A/B testing for design decisions",
                "Implement progressive enhancement strategies",
                "Create user personas and journey maps",
                "Establish UX metrics and KPIs for continuous improvement"
            ],
            "accessibility_guidelines": [
                "Provide text alternatives for images (WCAG 1.1.1)",
                "Ensure keyboard accessibility (WCAG 2.1.1)",
                "Use sufficient color contrast (WCAG 1.4.3)",
                "Make content adaptable to different presentations (WCAG 1.3.1)",
                "Help users navigate and find content (WCAG 2.4.1)",
                "Make it easier for users to see and hear content (WCAG 1.4.1)",
                "Make all functionality available from keyboard (WCAG 2.1.1)",
                "Give users enough time to read content (WCAG 2.2.1)",
                "Don't use content that causes seizures (WCAG 2.3.1)",
                "Help users navigate and find content (WCAG 2.4.1)",
                "Make text readable and understandable (WCAG 3.1.1)",
                "Make content appear and operate predictably (WCAG 3.2.1)",
                "Help users avoid and correct mistakes (WCAG 3.3.1)",
                "Maximize compatibility with assistive technologies (WCAG 4.1.1)",
                "Ensure content is robust enough for various user agents (WCAG 4.1.2)"
            ]
        }

    def _load_accessibility_patterns(self) -> Dict[str, List[str]]:
        """Load accessibility detection patterns."""
        return {
            "semantic_html": [r"<button", r"<nav", r"<main", r"<header", r"<footer"],
            "aria_attributes": [r"aria-label", r"aria-labelledby", r"aria-describedby"],
            "alt_text": [r"alt=", r"alt\s*=\s*['\"].*['\"]"],
            "form_labels": [r"<label.*for=", r"aria-label.*input"],
            "keyboard_navigation": [r"tabindex", r"focus", r"keydown", r"keyup"]
        }

    def _perform_ux_analysis(self, code: str, task_type: str, description: str) -> Dict[str, Any]:
        """Perform comprehensive UX analysis."""
        findings = {
            "issues": [],
            "accessibility_score": 8.0,
            "usability_score": 7.0,
            "responsive_score": 6.0,
            "flow_score": 7.0,
            "error_handling_score": 6.0,
            "mobile_score": 6.0
        }

        if code:
            # Accessibility analysis
            accessibility_analysis = self.check_accessibility(code)
            findings["accessibility_score"] = accessibility_analysis["accessibility_score"]
            findings["issues"].extend(accessibility_analysis["accessibility_issues"])

            # Usability assessment
            usability_analysis = self.assess_usability(code)
            findings["usability_score"] = usability_analysis["usability_score"]
            findings["issues"].extend(usability_analysis["usability_issues"])

            # Responsive design verification
            responsive_analysis = self.verify_responsive_design(code)
            findings["responsive_score"] = responsive_analysis["responsive_score"]
            findings["mobile_score"] = responsive_analysis["responsive_score"]  # Mobile score based on responsive design
            findings["issues"].extend(responsive_analysis["responsive_issues"])

            # User flow analysis
            flow_analysis = self.analyze_user_flow(code)
            findings["flow_score"] = flow_analysis["flow_score"]
            findings["issues"].extend(flow_analysis["flow_issues"])

            # Error handling evaluation
            error_analysis = self.evaluate_error_handling(code)
            findings["error_handling_score"] = error_analysis["error_handling_score"]
            findings["issues"].extend(error_analysis["error_handling_issues"])

        return findings

    def _calculate_ux_score(self, findings: Dict[str, Any]) -> float:
        """Calculate overall UX score based on findings."""
        # Weighted combination of different UX dimensions
        weights = {
            "accessibility": 0.25,
            "usability": 0.25,
            "responsive": 0.20,
            "flow": 0.15,
            "error_handling": 0.15
        }

        score = (
            findings.get("accessibility_score", 0) * weights["accessibility"] +
            findings.get("usability_score", 0) * weights["usability"] +
            findings.get("responsive_score", 0) * weights["responsive"] +
            findings.get("flow_score", 0) * weights["flow"] +
            findings.get("error_handling_score", 0) * weights["error_handling"]
        )

        # Deduct points for critical issues
        critical_issues = len([issue for issue in findings.get("issues", [])
                              if issue.get("severity") == "high"])
        score -= critical_issues * 0.8

        return max(0.0, min(10.0, score))

    def _assess_usability_heuristics(self, findings: Dict[str, Any]) -> Dict[str, Any]:
        """Assess usability based on Nielsen's heuristics."""
        return {
            "overall_usability": findings.get("usability_score", 0),
            "heuristic_compliance": "partial",  # Would need more detailed analysis
            "user_satisfaction_estimate": self._estimate_user_satisfaction(findings),
            "task_completion_likelihood": self._estimate_task_completion(findings)
        }

    def _generate_ux_recommendations(self, findings: Dict[str, Any], task_type: str) -> List[Dict[str, Any]]:
        """Generate specific UX recommendations."""
        recommendations = []

        # Accessibility recommendations
        if findings.get("accessibility_score", 0) < 8.0:
            recommendations.append({
                "type": "accessibility_improvement",
                "priority": "high",
                "description": "Address accessibility compliance issues",
                "implementation": "Add alt text, improve keyboard navigation, enhance color contrast"
            })

        # Usability recommendations
        if findings.get("usability_score", 0) < 7.0:
            recommendations.append({
                "type": "usability_enhancement",
                "priority": "high",
                "description": "Improve overall usability",
                "implementation": "Enhance navigation clarity, improve error messages, add user feedback"
            })

        # Responsive design recommendations
        if findings.get("responsive_score", 0) < 7.0:
            recommendations.append({
                "type": "responsive_design",
                "priority": "medium",
                "description": "Enhance responsive design implementation",
                "implementation": "Add media queries, use flexible layouts, optimize for mobile"
            })

        # User flow recommendations
        if findings.get("flow_score", 0) < 7.0:
            recommendations.append({
                "type": "user_flow_optimization",
                "priority": "medium",
                "description": "Improve user flow and navigation",
                "implementation": "Simplify navigation, add progress indicators, improve information hierarchy"
            })

        # Error handling recommendations
        if findings.get("error_handling_score", 0) < 6.0:
            recommendations.append({
                "type": "error_handling_improvement",
                "priority": "medium",
                "description": "Enhance error handling user experience",
                "implementation": "Provide clearer error messages, add recovery options, implement inline validation"
            })

        return recommendations

    def _calculate_ux_risk_level(self, findings: Dict[str, Any]) -> str:
        """Calculate overall UX risk level."""
        risk_factors = 0

        # High-risk indicators
        if findings.get("accessibility_score", 10) < 6.0:
            risk_factors += 3
        elif findings.get("accessibility_score", 10) < 8.0:
            risk_factors += 1

        if findings.get("usability_score", 10) < 5.0:
            risk_factors += 2
        elif findings.get("usability_score", 10) < 7.0:
            risk_factors += 1

        high_severity_issues = len([issue for issue in findings.get("issues", [])
                                  if issue.get("severity") == "high"])
        risk_factors += high_severity_issues

        if risk_factors >= 4:
            return "High"
        elif risk_factors >= 2:
            return "Medium"
        else:
            return "Low"

    def _get_wcag_guideline(self, violation_type: str) -> str:
        """Get relevant WCAG guideline for violation type."""
        wcag_mapping = {
            "missing_alt_text": "WCAG 1.1.1 Non-text Content",
            "insufficient_color_contrast": "WCAG 1.4.3 Contrast (Minimum)",
            "missing_semantic_html": "WCAG 1.3.1 Info and Relationships",
            "keyboard_navigation_issues": "WCAG 2.1.1 Keyboard",
            "missing_form_labels": "WCAG 1.3.1 Info and Relationships",
            "missing_aria_attributes": "WCAG 4.1.2 Name, Role, Value",
            "insufficient_heading_structure": "WCAG 1.3.1 Info and Relationships"
        }
        return wcag_mapping.get(violation_type, "WCAG Compliance Required")

    def _assess_wcag_compliance_level(self, accessibility_score: float) -> str:
        """Assess WCAG compliance level."""
        if accessibility_score >= 9.0:
            return "AAA"
        elif accessibility_score >= 7.0:
            return "AA"
        elif accessibility_score >= 5.0:
            return "A"
        else:
            return "Non-compliant"

    def _assess_mobile_readiness(self, responsive_score: float) -> str:
        """Assess mobile readiness level."""
        if responsive_score >= 8.0:
            return "Excellent"
        elif responsive_score >= 6.0:
            return "Good"
        elif responsive_score >= 4.0:
            return "Fair"
        else:
            return "Poor"

    def _assess_journey_clarity(self, flow_score: float) -> str:
        """Assess user journey clarity."""
        if flow_score >= 8.0:
            return "Clear"
        elif flow_score >= 6.0:
            return "Moderate"
        else:
            return "Confusing"

    def _assess_error_ux_level(self, error_handling_score: float) -> str:
        """Assess error handling UX level."""
        if error_handling_score >= 8.0:
            return "Excellent"
        elif error_handling_score >= 6.0:
            return "Good"
        elif error_handling_score >= 4.0:
            return "Fair"
        else:
            return "Poor"

    def _estimate_user_satisfaction(self, findings: Dict[str, Any]) -> str:
        """Estimate user satisfaction level."""
        avg_score = (
            findings.get("accessibility_score", 0) +
            findings.get("usability_score", 0) +
            findings.get("responsive_score", 0) +
            findings.get("flow_score", 0) +
            findings.get("error_handling_score", 0)
        ) / 5

        if avg_score >= 8.0:
            return "High"
        elif avg_score >= 6.0:
            return "Medium"
        else:
            return "Low"

    def _estimate_task_completion(self, findings: Dict[str, Any]) -> str:
        """Estimate task completion likelihood."""
        # Flow score is most important for task completion
        flow_score = findings.get("flow_score", 0)
        error_score = findings.get("error_handling_score", 0)

        combined_score = (flow_score * 0.6) + (error_score * 0.4)

        if combined_score >= 8.0:
            return "High"
        elif combined_score >= 6.0:
            return "Medium"
        else:
            return "Low"

    def _get_accessibility_recommendations(self, accessibility_issues: List[Dict[str, Any]]) -> List[str]:
        """Get specific accessibility recommendations."""
        recommendations = []
        issue_types = {issue.get("type") for issue in accessibility_issues}

        if "missing_alt_text" in issue_types:
            recommendations.append("Add descriptive alt text for all images")

        if "missing_form_labels" in issue_types:
            recommendations.append("Associate labels with form inputs using 'for' attribute")

        if "keyboard_navigation_issues" in issue_types:
            recommendations.append("Ensure all interactive elements are keyboard accessible")

        if "insufficient_color_contrast" in issue_types:
            recommendations.append("Increase color contrast to meet WCAG AA standards (4.5:1)")

        if "missing_semantic_html" in issue_types:
            recommendations.append("Use semantic HTML elements (button, nav, main, etc.)")

        return recommendations

    def _get_usability_recommendations(self, usability_issues: List[Dict[str, Any]], heuristic_scores: Dict[str, float]) -> List[str]:
        """Get specific usability recommendations."""
        recommendations = []

        # Check which heuristics scored low
        for heuristic, score in heuristic_scores.items():
            if score < 5.0:
                if "visibility_of_system_status" in heuristic:
                    recommendations.append("Add loading indicators and system status feedback")
                elif "user_control_and_freedom" in heuristic:
                    recommendations.append("Provide undo/redo functionality and clear exits")
                elif "error_prevention" in heuristic:
                    recommendations.append("Implement input validation and error prevention")
                elif "consistency_and_standards" in heuristic:
                    recommendations.append("Ensure consistent design patterns throughout")

        return recommendations

    def _get_responsive_recommendations(self, responsive_issues: List[Dict[str, Any]]) -> List[str]:
        """Get specific responsive design recommendations."""
        recommendations = []
        issue_types = {issue.get("type") for issue in responsive_issues}

        if "fixed_widths" in issue_types:
            recommendations.append("Replace fixed pixel widths with relative units")

        if "small_touch_targets" in issue_types:
            recommendations.append("Ensure touch targets are at least 44x44 pixels")

        if "viewport_issues" in issue_types:
            recommendations.append("Add proper viewport meta tag")

        if "horizontal_scrolling" in issue_types:
            recommendations.append("Prevent horizontal scrolling on mobile devices")

        return recommendations

    def _get_flow_recommendations(self, flow_issues: List[Dict[str, Any]]) -> List[str]:
        """Get specific user flow recommendations."""
        recommendations = []
        issue_types = {issue.get("type") for issue in flow_issues}

        if "dead_ends" in issue_types:
            recommendations.append("Provide clear navigation paths from all pages")

        if "confusing_navigation" in issue_types:
            recommendations.append("Simplify navigation structure and labels")

        if "complex_checkout" in issue_types:
            recommendations.append("Streamline checkout process and reduce steps")

        return recommendations

    def _get_error_handling_recommendations(self, error_handling_issues: List[Dict[str, Any]]) -> List[str]:
        """Get specific error handling recommendations."""
        recommendations = []
        issue_types = {issue.get("type") for issue in error_handling_issues}

        if "generic_errors" in issue_types:
            recommendations.append("Provide specific, actionable error messages")

        if "technical_jargon" in issue_types:
            recommendations.append("Use plain language in error messages")

        if "no_error_handling" in issue_types:
            recommendations.append("Implement comprehensive error handling")

        if "blame_user" in issue_types:
            recommendations.append("Frame errors as system guidance rather than user blame")

        return recommendations

    def _create_error_response(self, error: Exception, task_type: str) -> Dict[str, Any]:
        """Create standardized error response."""
        return {
            "director": self.name,
            "task_type": task_type,
            "timestamp": datetime.now().isoformat(),
            "ux_score": 0.0,
            "confidence": 0.0,
            "error": str(error),
            "status": "error",
            "recommendations": [f"Resolve evaluation error before proceeding (Address the following error: {str(error)})"]
        }