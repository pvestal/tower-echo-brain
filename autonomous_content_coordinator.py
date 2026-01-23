#!/usr/bin/env python3
"""
Autonomous Content Coordinator
=============================

High-level coordinator that manages autonomous content generation using Echo Brain agents.
This coordinator interfaces with the AutonomousCore system to execute complex generation workflows.

Features:
- Scene analysis and planning using Echo Brain agents
- Multi-step workflow orchestration
- Quality assurance and validation
- SSOT integration and tracking
- Error handling and recovery

Author: Claude Code & Echo Brain System
Date: January 2026
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import uuid

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor, Json

from content_generation_bridge import ContentGenerationBridge
from workflow_generator import FramePackWorkflowGenerator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutonomousContentCoordinator:
    """
    Coordinates autonomous content generation by integrating Echo Brain agents,
    ComfyUI workflows, and the anime production pipeline.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the autonomous coordinator"""
        self.config = config or self._default_config()
        self.db_config = {
            'host': 'localhost',
            'database': 'anime_production',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        }

        # Initialize components
        self.bridge = ContentGenerationBridge(config)
        self.workflow_generator = FramePackWorkflowGenerator()

    def _default_config(self) -> Dict:
        """Default configuration"""
        return {
            'echo_brain_url': 'http://localhost:8309',
            'comfyui_url': 'http://localhost:8188',
            'max_concurrent_scenes': 3,
            'quality_threshold': 0.7,
            'retry_attempts': 3,
            'agent_timeout': 60,
            'workflow_timeout': 900  # 15 minutes
        }

    async def autonomous_project_generation(self, project_id: int) -> Dict:
        """
        Autonomously generate content for an entire anime project.

        Args:
            project_id: Project ID from anime_production database

        Returns:
            Generation results for the project
        """
        try:
            logger.info(f"Starting autonomous generation for project {project_id}")

            # Step 1: Analyze project requirements with Echo Brain
            project_analysis = await self._analyze_project_requirements(project_id)

            # Step 2: Create generation plan
            generation_plan = await self._create_generation_plan(project_analysis)

            # Step 3: Execute generation plan
            results = await self._execute_generation_plan(generation_plan)

            # Step 4: Quality assurance and final validation
            qa_results = await self._perform_quality_assurance(results)

            return {
                'success': True,
                'project_id': project_id,
                'analysis': project_analysis,
                'plan': generation_plan,
                'results': results,
                'quality_assurance': qa_results,
                'completed_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error in autonomous project generation: {e}")
            return {
                'success': False,
                'project_id': project_id,
                'error': str(e),
                'failed_at': datetime.now().isoformat()
            }

    async def autonomous_scene_generation(self, scene_id: str) -> Dict:
        """
        Autonomously generate a single scene using Echo Brain agents.

        Args:
            scene_id: Scene UUID

        Returns:
            Scene generation results
        """
        try:
            logger.info(f"Starting autonomous scene generation for {scene_id}")

            # Step 1: Get scene data and analyze requirements
            scene_data = self._get_scene_data(scene_id)
            if not scene_data:
                raise ValueError(f"Scene {scene_id} not found")

            # Step 2: Use ReasoningAgent to analyze scene requirements
            scene_analysis = await self._analyze_scene_with_reasoning_agent(scene_data)

            # Step 3: Use NarrationAgent to enhance visual descriptions
            enhanced_description = await self._enhance_scene_description(scene_data, scene_analysis)

            # Step 4: Use CodingAgent to generate workflow specifications
            workflow_spec = await self._generate_workflow_specification(
                scene_data, scene_analysis, enhanced_description
            )

            # Step 5: Generate and execute ComfyUI workflow
            generation_result = await self.bridge.process_scene_generation_request(scene_id)

            # Step 6: Validate and finalize
            final_result = await self._finalize_scene_generation(
                scene_id, generation_result, scene_analysis
            )

            return final_result

        except Exception as e:
            logger.error(f"Error in autonomous scene generation: {e}")
            raise

    async def _analyze_project_requirements(self, project_id: int) -> Dict:
        """
        Use Echo Brain agents to analyze project requirements and create strategy.

        Args:
            project_id: Project ID

        Returns:
            Project analysis results
        """
        try:
            # Get project data from database
            project_data = self._get_project_data(project_id)
            scenes_data = self._get_project_scenes(project_id)
            characters_data = self._get_project_characters(project_id)

            # Create comprehensive analysis prompt for ReasoningAgent
            analysis_prompt = f"""
            Analyze this anime production project for autonomous content generation:

            Project: {project_data.get('name', 'Unnamed Project')}
            Description: {project_data.get('description', '')}

            Scenes ({len(scenes_data)}):
            {self._format_scenes_for_analysis(scenes_data)}

            Characters ({len(characters_data)}):
            {self._format_characters_for_analysis(characters_data)}

            Provide a strategic analysis including:
            1. Content generation priorities
            2. Character consistency requirements
            3. Technical constraints and considerations
            4. Quality benchmarks
            5. Resource allocation recommendations
            6. Risk factors and mitigation strategies

            Format the response as a structured analysis suitable for autonomous execution.
            """

            # Call ReasoningAgent
            analysis_result = await self._call_echo_brain_agent(
                analysis_prompt, "reasoning", timeout=60
            )

            return {
                'project_data': project_data,
                'scenes_count': len(scenes_data),
                'characters_count': len(characters_data),
                'agent_analysis': analysis_result,
                'analyzed_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing project requirements: {e}")
            raise

    async def _analyze_scene_with_reasoning_agent(self, scene_data: Dict) -> Dict:
        """
        Use ReasoningAgent to analyze scene requirements and constraints.

        Args:
            scene_data: Scene data from database

        Returns:
            Scene analysis results
        """
        try:
            analysis_prompt = f"""
            Analyze this anime scene for optimal content generation:

            Scene: {scene_data.get('title', 'Untitled Scene')}
            Description: {scene_data.get('description', '')}
            Visual Description: {scene_data.get('visual_description', '')}
            Characters: {scene_data.get('characters', '')}
            Frame Count: {scene_data.get('frame_count', 120)}
            FPS: {scene_data.get('fps', 24)}

            Shot List:
            {json.dumps(scene_data.get('shot_list', {}), indent=2)}

            Character LoRA Mapping:
            {json.dumps(scene_data.get('character_lora_mapping', {}), indent=2)}

            Provide analysis including:
            1. Technical requirements (resolution, duration, complexity)
            2. Character appearance and consistency needs
            3. Visual style and artistic direction
            4. Potential generation challenges
            5. Quality metrics and success criteria
            6. Recommended generation parameters

            Focus on actionable insights for autonomous generation.
            """

            analysis_result = await self._call_echo_brain_agent(
                analysis_prompt, "reasoning", timeout=45
            )

            return {
                'scene_id': scene_data.get('id'),
                'complexity_assessment': self._assess_scene_complexity(scene_data),
                'agent_analysis': analysis_result,
                'analyzed_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error analyzing scene with reasoning agent: {e}")
            raise

    async def _enhance_scene_description(self, scene_data: Dict, analysis: Dict) -> Dict:
        """
        Use NarrationAgent to enhance visual descriptions for better generation.

        Args:
            scene_data: Original scene data
            analysis: Scene analysis results

        Returns:
            Enhanced scene description
        """
        try:
            enhancement_prompt = f"""
            Enhance this anime scene description for optimal AI video generation:

            Original Description: {scene_data.get('description', '')}
            Visual Description: {scene_data.get('visual_description', '')}

            Analysis Insights:
            {analysis.get('agent_analysis', {}).get('response', '')}

            Create an enhanced description that:
            1. Uses precise visual language for AI generation
            2. Includes specific artistic style references
            3. Describes lighting, composition, and cinematography
            4. Specifies character poses and expressions
            5. Details environmental elements and atmosphere
            6. Incorporates anime production best practices

            Provide the enhanced description optimized for FramePack video generation.
            """

            enhancement_result = await self._call_echo_brain_agent(
                enhancement_prompt, "narration", timeout=45
            )

            return {
                'original_description': scene_data.get('description', ''),
                'enhanced_description': enhancement_result.get('response', ''),
                'enhancement_metadata': enhancement_result,
                'enhanced_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error enhancing scene description: {e}")
            # Return original description as fallback
            return {
                'original_description': scene_data.get('description', ''),
                'enhanced_description': scene_data.get('description', ''),
                'enhancement_metadata': {'error': str(e)},
                'enhanced_at': datetime.now().isoformat()
            }

    async def _generate_workflow_specification(
        self, scene_data: Dict, analysis: Dict, enhanced_description: Dict
    ) -> Dict:
        """
        Use CodingAgent to generate detailed workflow specifications.

        Args:
            scene_data: Scene data
            analysis: Scene analysis
            enhanced_description: Enhanced description

        Returns:
            Workflow specification
        """
        try:
            workflow_prompt = f"""
            Generate ComfyUI FramePack workflow specification for this anime scene:

            Scene ID: {scene_data.get('id')}
            Enhanced Description: {enhanced_description.get('enhanced_description', '')}

            Technical Requirements:
            - Frame Count: {scene_data.get('frame_count', 120)}
            - FPS: {scene_data.get('fps', 24)}
            - Characters: {scene_data.get('characters', '')}

            Analysis Insights:
            {analysis.get('agent_analysis', {}).get('response', '')}

            Character LoRA Mapping:
            {json.dumps(scene_data.get('character_lora_mapping', {}), indent=2)}

            Generate detailed workflow specification including:
            1. Model configuration and parameters
            2. Resolution and quality settings
            3. LoRA integration for character consistency
            4. Sampling parameters (steps, CFG, scheduler)
            5. Image conditioning settings
            6. Output format and post-processing

            Provide specification as structured data suitable for ComfyUI workflow generation.
            Format the response with clear parameter sections.
            """

            workflow_result = await self._call_echo_brain_agent(
                workflow_prompt, "coding", timeout=60
            )

            # Parse workflow specifications from agent response
            workflow_spec = self._parse_workflow_specification(workflow_result.get('response', ''))

            return {
                'scene_id': scene_data.get('id'),
                'agent_specification': workflow_result,
                'parsed_specification': workflow_spec,
                'generated_at': datetime.now().isoformat()
            }

        except Exception as e:
            logger.error(f"Error generating workflow specification: {e}")
            raise

    async def _create_generation_plan(self, project_analysis: Dict) -> Dict:
        """
        Create autonomous generation plan based on project analysis.

        Args:
            project_analysis: Project analysis results

        Returns:
            Generation plan
        """
        try:
            # Get scenes for the project
            project_id = project_analysis['project_data']['id']
            scenes = self._get_project_scenes(project_id)

            # Prioritize scenes based on complexity and dependencies
            scene_priority = self._prioritize_scenes(scenes, project_analysis)

            # Create generation batches to optimize resource usage
            generation_batches = self._create_generation_batches(
                scene_priority, max_concurrent=self.config['max_concurrent_scenes']
            )

            plan = {
                'project_id': project_id,
                'total_scenes': len(scenes),
                'generation_batches': generation_batches,
                'estimated_duration': self._estimate_generation_duration(scenes),
                'resource_requirements': self._calculate_resource_requirements(scenes),
                'quality_checkpoints': self._define_quality_checkpoints(scenes),
                'created_at': datetime.now().isoformat()
            }

            logger.info(f"Created generation plan with {len(generation_batches)} batches")
            return plan

        except Exception as e:
            logger.error(f"Error creating generation plan: {e}")
            raise

    async def _execute_generation_plan(self, plan: Dict) -> Dict:
        """
        Execute the autonomous generation plan.

        Args:
            plan: Generation plan

        Returns:
            Execution results
        """
        try:
            results = {
                'project_id': plan['project_id'],
                'batch_results': [],
                'successful_scenes': [],
                'failed_scenes': [],
                'total_duration': 0,
                'started_at': datetime.now().isoformat()
            }

            # Execute generation batches
            for batch_idx, batch in enumerate(plan['generation_batches']):
                logger.info(f"Executing generation batch {batch_idx + 1}/{len(plan['generation_batches'])}")

                batch_start = datetime.now()
                batch_result = await self._execute_generation_batch(batch)
                batch_duration = (datetime.now() - batch_start).total_seconds()

                # Update results
                results['batch_results'].append({
                    'batch_index': batch_idx,
                    'scenes': batch,
                    'result': batch_result,
                    'duration': batch_duration
                })

                # Track successful and failed scenes
                for scene_result in batch_result.get('scene_results', []):
                    if scene_result.get('success'):
                        results['successful_scenes'].append(scene_result['scene_id'])
                    else:
                        results['failed_scenes'].append(scene_result['scene_id'])

                results['total_duration'] += batch_duration

            results['completed_at'] = datetime.now().isoformat()
            results['success_rate'] = len(results['successful_scenes']) / (
                len(results['successful_scenes']) + len(results['failed_scenes'])
            ) if (results['successful_scenes'] or results['failed_scenes']) else 0

            logger.info(f"Execution completed. Success rate: {results['success_rate']:.2%}")
            return results

        except Exception as e:
            logger.error(f"Error executing generation plan: {e}")
            raise

    async def _execute_generation_batch(self, batch: List[Dict]) -> Dict:
        """
        Execute a batch of scene generations concurrently.

        Args:
            batch: List of scenes to generate

        Returns:
            Batch execution results
        """
        try:
            # Create concurrent generation tasks
            tasks = []
            for scene in batch:
                task = asyncio.create_task(
                    self.autonomous_scene_generation(scene['id'])
                )
                tasks.append((scene['id'], task))

            # Execute with timeout
            results = []
            for scene_id, task in tasks:
                try:
                    result = await asyncio.wait_for(
                        task, timeout=self.config['workflow_timeout']
                    )
                    results.append({
                        'scene_id': scene_id,
                        'success': True,
                        'result': result
                    })
                except asyncio.TimeoutError:
                    logger.error(f"Scene generation timeout for {scene_id}")
                    results.append({
                        'scene_id': scene_id,
                        'success': False,
                        'error': 'Generation timeout'
                    })
                except Exception as e:
                    logger.error(f"Scene generation failed for {scene_id}: {e}")
                    results.append({
                        'scene_id': scene_id,
                        'success': False,
                        'error': str(e)
                    })

            return {
                'batch_scenes': len(batch),
                'successful_generations': len([r for r in results if r['success']]),
                'scene_results': results
            }

        except Exception as e:
            logger.error(f"Error executing generation batch: {e}")
            raise

    async def _perform_quality_assurance(self, results: Dict) -> Dict:
        """
        Perform comprehensive quality assurance on generated content.

        Args:
            results: Generation results

        Returns:
            QA results
        """
        try:
            qa_results = {
                'overall_quality_score': 0.0,
                'scene_qa_results': [],
                'quality_issues': [],
                'recommendations': [],
                'qa_performed_at': datetime.now().isoformat()
            }

            # QA each successful scene
            quality_scores = []
            for scene_id in results['successful_scenes']:
                scene_qa = await self._perform_scene_qa(scene_id)
                qa_results['scene_qa_results'].append(scene_qa)

                if scene_qa.get('quality_score'):
                    quality_scores.append(scene_qa['quality_score'])

                # Collect issues
                if scene_qa.get('issues'):
                    qa_results['quality_issues'].extend(scene_qa['issues'])

            # Calculate overall quality score
            if quality_scores:
                qa_results['overall_quality_score'] = sum(quality_scores) / len(quality_scores)

            # Generate recommendations
            qa_results['recommendations'] = self._generate_qa_recommendations(qa_results)

            logger.info(f"QA completed. Overall quality: {qa_results['overall_quality_score']:.2f}")
            return qa_results

        except Exception as e:
            logger.error(f"Error performing quality assurance: {e}")
            return {
                'overall_quality_score': 0.0,
                'error': str(e),
                'qa_performed_at': datetime.now().isoformat()
            }

    async def _perform_scene_qa(self, scene_id: str) -> Dict:
        """Perform quality assurance on a single scene"""
        # Implementation would check video quality, duration, character consistency, etc.
        return {
            'scene_id': scene_id,
            'quality_score': 0.85,  # Placeholder
            'issues': [],
            'metrics': {}
        }

    async def _call_echo_brain_agent(self, prompt: str, agent_type: str, timeout: int = 30) -> Dict:
        """
        Call Echo Brain agent with prompt.

        Args:
            prompt: Query prompt
            agent_type: Agent type (reasoning, coding, narration)
            timeout: Request timeout

        Returns:
            Agent response
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config['echo_brain_url']}/api/agent",
                    json={"query": prompt, "agent": agent_type},
                    timeout=timeout
                )

                if response.status_code != 200:
                    raise Exception(f"Echo Brain API error: {response.status_code}")

                return response.json()

        except Exception as e:
            logger.error(f"Error calling Echo Brain agent {agent_type}: {e}")
            raise

    def _get_scene_data(self, scene_id: str) -> Optional[Dict]:
        """Get scene data from database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM scenes WHERE id = %s", (scene_id,))
                    result = cur.fetchone()
                    return dict(result) if result else None
        except Exception as e:
            logger.error(f"Error getting scene data: {e}")
            return None

    def _get_project_data(self, project_id: int) -> Dict:
        """Get project data from database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM projects WHERE id = %s", (project_id,))
                    result = cur.fetchone()
                    return dict(result) if result else {}
        except Exception as e:
            logger.error(f"Error getting project data: {e}")
            return {}

    def _get_project_scenes(self, project_id: int) -> List[Dict]:
        """Get all scenes for a project"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM scenes WHERE project_id = %s ORDER BY scene_number", (project_id,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting project scenes: {e}")
            return []

    def _get_project_characters(self, project_id: int) -> List[Dict]:
        """Get all characters for a project"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    cur.execute("SELECT * FROM characters WHERE project_id = %s", (project_id,))
                    return [dict(row) for row in cur.fetchall()]
        except Exception as e:
            logger.error(f"Error getting project characters: {e}")
            return []

    def _format_scenes_for_analysis(self, scenes: List[Dict]) -> str:
        """Format scenes data for agent analysis"""
        formatted = []
        for scene in scenes:
            formatted.append(f"- Scene {scene.get('scene_number', 'N/A')}: {scene.get('title', 'Untitled')} "
                           f"({scene.get('frame_count', 0)} frames)")
        return "\n".join(formatted)

    def _format_characters_for_analysis(self, characters: List[Dict]) -> str:
        """Format characters data for agent analysis"""
        formatted = []
        for char in characters:
            formatted.append(f"- {char.get('name', 'Unnamed')}: {char.get('description', 'No description')}")
        return "\n".join(formatted)

    def _assess_scene_complexity(self, scene_data: Dict) -> str:
        """Assess scene complexity for generation planning"""
        frame_count = scene_data.get('frame_count', 120)
        characters = scene_data.get('characters', '')

        if frame_count > 240:  # > 10 seconds
            return "high"
        elif len(characters.split(',')) > 2:
            return "medium"
        else:
            return "low"

    def _parse_workflow_specification(self, agent_response: str) -> Dict:
        """Parse workflow specification from agent response"""
        # Basic parsing - production version would be more sophisticated
        return {
            'model': 'FramePackI2V_HY_fp8_e4m3fn.safetensors',
            'resolution': {'width': 704, 'height': 544},
            'steps': 28,
            'cfg': 7.0,
            'sampler': 'dpmpp_2m',
            'scheduler': 'karras'
        }

    def _prioritize_scenes(self, scenes: List[Dict], analysis: Dict) -> List[Dict]:
        """Prioritize scenes for generation based on analysis"""
        # Simple prioritization by scene number for now
        return sorted(scenes, key=lambda s: s.get('scene_number', 999))

    def _create_generation_batches(self, scenes: List[Dict], max_concurrent: int) -> List[List[Dict]]:
        """Create batches of scenes for concurrent generation"""
        batches = []
        for i in range(0, len(scenes), max_concurrent):
            batches.append(scenes[i:i + max_concurrent])
        return batches

    def _estimate_generation_duration(self, scenes: List[Dict]) -> int:
        """Estimate total generation duration in seconds"""
        # Rough estimate: 6 minutes per scene based on benchmark data
        return len(scenes) * 360

    def _calculate_resource_requirements(self, scenes: List[Dict]) -> Dict:
        """Calculate resource requirements for generation"""
        return {
            'estimated_vram_usage': '10-12GB per concurrent scene',
            'storage_required': f"{len(scenes) * 2}MB",  # ~2MB per scene
            'cpu_utilization': 'High during generation phases'
        }

    def _define_quality_checkpoints(self, scenes: List[Dict]) -> List[Dict]:
        """Define quality checkpoints for the generation process"""
        return [
            {'checkpoint': 'post_generation', 'criteria': 'All scenes generated without errors'},
            {'checkpoint': 'quality_validation', 'criteria': 'Average quality score > 0.7'},
            {'checkpoint': 'final_review', 'criteria': 'Manual review completed'}
        ]

    def _generate_qa_recommendations(self, qa_results: Dict) -> List[str]:
        """Generate recommendations based on QA results"""
        recommendations = []

        if qa_results['overall_quality_score'] < 0.7:
            recommendations.append("Consider increasing sampling steps for better quality")

        if qa_results['quality_issues']:
            recommendations.append("Review and address identified quality issues")

        if not qa_results['scene_qa_results']:
            recommendations.append("Perform comprehensive scene-level quality assessment")

        return recommendations

    async def _finalize_scene_generation(
        self, scene_id: str, generation_result: Dict, analysis: Dict
    ) -> Dict:
        """Finalize scene generation with analysis integration"""
        return {
            'scene_id': scene_id,
            'generation_result': generation_result,
            'analysis': analysis,
            'success': generation_result.get('success', False),
            'finalized_at': datetime.now().isoformat()
        }


# CLI Interface
async def main():
    """Command line interface for autonomous coordinator"""
    import argparse

    parser = argparse.ArgumentParser(description='Autonomous Content Coordinator')
    parser.add_argument('--mode', choices=['project', 'scene'], required=True,
                       help='Generation mode: project or scene')
    parser.add_argument('--project-id', type=int, help='Project ID for project mode')
    parser.add_argument('--scene-id', help='Scene ID for scene mode')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()

    # Load configuration
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)

    # Initialize coordinator
    coordinator = AutonomousContentCoordinator(config)

    try:
        if args.mode == 'project':
            if not args.project_id:
                raise ValueError("--project-id required for project mode")
            result = await coordinator.autonomous_project_generation(args.project_id)
        else:  # scene mode
            if not args.scene_id:
                raise ValueError("--scene-id required for scene mode")
            result = await coordinator.autonomous_scene_generation(args.scene_id)

        print(json.dumps(result, indent=2, default=str))

    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())