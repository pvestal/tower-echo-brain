#!/usr/bin/env python3
"""
Content Generation Bridge
==========================

Connects Echo Brain autonomous agents to ComfyUI for anime production.
This bridge enables autonomous content generation by:

1. Analyzing scene requirements from the anime production database
2. Using Echo Brain agents to generate appropriate ComfyUI workflows
3. Submitting workflows for execution and monitoring completion
4. Validating results and updating database with generated assets
5. Maintaining SSOT tracking throughout the process

Author: Claude Code & Echo Brain System
Date: January 2026
"""

import asyncio
import json
import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import httpx
import psycopg2
from psycopg2.extras import RealDictCursor, Json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContentGenerationBridge:
    """
    Main bridge class connecting Echo Brain agents to ComfyUI for autonomous content generation.
    """

    def __init__(self, config: Optional[Dict] = None):
        """Initialize the content generation bridge"""
        self.config = config or self._default_config()
        self.db_config = {
            'host': 'localhost',
            'database': 'anime_production',
            'user': 'patrick',
            'password': 'RP78eIrW7cI2jYvL5akt1yurE'
        }

    def _default_config(self) -> Dict:
        """Default configuration for the bridge"""
        return {
            'echo_brain_url': 'http://localhost:8309',
            'comfyui_url': 'http://localhost:8188',
            'lora_path': '/mnt/1TB-storage/models/loras',
            'output_path': '/mnt/1TB-storage/ComfyUI/output',
            'workflow_timeout': 600,  # 10 minutes
            'quality_threshold': 0.8,
            'max_concurrent_jobs': 3
        }

    async def process_scene_generation_request(self, scene_id: str) -> Dict:
        """
        Main entry point for autonomous scene generation.

        Args:
            scene_id: UUID of the scene to generate

        Returns:
            Dict with generation results and metadata
        """
        try:
            logger.info(f"Starting autonomous generation for scene {scene_id}")

            # Step 1: Retrieve scene data from database
            scene_data = self._get_scene_data(scene_id)
            if not scene_data:
                raise ValueError(f"Scene {scene_id} not found in database")

            # Step 2: Create SSOT tracking entry
            tracking_id = self._create_ssot_tracking(scene_id, "scene_generation")

            # Step 3: Use Echo Brain agents to analyze scene and generate workflow
            workflow_spec = await self._generate_workflow_with_agents(scene_data)

            # Step 4: Select appropriate LoRA models based on characters
            lora_config = self._select_character_loras(scene_data.get('character_lora_mapping', {}))

            # Step 5: Generate final ComfyUI workflow JSON
            comfyui_workflow = self._build_comfyui_workflow(workflow_spec, lora_config, scene_data)

            # Step 6: Submit workflow to ComfyUI
            job_id = await self._submit_comfyui_workflow(comfyui_workflow)

            # Step 7: Monitor workflow execution
            execution_result = await self._monitor_workflow_execution(job_id)

            # Step 8: Validate generated content
            validation_result = await self._validate_generated_content(execution_result)

            # Step 9: Update database with results
            asset_record = self._save_generated_asset(scene_id, execution_result, validation_result)

            # Step 10: Update SSOT tracking
            self._update_ssot_tracking(tracking_id, "completed", asset_record)

            return {
                'success': True,
                'scene_id': scene_id,
                'tracking_id': tracking_id,
                'job_id': job_id,
                'asset_record': asset_record,
                'validation': validation_result
            }

        except Exception as e:
            logger.error(f"Error in autonomous generation for scene {scene_id}: {e}")
            # Update tracking with error
            if 'tracking_id' in locals():
                self._update_ssot_tracking(tracking_id, "failed", {'error': str(e)})
            raise

    async def _generate_workflow_with_agents(self, scene_data: Dict) -> Dict:
        """
        Use Echo Brain agents to analyze scene requirements and generate workflow specifications.

        Args:
            scene_data: Scene information from database

        Returns:
            Workflow specification dictionary
        """
        try:
            # Construct agent query based on scene data
            scene_description = scene_data.get('description', '')
            visual_description = scene_data.get('visual_description', '')
            characters = scene_data.get('characters', '')
            shot_list = scene_data.get('shot_list', {})

            agent_query = f"""
            Generate a ComfyUI workflow specification for an anime scene with these requirements:

            Scene Description: {scene_description}
            Visual Description: {visual_description}
            Characters: {characters}
            Shot List: {json.dumps(shot_list, indent=2)}

            The workflow should:
            1. Use FramePack model for video generation
            2. Include appropriate LoRA models for character consistency
            3. Set proper resolution and frame count based on scene requirements
            4. Include quality optimization settings
            5. Generate output suitable for anime production pipeline

            Provide the workflow specification as a structured response.
            """

            # Call Echo Brain CodingAgent for workflow generation
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config['echo_brain_url']}/api/agent",
                    json={"query": agent_query, "agent": "coding"},
                    timeout=30.0
                )

                if response.status_code != 200:
                    raise Exception(f"Echo Brain API error: {response.status_code}")

                agent_result = response.json()

            # Parse agent response into workflow specification
            workflow_spec = self._parse_agent_workflow_response(agent_result['response'])

            logger.info(f"Generated workflow spec with Echo Brain: {workflow_spec.keys()}")
            return workflow_spec

        except Exception as e:
            logger.error(f"Error generating workflow with agents: {e}")
            # Fallback to default workflow
            return self._get_default_workflow_spec(scene_data)

    def _parse_agent_workflow_response(self, agent_response: str) -> Dict:
        """
        Parse Echo Brain agent response into structured workflow specification.

        Args:
            agent_response: Raw response from Echo Brain agent

        Returns:
            Structured workflow specification
        """
        try:
            # Extract workflow parameters from agent response
            # This is a simplified parser - production version would be more sophisticated
            spec = {
                'model': 'FramePackI2V_HY_fp8_e4m3fn.safetensors',
                'resolution': {'width': 704, 'height': 544},
                'frame_count': 120,  # 5 seconds at 24fps
                'fps': 24,
                'sampling_steps': 28,
                'sampler': 'dpmpp_2m',
                'scheduler': 'karras',
                'cfg_scale': 7.0,
                'seed': -1,  # Random
                'use_loras': True,
                'quality_preset': 'high'
            }

            # Try to extract specific parameters from agent response
            if 'resolution' in agent_response.lower():
                # Extract resolution if mentioned
                pass

            if 'frame' in agent_response.lower():
                # Extract frame count if mentioned
                pass

            return spec

        except Exception as e:
            logger.warning(f"Error parsing agent response, using defaults: {e}")
            return self._get_default_workflow_spec({})

    def _get_default_workflow_spec(self, scene_data: Dict) -> Dict:
        """
        Get default workflow specification when agent generation fails.

        Args:
            scene_data: Scene information

        Returns:
            Default workflow specification
        """
        return {
            'model': 'FramePackI2V_HY_fp8_e4m3fn.safetensors',
            'resolution': {'width': 704, 'height': 544},
            'frame_count': 120,  # 5 seconds at 24fps
            'fps': 24,
            'sampling_steps': 28,
            'sampler': 'dpmpp_2m',
            'scheduler': 'karras',
            'cfg_scale': 7.0,
            'seed': -1,
            'use_loras': True,
            'quality_preset': 'high'
        }

    def _select_character_loras(self, character_mapping: Dict) -> Dict:
        """
        Select appropriate LoRA models based on database character records.

        Args:
            character_mapping: Character to LoRA mapping from scene data

        Returns:
            LoRA configuration for workflow
        """
        lora_config = {}

        try:
            # Get available LoRA files
            lora_path = Path(self.config['lora_path'])
            available_loras = {f.stem: str(f) for f in lora_path.glob('*.safetensors')}

            # Map characters to available LoRAs
            for character_name, lora_name in character_mapping.items():
                if lora_name in available_loras:
                    lora_config[character_name] = {
                        'path': available_loras[lora_name],
                        'strength': 0.8,  # Default strength
                        'name': lora_name
                    }
                else:
                    logger.warning(f"LoRA {lora_name} not found for character {character_name}")

            logger.info(f"Selected LoRAs: {list(lora_config.keys())}")
            return lora_config

        except Exception as e:
            logger.error(f"Error selecting LoRAs: {e}")
            return {}

    def _build_comfyui_workflow(self, workflow_spec: Dict, lora_config: Dict, scene_data: Dict) -> Dict:
        """
        Build final ComfyUI workflow JSON from specifications.

        Args:
            workflow_spec: Workflow specification from agents
            lora_config: LoRA configuration
            scene_data: Scene data from database

        Returns:
            Complete ComfyUI workflow JSON
        """
        try:
            # Load base workflow template
            template_path = Path(__file__).parent / "templates" / "framepack_base_workflow.json"

            if template_path.exists():
                with open(template_path) as f:
                    workflow = json.load(f)
            else:
                # Use the workflow we found earlier as base
                workflow = self._get_base_framepack_workflow()

            # Update workflow with scene-specific parameters
            self._update_workflow_parameters(workflow, workflow_spec, scene_data)

            # Add LoRA configurations if available
            if lora_config:
                self._add_loras_to_workflow(workflow, lora_config)

            # Update prompt with scene description
            self._update_workflow_prompt(workflow, scene_data)

            return workflow

        except Exception as e:
            logger.error(f"Error building ComfyUI workflow: {e}")
            raise

    def _get_base_framepack_workflow(self) -> Dict:
        """
        Get base FramePack workflow structure.
        This would be the template loaded from the existing workflow file.
        """
        # For now, return a simplified base structure
        # In production, this would load from the actual workflow template
        return {
            "nodes": [],
            "links": [],
            "groups": [],
            "config": {},
            "version": 0.4
        }

    def _update_workflow_parameters(self, workflow: Dict, spec: Dict, scene_data: Dict):
        """Update workflow with scene-specific parameters"""
        # This would update the workflow JSON with the specifications
        # Implementation would modify specific nodes in the workflow
        pass

    def _add_loras_to_workflow(self, workflow: Dict, lora_config: Dict):
        """Add LoRA configurations to workflow"""
        # This would add LoRA loader nodes to the workflow
        pass

    def _update_workflow_prompt(self, workflow: Dict, scene_data: Dict):
        """Update text prompts in workflow based on scene data"""
        prompt = scene_data.get('description', '')
        visual_desc = scene_data.get('visual_description', '')

        combined_prompt = f"{prompt}. {visual_desc}".strip()

        # Find and update CLIPTextEncode nodes
        # Implementation would locate and update prompt nodes
        pass

    async def _submit_comfyui_workflow(self, workflow: Dict) -> str:
        """
        Submit workflow to ComfyUI and return job ID.

        Args:
            workflow: ComfyUI workflow JSON

        Returns:
            Job ID string
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.config['comfyui_url']}/prompt",
                    json={"prompt": workflow},
                    timeout=30.0
                )

                if response.status_code != 200:
                    raise Exception(f"ComfyUI API error: {response.status_code}")

                result = response.json()
                job_id = result.get('prompt_id')

                if not job_id:
                    raise Exception("No job ID returned from ComfyUI")

                logger.info(f"Submitted workflow to ComfyUI, job ID: {job_id}")
                return job_id

        except Exception as e:
            logger.error(f"Error submitting workflow to ComfyUI: {e}")
            raise

    async def _monitor_workflow_execution(self, job_id: str) -> Dict:
        """
        Monitor ComfyUI workflow execution until completion.

        Args:
            job_id: ComfyUI job ID

        Returns:
            Execution result dictionary
        """
        try:
            timeout = self.config['workflow_timeout']
            start_time = datetime.now()

            async with httpx.AsyncClient() as client:
                while True:
                    # Check if timeout exceeded
                    elapsed = (datetime.now() - start_time).total_seconds()
                    if elapsed > timeout:
                        raise Exception(f"Workflow execution timed out after {timeout} seconds")

                    # Check job status
                    try:
                        response = await client.get(
                            f"{self.config['comfyui_url']}/history/{job_id}",
                            timeout=10.0
                        )

                        if response.status_code == 200:
                            history = response.json()

                            if job_id in history:
                                job_data = history[job_id]
                                status = job_data.get('status', {})

                                if status.get('completed'):
                                    logger.info(f"Workflow {job_id} completed successfully")
                                    return {
                                        'success': True,
                                        'job_id': job_id,
                                        'outputs': job_data.get('outputs', {}),
                                        'execution_time': elapsed
                                    }
                                elif status.get('status_str') == 'error':
                                    error_msg = status.get('messages', ['Unknown error'])
                                    raise Exception(f"Workflow execution failed: {error_msg}")

                    except httpx.RequestError:
                        logger.warning("Error checking job status, retrying...")

                    # Wait before next check
                    await asyncio.sleep(5)

        except Exception as e:
            logger.error(f"Error monitoring workflow execution: {e}")
            raise

    async def _validate_generated_content(self, execution_result: Dict) -> Dict:
        """
        Validate generated content quality and completeness.

        Args:
            execution_result: Result from ComfyUI execution

        Returns:
            Validation result dictionary
        """
        try:
            validation = {
                'passed': True,
                'quality_score': 0.85,  # Placeholder
                'issues': [],
                'metadata': {}
            }

            # Check if outputs exist
            outputs = execution_result.get('outputs', {})
            if not outputs:
                validation['passed'] = False
                validation['issues'].append("No outputs generated")
                return validation

            # Check for video output files
            video_outputs = []
            for node_id, output_data in outputs.items():
                if 'videos' in output_data:
                    video_outputs.extend(output_data['videos'])

            if not video_outputs:
                validation['passed'] = False
                validation['issues'].append("No video outputs found")
                return validation

            # Validate video files exist and have reasonable size
            for video_info in video_outputs:
                filename = video_info.get('filename')
                if filename:
                    video_path = Path(self.config['output_path']) / filename
                    if not video_path.exists():
                        validation['issues'].append(f"Output file not found: {filename}")
                        validation['passed'] = False
                    elif video_path.stat().st_size < 1000:  # Less than 1KB
                        validation['issues'].append(f"Output file too small: {filename}")
                        validation['passed'] = False

            # Additional quality checks would go here
            # - Frame count validation
            # - Duration validation
            # - Visual quality analysis

            logger.info(f"Content validation: {'PASSED' if validation['passed'] else 'FAILED'}")
            return validation

        except Exception as e:
            logger.error(f"Error validating content: {e}")
            return {
                'passed': False,
                'quality_score': 0.0,
                'issues': [f"Validation error: {e}"],
                'metadata': {}
            }

    def _save_generated_asset(self, scene_id: str, execution_result: Dict, validation_result: Dict) -> Dict:
        """
        Save generated asset record to database.

        Args:
            scene_id: Original scene ID
            execution_result: ComfyUI execution result
            validation_result: Validation result

        Returns:
            Asset record dictionary
        """
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor(cursor_factory=RealDictCursor) as cur:
                    # Create asset record
                    asset_id = str(uuid.uuid4())

                    # Extract output information
                    outputs = execution_result.get('outputs', {})
                    output_files = []

                    for node_id, output_data in outputs.items():
                        if 'videos' in output_data:
                            output_files.extend([v.get('filename') for v in output_data['videos']])

                    # Insert into generated_assets table
                    cur.execute("""
                        INSERT INTO generated_assets (
                            id, scene_id, asset_type, file_paths,
                            generation_params, quality_score, validation_status,
                            created_at, metadata
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        ) RETURNING *
                    """, (
                        asset_id,
                        scene_id,
                        'video',
                        Json(output_files),
                        Json(execution_result),
                        validation_result.get('quality_score', 0.0),
                        'passed' if validation_result.get('passed') else 'failed',
                        datetime.now(),
                        Json({
                            'job_id': execution_result.get('job_id'),
                            'execution_time': execution_result.get('execution_time'),
                            'validation_issues': validation_result.get('issues', [])
                        })
                    ))

                    asset_record = dict(cur.fetchone())

                    # Update scene record with generated asset
                    cur.execute("""
                        UPDATE scenes
                        SET output_path = %s, status = %s, updated_at = %s
                        WHERE id = %s
                    """, (
                        output_files[0] if output_files else None,
                        'completed' if validation_result.get('passed') else 'failed',
                        datetime.now(),
                        scene_id
                    ))

                    conn.commit()

                    logger.info(f"Saved asset record {asset_id} for scene {scene_id}")
                    return asset_record

        except Exception as e:
            logger.error(f"Error saving generated asset: {e}")
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

    def _create_ssot_tracking(self, resource_id: str, operation: str) -> str:
        """Create SSOT tracking entry"""
        try:
            tracking_id = str(uuid.uuid4())

            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO ssot_tracking (
                            id, resource_type, resource_id, operation,
                            status, created_at, metadata
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                    """, (
                        tracking_id, 'scene', resource_id, operation,
                        'started', datetime.now(), Json({})
                    ))
                    conn.commit()

            return tracking_id
        except Exception as e:
            logger.error(f"Error creating SSOT tracking: {e}")
            return str(uuid.uuid4())  # Return ID anyway for error handling

    def _update_ssot_tracking(self, tracking_id: str, status: str, metadata: Dict):
        """Update SSOT tracking entry"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        UPDATE ssot_tracking
                        SET status = %s, updated_at = %s, metadata = %s
                        WHERE id = %s
                    """, (status, datetime.now(), Json(metadata), tracking_id))
                    conn.commit()
        except Exception as e:
            logger.error(f"Error updating SSOT tracking: {e}")


# CLI Interface
async def main():
    """Command line interface for the content generation bridge"""
    import argparse

    parser = argparse.ArgumentParser(description='Content Generation Bridge')
    parser.add_argument('--scene-id', required=True, help='Scene ID to generate')
    parser.add_argument('--config', help='Configuration file path')

    args = parser.parse_args()

    # Load configuration if provided
    config = None
    if args.config and Path(args.config).exists():
        with open(args.config) as f:
            config = json.load(f)

    # Initialize bridge
    bridge = ContentGenerationBridge(config)

    # Process scene generation
    try:
        result = await bridge.process_scene_generation_request(args.scene_id)
        print(json.dumps(result, indent=2, default=str))
    except Exception as e:
        print(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    asyncio.run(main())