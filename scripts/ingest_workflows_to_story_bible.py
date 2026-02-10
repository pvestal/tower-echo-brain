#!/usr/bin/env python3
"""
Workflow Ingestion Script for Story Bible
Embeds ComfyUI workflows from anime production into story_bible collection
"""

import os
import json
import requests
import logging
from typing import List, Dict, Any
from datetime import datetime
from pathlib import Path

# Configuration
QDRANT_URL = "http://localhost:6333"
OLLAMA_URL = "http://localhost:11434"
COLLECTION_NAME = "story_bible"
WORKFLOW_DIRS = [
    "/opt/tower-anime-production/workflows/comfyui",
    "/mnt/1TB-storage/workflows"
]

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def get_embedding(text: str) -> List[float]:
    """Get embedding from Ollama using nomic-embed-text"""
    try:
        response = requests.post(
            f"{OLLAMA_URL}/api/embeddings",
            json={
                "model": "nomic-embed-text",
                "prompt": text
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["embedding"]
    except Exception as e:
        logger.error(f"Failed to get embedding: {e}")
        raise

def extract_workflow_info(workflow_data: Dict) -> Dict[str, Any]:
    """Extract meaningful information from ComfyUI workflow JSON"""
    info = {
        "prompts": [],
        "negative_prompts": [],
        "models": [],
        "loras": [],
        "settings": {},
        "node_types": set(),
        "purpose": "unknown"
    }

    # Walk through all nodes
    for node_id, node_data in workflow_data.items():
        if not isinstance(node_data, dict) or "class_type" not in node_data:
            continue

        class_type = node_data["class_type"]
        info["node_types"].add(class_type)
        inputs = node_data.get("inputs", {})

        # Extract prompts
        if class_type == "CLIPTextEncode":
            text = inputs.get("text", "")
            if text:
                title = node_data.get("_meta", {}).get("title", "")
                if "negative" in title.lower():
                    info["negative_prompts"].append(text)
                else:
                    info["prompts"].append(text)

        # Extract model info
        elif class_type in ["CheckpointLoaderSimple", "CheckpointLoader"]:
            model_name = inputs.get("ckpt_name", "")
            if model_name:
                info["models"].append(model_name)

        # Extract LoRA info
        elif class_type == "LoraLoader":
            lora_name = inputs.get("lora_name", "")
            lora_strength = inputs.get("strength_model", 1.0)
            if lora_name:
                info["loras"].append({"name": lora_name, "strength": lora_strength})

        # Extract sampling settings
        elif class_type == "KSampler":
            info["settings"]["sampler"] = {
                "steps": inputs.get("steps", 20),
                "cfg": inputs.get("cfg", 7.0),
                "sampler_name": inputs.get("sampler_name", "euler"),
                "scheduler": inputs.get("scheduler", "normal"),
                "seed": inputs.get("seed", -1)
            }

        # Extract video settings
        elif "Video" in class_type or "SVD" in class_type:
            info["settings"]["video"] = {
                "class_type": class_type,
                "inputs": inputs
            }

    # Determine purpose from filename and content
    return info

def determine_workflow_purpose(filename: str, workflow_info: Dict) -> str:
    """Determine workflow purpose from filename and content"""
    filename_lower = filename.lower()

    if "action" in filename_lower or "combat" in filename_lower:
        return "action_scene"
    elif "30sec" in filename_lower or "video" in filename_lower:
        return "video_generation"
    elif "rife" in filename_lower:
        return "frame_interpolation"
    elif "lora" in filename_lower:
        return "character_specific"
    elif "generic" in filename_lower:
        return "general_purpose"
    elif "test" in filename_lower:
        return "testing"
    elif "fixed" in filename_lower:
        return "stable_workflow"

    # Check content
    node_types = workflow_info.get("node_types", set())
    if any("Video" in nt or "SVD" in nt for nt in node_types):
        return "video_generation"
    elif any("LoRA" in nt for nt in node_types):
        return "lora_enhanced"
    else:
        return "image_generation"

def format_workflow_document(filename: str, workflow_data: Dict, workflow_info: Dict) -> str:
    """Format workflow data into a comprehensive document"""
    purpose = determine_workflow_purpose(filename, workflow_info)

    doc = f"COMFYUI WORKFLOW: {filename}\n\n"
    doc += f"Purpose: {purpose}\n"
    doc += f"Type: ComfyUI workflow for anime production\n\n"

    # Prompts
    if workflow_info["prompts"]:
        doc += "Positive Prompts:\n"
        for prompt in workflow_info["prompts"]:
            doc += f"  - {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n"
        doc += "\n"

    if workflow_info["negative_prompts"]:
        doc += "Negative Prompts:\n"
        for prompt in workflow_info["negative_prompts"]:
            doc += f"  - {prompt[:200]}{'...' if len(prompt) > 200 else ''}\n"
        doc += "\n"

    # Models and LoRAs
    if workflow_info["models"]:
        doc += f"Models: {', '.join(workflow_info['models'])}\n\n"

    if workflow_info["loras"]:
        doc += "LoRAs:\n"
        for lora in workflow_info["loras"]:
            doc += f"  - {lora['name']} (strength: {lora['strength']})\n"
        doc += "\n"

    # Settings
    if workflow_info["settings"].get("sampler"):
        sampler = workflow_info["settings"]["sampler"]
        doc += f"Sampling: {sampler['steps']} steps, CFG {sampler['cfg']}, {sampler['sampler_name']}\n\n"

    # Node types (technical info)
    if workflow_info["node_types"]:
        node_list = sorted(list(workflow_info["node_types"]))
        doc += f"Node Types: {', '.join(node_list[:10])}"
        if len(node_list) > 10:
            doc += f" + {len(node_list) - 10} more"
        doc += "\n\n"

    # File info
    doc += f"Location: {filename}\n"
    doc += f"Total Nodes: {len(workflow_data)}\n"

    return doc.strip()

def ingest_workflows():
    """Main workflow ingestion function"""
    logger.info("Starting workflow ingestion...")

    # Get current point count
    try:
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        current_count = response.json()["result"]["points_count"]
        logger.info(f"Current story_bible collection has {current_count} points")
    except Exception as e:
        logger.error(f"Failed to get current collection count: {e}")
        return

    points = []
    point_id = current_count  # Start from current count

    # Process all workflow directories
    for workflow_dir in WORKFLOW_DIRS:
        if not os.path.exists(workflow_dir):
            logger.info(f"Directory {workflow_dir} not found, skipping")
            continue

        logger.info(f"Processing workflows in {workflow_dir}")

        for filepath in Path(workflow_dir).glob("*.json"):
            try:
                logger.info(f"Processing {filepath.name}")

                # Load workflow
                with open(filepath, 'r', encoding='utf-8') as f:
                    workflow_data = json.load(f)

                # Extract workflow information
                workflow_info = extract_workflow_info(workflow_data)

                # Format document
                doc_text = format_workflow_document(filepath.name, workflow_data, workflow_info)

                # Skip if document is too short
                if len(doc_text.strip()) < 50:
                    logger.warning(f"Skipping {filepath.name}: document too short")
                    continue

                # Get embedding
                embedding = get_embedding(doc_text)

                # Determine workflow category
                purpose = determine_workflow_purpose(filepath.name, workflow_info)

                # Create point
                points.append({
                    "id": point_id,
                    "vector": embedding,
                    "payload": {
                        "content": doc_text,
                        "type": "workflow",
                        "name": filepath.name,
                        "purpose": purpose,
                        "models": workflow_info["models"],
                        "loras": [lora["name"] for lora in workflow_info["loras"]],
                        "node_count": len(workflow_data),
                        "file_path": str(filepath),
                        "ingested_at": datetime.now().isoformat()
                    }
                })
                point_id += 1

            except Exception as e:
                logger.error(f"Failed to process {filepath}: {e}")
                continue

    # Upload to Qdrant
    if points:
        logger.info(f"Uploading {len(points)} workflow points to Qdrant...")

        # Upload in batches
        batch_size = 50
        for i in range(0, len(points), batch_size):
            batch = points[i:i+batch_size]

            response = requests.put(
                f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points",
                json={"points": batch}
            )
            response.raise_for_status()
            logger.info(f"Uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")

        logger.info(f"Successfully ingested {len(points)} workflows into story_bible collection")
    else:
        logger.warning("No workflow files found to ingest")

def verify_ingestion():
    """Verify workflows were ingested successfully"""
    try:
        # Check collection size
        response = requests.get(f"{QDRANT_URL}/collections/{COLLECTION_NAME}")
        response.raise_for_status()
        info = response.json()["result"]

        logger.info(f"Collection after ingestion:")
        logger.info(f"  Total points: {info['points_count']}")

        # Search for workflow content
        test_response = requests.post(
            f"{QDRANT_URL}/collections/{COLLECTION_NAME}/points/search",
            json={
                "vector": [0.1] * 768,
                "limit": 5,
                "with_payload": True,
                "filter": {
                    "must": [{"key": "type", "match": {"value": "workflow"}}]
                }
            }
        )
        test_response.raise_for_status()
        workflows = test_response.json()["result"]

        logger.info(f"Found {len(workflows)} workflow documents:")
        for workflow in workflows:
            payload = workflow["payload"]
            logger.info(f"  - {payload['name']} ({payload['purpose']}) - {payload['node_count']} nodes")

    except Exception as e:
        logger.error(f"Failed to verify ingestion: {e}")

if __name__ == "__main__":
    try:
        ingest_workflows()
        verify_ingestion()
        logger.info("Workflow ingestion completed successfully!")
    except Exception as e:
        logger.error(f"Workflow ingestion failed: {e}")
        raise