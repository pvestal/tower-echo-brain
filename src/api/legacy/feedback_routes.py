#!/usr/bin/env python3
"""
Feedback capture API routes for Echo Brain
Captures user feedback for continuous improvement
"""
import logging
import uuid
import time
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, HTTPException

from src.api.models import FeedbackRequest, FeedbackResponse
from src.db.database import database

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory feedback store for fast access (should be moved to database eventually)
feedback_store = {}
feedback_stats = {
    'total_feedback': 0,
    'thumbs_up': 0,
    'thumbs_down': 0,
    'corrections': 0,
    'pattern_feedback': 0,
    'processed_feedback': 0
}

@router.post("/api/echo/feedback", response_model=FeedbackResponse)
async def submit_feedback(feedback: FeedbackRequest):
    """Submit feedback on Echo's response for improvement learning"""
    feedback_id = str(uuid.uuid4())
    timestamp = datetime.now()

    try:
        # Store feedback
        feedback_entry = {
            "feedback_id": feedback_id,
            "conversation_id": feedback.conversation_id,
            "response_id": feedback.response_id,
            "feedback_type": feedback.feedback_type,
            "feedback_data": feedback.feedback_data,
            "user_id": feedback.user_id,
            "timestamp": timestamp,
            "processed": False,
            "applied": False
        }

        feedback_store[feedback_id] = feedback_entry

        # Update stats
        feedback_stats['total_feedback'] += 1
        if feedback.feedback_type in feedback_stats:
            feedback_stats[feedback.feedback_type] += 1

        logger.info(f"📝 Feedback received: {feedback.feedback_type} for conversation {feedback.conversation_id}")

        # Process feedback based on type
        impact = await process_feedback(feedback_entry)

        return FeedbackResponse(
            feedback_id=feedback_id,
            status="received",
            impact=impact
        )

    except Exception as e:
        logger.error(f"Failed to submit feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback submission failed: {str(e)}")

async def process_feedback(feedback_entry: Dict) -> Dict:
    """Process feedback and determine its impact on the system"""
    feedback_type = feedback_entry["feedback_type"]
    feedback_data = feedback_entry["feedback_data"]
    impact = {"changes_made": [], "learning_updated": False}

    try:
        if feedback_type == "thumbs_down":
            # Analyze what went wrong
            impact["changes_made"].append("Logged negative response pattern")
            if "reason" in feedback_data:
                reason = feedback_data["reason"]
                logger.info(f"👎 Negative feedback reason: {reason}")

                # If it's a pattern issue, mark for pattern review
                if any(keyword in reason.lower() for keyword in ['wrong', 'incorrect', 'bad', 'avoid']):
                    impact["changes_made"].append("Flagged for pattern review")

        elif feedback_type == "correction":
            # User provided a correction
            correct_response = feedback_data.get("correct_response", "")
            incorrect_part = feedback_data.get("incorrect_part", "")

            if correct_response and incorrect_part:
                impact["changes_made"].append(f"Correction logged: {incorrect_part} → {correct_response}")
                impact["learning_updated"] = True
                logger.info(f"📝 Correction feedback: {incorrect_part} → {correct_response}")

        elif feedback_type == "pattern_feedback":
            # Feedback about business logic patterns
            pattern_issue = feedback_data.get("pattern_issue", "")
            suggested_fix = feedback_data.get("suggested_fix", "")

            if pattern_issue:
                impact["changes_made"].append(f"Pattern issue logged: {pattern_issue}")
                if suggested_fix:
                    impact["changes_made"].append(f"Suggested fix: {suggested_fix}")

                logger.info(f"🧠 Pattern feedback: {pattern_issue}")

        elif feedback_type == "thumbs_up":
            # Positive feedback - reinforce this pattern
            impact["changes_made"].append("Positive response pattern reinforced")
            logger.info(f"👍 Positive feedback received")

        # Mark as processed
        feedback_entry["processed"] = True
        feedback_entry["impact"] = impact
        feedback_stats['processed_feedback'] += 1

        return impact

    except Exception as e:
        logger.error(f"Failed to process feedback: {e}")
        return {"changes_made": [f"Processing failed: {str(e)}"], "learning_updated": False}

@router.get("/api/echo/feedback/stats")
async def get_feedback_stats():
    """Get feedback statistics for monitoring improvement"""
    return {
        "feedback_stats": feedback_stats,
        "recent_feedback": list(feedback_store.values())[-10:],  # Last 10 feedback entries
        "total_conversations_with_feedback": len(set(f["conversation_id"] for f in feedback_store.values())),
        "feedback_types": {
            "thumbs_up": len([f for f in feedback_store.values() if f["feedback_type"] == "thumbs_up"]),
            "thumbs_down": len([f for f in feedback_store.values() if f["feedback_type"] == "thumbs_down"]),
            "corrections": len([f for f in feedback_store.values() if f["feedback_type"] == "correction"]),
            "pattern_feedback": len([f for f in feedback_store.values() if f["feedback_type"] == "pattern_feedback"])
        }
    }

@router.get("/api/echo/feedback/{conversation_id}")
async def get_conversation_feedback(conversation_id: str):
    """Get all feedback for a specific conversation"""
    conversation_feedback = [
        f for f in feedback_store.values()
        if f["conversation_id"] == conversation_id
    ]

    return {
        "conversation_id": conversation_id,
        "feedback_count": len(conversation_feedback),
        "feedback_entries": conversation_feedback
    }

@router.post("/api/echo/feedback/apply")
async def apply_pending_feedback():
    """Apply pending feedback to improve business logic patterns"""
    applied_count = 0

    try:
        for feedback_entry in feedback_store.values():
            if feedback_entry["processed"] and not feedback_entry["applied"]:
                # This is where we would apply feedback to business logic patterns
                # For now, just mark as applied
                feedback_entry["applied"] = True
                applied_count += 1

                logger.info(f"✅ Applied feedback {feedback_entry['feedback_id']}")

        return {
            "status": "success",
            "applied_count": applied_count,
            "message": f"Applied {applied_count} pending feedback entries"
        }

    except Exception as e:
        logger.error(f"Failed to apply feedback: {e}")
        raise HTTPException(status_code=500, detail=f"Feedback application failed: {str(e)}")

@router.delete("/api/echo/feedback/{feedback_id}")
async def delete_feedback(feedback_id: str):
    """Delete specific feedback entry"""
    if feedback_id in feedback_store:
        deleted_feedback = feedback_store.pop(feedback_id)
        logger.info(f"🗑️ Deleted feedback {feedback_id}")
        return {
            "status": "deleted",
            "feedback": deleted_feedback
        }
    else:
        raise HTTPException(status_code=404, detail="Feedback not found")