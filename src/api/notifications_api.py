#!/usr/bin/env python3
"""
Notifications API endpoints for Echo Brain
Provides unified notification management across multiple channels
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Dict, Any, Optional, List
from datetime import datetime
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Import notification service
try:
    from src.services.notification_service import (
        get_notification_service,
        NotificationType,
        NotificationChannel,
        notify_info,
        notify_warning,
        notify_error,
        notify_success
    )
    NOTIFICATIONS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Notification service not available: {e}")
    NOTIFICATIONS_AVAILABLE = False

router = APIRouter(prefix="/api/notifications", tags=["notifications"])

class NotificationRequest(BaseModel):
    """Notification request model"""
    message: str
    title: Optional[str] = None
    type: str = "info"  # info, warning, error, success, alert, reminder
    channels: Optional[List[str]] = None  # ntfy, telegram, email, or null for all
    priority: Optional[int] = None  # 1-5 for ntfy
    schedule: Optional[str] = None  # ISO datetime string for scheduling
    metadata: Optional[Dict[str, Any]] = None

class BulkNotificationRequest(BaseModel):
    """Bulk notification request"""
    notifications: List[NotificationRequest]
    delay_seconds: Optional[float] = 1.0  # Delay between notifications

@router.get("/status")
async def get_notification_status():
    """Get notification service status and available channels"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        service = await get_notification_service()
        if not service:
            return {
                "available": False,
                "message": "Notification service not initialized"
            }

        status = service.get_status()
        return {
            "available": True,
            "notification_service": status
        }
    except Exception as e:
        logger.error(f"Error getting notification status: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get notification status: {str(e)}")

@router.post("/send")
async def send_notification(notification: NotificationRequest, background_tasks: BackgroundTasks):
    """Send a notification"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        service = await get_notification_service()
        if not service:
            raise HTTPException(status_code=503, detail="Notification service not initialized")

        # Parse notification type
        try:
            ntype = NotificationType(notification.type.lower())
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid notification type: {notification.type}")

        # Parse channels
        channels = NotificationChannel.ALL
        if notification.channels:
            try:
                channel_list = [NotificationChannel(ch.lower()) for ch in notification.channels]
                channels = channel_list
            except ValueError as e:
                raise HTTPException(status_code=400, detail=f"Invalid channel: {str(e)}")

        # Parse schedule if provided
        schedule = None
        if notification.schedule:
            try:
                schedule = datetime.fromisoformat(notification.schedule.replace('Z', '+00:00'))
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid schedule format. Use ISO 8601 format.")

        # Send notification
        results = await service.send_notification(
            message=notification.message,
            title=notification.title,
            notification_type=ntype,
            channels=channels,
            priority=notification.priority,
            schedule=schedule,
            metadata=notification.metadata
        )

        return {
            "success": True,
            "results": results,
            "message": "Notification sent",
            "channels_successful": sum(results.values()),
            "channels_attempted": len(results)
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error sending notification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send notification: {str(e)}")

@router.post("/send/bulk")
async def send_bulk_notifications(bulk_request: BulkNotificationRequest, background_tasks: BackgroundTasks):
    """Send multiple notifications with optional delay"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        service = await get_notification_service()
        if not service:
            raise HTTPException(status_code=503, detail="Notification service not initialized")

        # Add bulk sending to background tasks
        background_tasks.add_task(_send_bulk_notifications, service, bulk_request)

        return {
            "success": True,
            "message": f"Queued {len(bulk_request.notifications)} notifications for sending",
            "count": len(bulk_request.notifications),
            "delay_seconds": bulk_request.delay_seconds
        }

    except Exception as e:
        logger.error(f"Error queuing bulk notifications: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue bulk notifications: {str(e)}")

@router.post("/send/info")
async def send_info_notification(message: str, title: str = "Information"):
    """Send info notification - convenience endpoint"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        success = await notify_info(message, title)
        return {"success": success, "type": "info", "message": message, "title": title}
    except Exception as e:
        logger.error(f"Error sending info notification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send info notification: {str(e)}")

@router.post("/send/warning")
async def send_warning_notification(message: str, title: str = "Warning"):
    """Send warning notification - convenience endpoint"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        success = await notify_warning(message, title)
        return {"success": success, "type": "warning", "message": message, "title": title}
    except Exception as e:
        logger.error(f"Error sending warning notification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send warning notification: {str(e)}")

@router.post("/send/error")
async def send_error_notification(message: str, title: str = "Error"):
    """Send error notification - convenience endpoint"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        success = await notify_error(message, title)
        return {"success": success, "type": "error", "message": message, "title": title}
    except Exception as e:
        logger.error(f"Error sending error notification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send error notification: {str(e)}")

@router.post("/send/success")
async def send_success_notification(message: str, title: str = "Success"):
    """Send success notification - convenience endpoint"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        success = await notify_success(message, title)
        return {"success": success, "type": "success", "message": message, "title": title}
    except Exception as e:
        logger.error(f"Error sending success notification: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send success notification: {str(e)}")

@router.post("/test")
async def test_notifications():
    """Test all notification channels"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        service = await get_notification_service()
        if not service:
            raise HTTPException(status_code=503, detail="Notification service not initialized")

        # Send test notification to each type
        test_message = f"Test notification from Echo Brain API at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"

        results = await service.send_notification(
            message=test_message,
            title="Echo Brain API Test",
            notification_type=NotificationType.INFO,
            channels=NotificationChannel.ALL
        )

        return {
            "success": any(results.values()),
            "test_message": test_message,
            "results": results,
            "channels_successful": sum(results.values()),
            "channels_attempted": len(results)
        }

    except Exception as e:
        logger.error(f"Error testing notifications: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to test notifications: {str(e)}")

@router.get("/channels")
async def get_available_channels():
    """Get list of available notification channels"""
    if not NOTIFICATIONS_AVAILABLE:
        raise HTTPException(status_code=503, detail="Notification service not available")

    try:
        service = await get_notification_service()
        if not service:
            return {"available": False, "channels": []}

        status = service.get_status()
        available_channels = [
            channel for channel, available in status["available_channels"].items()
            if available
        ]

        return {
            "available": True,
            "channels": available_channels,
            "total": len(available_channels),
            "all_channels": list(status["available_channels"].keys())
        }

    except Exception as e:
        logger.error(f"Error getting channels: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get channels: {str(e)}")

# Background task for bulk notifications
async def _send_bulk_notifications(service, bulk_request: BulkNotificationRequest):
    """Background task to send bulk notifications"""
    import asyncio

    logger.info(f"📢 Sending {len(bulk_request.notifications)} bulk notifications")

    for i, notification in enumerate(bulk_request.notifications):
        try:
            # Parse notification type
            ntype = NotificationType(notification.type.lower())

            # Parse channels
            channels = NotificationChannel.ALL
            if notification.channels:
                channel_list = [NotificationChannel(ch.lower()) for ch in notification.channels]
                channels = channel_list

            # Parse schedule
            schedule = None
            if notification.schedule:
                schedule = datetime.fromisoformat(notification.schedule.replace('Z', '+00:00'))

            # Send notification
            await service.send_notification(
                message=notification.message,
                title=notification.title,
                notification_type=ntype,
                channels=channels,
                priority=notification.priority,
                schedule=schedule,
                metadata=notification.metadata
            )

            logger.info(f"📢 Bulk notification {i+1}/{len(bulk_request.notifications)} sent")

            # Rate limiting delay
            if i < len(bulk_request.notifications) - 1:  # Don't delay after last notification
                await asyncio.sleep(bulk_request.delay_seconds)

        except Exception as e:
            logger.error(f"❌ Failed to send bulk notification {i+1}: {e}")

    logger.info("📢 Bulk notification sending completed")