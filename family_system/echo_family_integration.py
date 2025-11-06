#!/usr/bin/env python3
"""
AI Assist Family Integration with Tower Services
Multi-user, multi-device family ecosystem
"""

import os
from typing import Dict, List, Optional
from datetime import datetime
import json

class TowerFamilyIntegration:
    """
    Integrates AI Assist with Tower services for family use
    """

    def __init__(self):
        self.services = {
            "echo_brain": {
                "port": 8309,
                "description": "Main AI assistant",
                "family_features": ["chat", "learning", "help"]
            },
            "knowledge_base": {
                "port": 8307,
                "description": "Family knowledge & recipes",
                "family_features": ["recipes", "how-to", "family_history"]
            },
            "apple_music": {
                "port": 8315,
                "description": "Family music library",
                "family_features": ["playlists", "kids_music", "audiobooks"]
            },
            "calendar": {
                "port": 8320,  # Future service
                "description": "Family calendar",
                "family_features": ["events", "reminders", "birthdays"]
            },
            "home_automation": {
                "port": 8321,  # Future service
                "description": "Smart home control",
                "family_features": ["lights", "temperature", "security"]
            },
            "jellyfin": {
                "port": 8096,
                "description": "Family media server",
                "family_features": ["movies", "shows", "photos"]
            }
        }

        self.device_registry = {}
        self.family_spaces = {}
        self._initialize_family_spaces()

    def _initialize_family_spaces(self):
        """Create family-specific spaces and resources"""

        # Shared family spaces
        self.family_spaces["shared"] = {
            "calendar": {},
            "shopping_list": [],
            "recipes": [],
            "photos": [],
            "notes": []
        }

        # Private spaces per user
        self.family_spaces[os.getenv("TOWER_USER", "patrick")] = {
            "projects": [],
            "work_notes": [],
            "api_keys": {},  # Admin only
            "system_config": {}
        }

        self.family_spaces["partner"] = {
            "personal_notes": [],
            "recipes": [],
            "shopping_lists": []
        }

        # Kids spaces (if applicable)
        self.family_spaces["kids"] = {
            "homework": [],
            "games": [],
            "learning": [],
            "screen_time": {"limit_minutes": 120}
        }

    def register_device(self, device_id: str, device_info: Dict) -> bool:
        """Register a family device (phone, tablet, computer, etc.)"""

        self.device_registry[device_id] = {
            "name": device_info.get("name"),
            "type": device_info.get("type"),  # phone, tablet, laptop, desktop, tv
            "primary_user": device_info.get("primary_user"),
            "allowed_users": device_info.get("allowed_users", []),
            "restrictions": device_info.get("restrictions", []),
            "registered_at": datetime.now().isoformat()
        }

        return True

    def get_device_permissions(self, device_id: str, user_id: str) -> Dict:
        """Get what a user can do on a specific device"""

        device = self.device_registry.get(device_id)
        if not device:
            return {"error": "Unknown device"}

        # Check if user is allowed on this device
        if user_id not in device["allowed_users"] and user_id != "patrick":
            return {"error": "User not authorized for this device"}

        permissions = {
            "can_access_echo": True,
            "can_modify_settings": user_id == device["primary_user"] or user_id == os.getenv("TOWER_USER", "patrick"),
            "can_install_apps": user_id == os.getenv("TOWER_USER", "patrick"),
            "can_access_media": True,
            "restrictions": device.get("restrictions", [])
        }

        return permissions

    def handle_family_query(self, user_id: str, query: str, device_id: str) -> Dict:
        """
        Handle family-context queries with device awareness
        """

        # Device context
        device = self.device_registry.get(device_id, {})
        location = device.get("location", "unknown")

        # Parse query for family context
        query_lower = query.lower()

        # Shopping list queries
        if any(word in query_lower for word in ["shopping", "grocery", "buy"]):
            return self._handle_shopping_query(user_id, query)

        # Calendar queries
        elif any(word in query_lower for word in ["calendar", "event", "appointment"]):
            return self._handle_calendar_query(user_id, query)

        # Media queries
        elif any(word in query_lower for word in ["watch", "movie", "show"]):
            return self._handle_media_query(user_id, query, device_id)

        # Home automation
        elif any(word in query_lower for word in ["lights", "temperature", "door"]):
            return self._handle_home_automation(user_id, query, device_id)

        # Kids queries
        elif user_id in ["child1", "child2"]:
            return self._handle_kids_query(user_id, query)

        # Admin queries
        elif user_id == "patrick" and "system" in query_lower:
            return self._handle_admin_query(query)

        # Default to AI Assist
        return {"service": "echo_brain", "action": "process", "query": query}

    def _handle_shopping_query(self, user_id: str, query: str) -> Dict:
        """Handle shopping list operations"""

        if "add" in query.lower():
            # Extract item to add
            item = query.lower().replace("add", "").replace("to shopping list", "").strip()
            self.family_spaces["shared"]["shopping_list"].append({
                "item": item,
                "added_by": user_id,
                "added_at": datetime.now().isoformat()
            })
            return {
                "response": f"Added '{item}' to the family shopping list",
                "list": self.family_spaces["shared"]["shopping_list"][-5:]  # Show last 5
            }

        elif "show" in query.lower() or "what's on" in query.lower():
            return {
                "response": "Family shopping list:",
                "list": self.family_spaces["shared"]["shopping_list"]
            }

        return {"response": "What would you like to do with the shopping list?"}

    def _handle_calendar_query(self, user_id: str, query: str) -> Dict:
        """Handle calendar operations"""

        if "add" in query.lower() or "schedule" in query.lower():
            # Parse event details (simplified)
            return {
                "response": "Event added to family calendar",
                "service": "calendar",
                "action": "add_event"
            }

        elif "today" in query.lower():
            # Get today's events
            return {
                "response": "Today's family schedule",
                "events": self.family_spaces["shared"]["calendar"].get("today", [])
            }

        return {"response": "Calendar query", "service": "calendar"}

    def _handle_media_query(self, user_id: str, query: str, device_id: str) -> Dict:
        """Handle media/Jellyfin queries"""

        device = self.device_registry.get(device_id, {})

        # Check if it's a kid requesting adult content
        if user_id in ["child1", "child2"]:
            if any(word in query.lower() for word in ["rated r", "adult", "horror"]):
                return {
                    "response": "This content requires parental permission",
                    "blocked": True
                }

        return {
            "response": "Accessing Jellyfin media server",
            "service": "jellyfin",
            "port": 8096,
            "device": device.get("name", "unknown")
        }

    def _handle_home_automation(self, user_id: str, query: str, device_id: str) -> Dict:
        """Handle smart home commands"""

        # Kids can't control certain things
        if user_id in ["child1", "child2"]:
            if any(word in query.lower() for word in ["lock", "unlock", "security", "camera"]):
                return {
                    "response": "Please ask a parent to help with security settings",
                    "blocked": True
                }

        if "lights" in query.lower():
            return {
                "response": "Controlling lights",
                "service": "home_automation",
                "action": "lights"
            }

        elif "temperature" in query.lower():
            return {
                "response": "Adjusting temperature",
                "service": "home_automation",
                "action": "climate"
            }

        return {"response": "Home automation command", "service": "home_automation"}

    def _handle_kids_query(self, user_id: str, query: str) -> Dict:
        """Handle queries from children with appropriate filters"""

        # Educational encouragement
        if "homework" in query.lower():
            return {
                "response": "Great job working on homework! How can I help?",
                "mode": "educational"
            }

        # Screen time check
        elif "game" in query.lower() or "play" in query.lower():
            remaining_time = self.family_spaces["kids"]["screen_time"]["limit_minutes"]
            return {
                "response": f"You have {remaining_time} minutes of screen time left today",
                "screen_time": remaining_time
            }

        return {
            "response": "How can I help you today?",
            "mode": "kid_friendly"
        }

    def _handle_admin_query(self, query: str) -> Dict:
        """Handle admin/system queries from Patrick"""

        return {
            "response": "Admin mode activated",
            "services": list(self.services.keys()),
            "devices": len(self.device_registry),
            "family_members": len(self.family_spaces) - 1  # Exclude shared
        }


class FamilyDeviceSetup:
    """Setup different devices for family use"""

    @staticmethod
    def setup_family_devices() -> Dict:
        """Configure typical family devices"""

        devices = {
            # Patrick's main computer
            "patrick_laptop": {
                "name": "Patrick's Laptop",
                "type": "laptop",
                "primary_user": os.getenv("TOWER_USER", "patrick"),
                "allowed_users": [os.getenv("TOWER_USER", "patrick")],
                "restrictions": [],
                "location": "office"
            },

            # Family tablet
            "family_tablet": {
                "name": "Family iPad",
                "type": "tablet",
                "primary_user": "shared",
                "allowed_users": [os.getenv("TOWER_USER", "patrick"), "partner", "child1", "child2"],
                "restrictions": ["no_adult_content", "parental_controls"],
                "location": "living_room"
            },

            # Partner's phone
            "partner_phone": {
                "name": "Partner's iPhone",
                "type": "phone",
                "primary_user": "partner",
                "allowed_users": ["partner", os.getenv("TOWER_USER", "patrick")],  # Patrick for tech support
                "restrictions": [],
                "location": "mobile"
            },

            # Kids tablet (if applicable)
            "kids_tablet": {
                "name": "Kids Fire Tablet",
                "type": "tablet",
                "primary_user": "kids",
                "allowed_users": ["child1", "child2", os.getenv("TOWER_USER", "patrick"), "partner"],
                "restrictions": ["kids_mode", "time_limits", "content_filter"],
                "location": "kids_room"
            },

            # Smart TV
            "living_room_tv": {
                "name": "Living Room TV",
                "type": "tv",
                "primary_user": "shared",
                "allowed_users": [os.getenv("TOWER_USER", "patrick"), "partner", "child1", "child2"],
                "restrictions": ["rating_limits"],
                "location": "living_room"
            }
        }

        return devices


if __name__ == "__main__":
    print("🏠 TOWER FAMILY INTEGRATION SYSTEM")
    print("=" * 60)

    # Initialize integration
    tower = TowerFamilyIntegration()

    # Setup devices
    print("\n📱 Setting up family devices...")
    devices = FamilyDeviceSetup.setup_family_devices()

    for device_id, info in devices.items():
        tower.register_device(device_id, info)
        print(f"  ✅ {info['name']} ({info['type']})")

    print(f"\nTotal devices registered: {len(devices)}")

    # Test scenarios
    print("\n🧪 Testing family scenarios:")

    # 1. Partner adding to shopping list from phone
    print("\n1. Partner adding to shopping list:")
    result = tower.handle_family_query(
        "partner",
        "Add milk to shopping list",
        "partner_phone"
    )
    print(f"   Response: {result['response']}")

    # 2. Child trying to watch adult content
    print("\n2. Child requesting adult content:")
    result = tower.handle_family_query(
        "child1",
        "Watch rated R movie",
        "kids_tablet"
    )
    print(f"   Response: {result['response']}")

    # 3. Patrick checking system status
    print("\n3. Patrick checking system:")
    result = tower.handle_family_query(
        os.getenv("TOWER_USER", "patrick"),
        "System status",
        "patrick_laptop"
    )
    print(f"   Response: {result['response']}")

    print("\n✅ Family integration ready!")
    print("\nFeatures:")
    print("  • Multi-device support")
    print("  • User-appropriate responses")
    print("  • Parental controls")
    print("  • Shared family resources")
    print("  • Admin oversight for support")