#!/usr/bin/env python3
"""
AI Assist Family Multi-User System
Provides privacy between users while maintaining admin oversight
"""

import os
from enum import Enum
from typing import Dict, List, Optional, Any
from datetime import datetime, date
import hashlib
import json

class UserRole(Enum):
    """User roles in the family system"""
    ADMIN = "admin"           # Full access (Patrick)
    ADULT = "adult"          # Family adult member
    TEEN = "teen"            # Teenager (some restrictions)
    CHILD = "child"          # Child (parental controls)
    GUEST = "guest"          # Temporary guest access

class FamilyMember:
    """Represents a family member in the system"""

    def __init__(self, user_id: str, name: str, role: UserRole,
                 birth_date: Optional[date] = None,
                 parent_id: Optional[str] = None):
        self.user_id = user_id
        self.name = name
        self.role = role
        self.birth_date = birth_date
        self.parent_id = parent_id  # For children, who their parent is
        self.created_at = datetime.now()
        self.preferences = {}
        self.restrictions = []

class EchoFamilySystem:
    """
    Multi-user system for AI Assist with family-appropriate access control
    """

    def __init__(self):
        self.family_members = {}
        self.shared_resources = {}
        self.family_calendar = {}
        self.shopping_list = []
        self.admin_ids = []
        self._initialize_family()

    def _initialize_family(self):
        """Initialize with Patrick's family configuration"""

        # Patrick as admin
        patrick = FamilyMember(
            user_id = os.getenv("TOWER_USER", os.getenv("TOWER_USER", "patrick")),
            name="Patrick",
            role=UserRole.ADMIN
        )
        self.family_members[os.getenv("TOWER_USER", "patrick")] = patrick
        self.admin_ids.append(os.getenv("TOWER_USER", "patrick"))

        # Example family members (customize as needed)
        # Wife/Partner
        partner = FamilyMember(
            user_id="partner",
            name="Partner",
            role=UserRole.ADULT
        )
        self.family_members["partner"] = partner

        # Children (if any)
        # child1 = FamilyMember(
        #     user_id="child1",
        #     name="Child1",
        #     role=UserRole.CHILD,
        #     birth_date=date(2015, 1, 1),
        #     parent_id="patrick"
        # )
        # self.family_members["child1"] = child1

    def authenticate_user(self, user_id: str, password: Optional[str] = None) -> Dict:
        """Authenticate a family member"""

        if user_id not in self.family_members:
            return {
                "authenticated": False,
                "error": "Unknown user"
            }

        member = self.family_members[user_id]

        # Children might use simpler auth (like a PIN)
        if member.role == UserRole.CHILD:
            # Simplified auth for children
            pass

        return {
            "authenticated": True,
            "user_id": user_id,
            "name": member.name,
            "role": member.role.value,
            "is_admin": user_id in self.admin_ids
        }

    def check_access_permission(self, requester_id: str, target_id: str,
                                resource_type: str) -> bool:
        """
        Check if requester can access target's resources

        Access rules:
        - Admins can access everything (for support/help)
        - Parents can access their children's data
        - Users can access their own data
        - Shared resources are accessible to all family
        - Guests have limited access
        """

        # Admin override - Patrick can help anyone
        if requester_id in self.admin_ids:
            return True

        # Self access
        if requester_id == target_id:
            return True

        # Parent accessing child's data
        if target_id in self.family_members:
            target = self.family_members[target_id]
            if target.parent_id == requester_id:
                return True

        # Shared family resources
        if resource_type in ["calendar", "shopping_list", "shared_notes"]:
            requester = self.family_members.get(requester_id)
            if requester and requester.role != UserRole.GUEST:
                return True

        return False

    def get_user_data(self, requester_id: str, target_id: str,
                      data_type: str) -> Dict:
        """Get user data with permission checking"""

        # Check permissions
        if not self.check_access_permission(requester_id, target_id, data_type):
            return {
                "error": "Access denied",
                "message": "You don't have permission to access this user's data"
            }

        # For admin access, add a note
        is_admin_access = (requester_id != target_id and
                          requester_id in self.admin_ids)

        result = {
            "data": f"Data for {target_id}",  # Actual data retrieval here
            "accessed_by": requester_id
        }

        if is_admin_access:
            result["admin_access"] = True
            result["reason"] = "Administrative access for support"
            # Log admin access for transparency
            self._log_admin_access(requester_id, target_id, data_type)

        return result

    def _log_admin_access(self, admin_id: str, target_id: str, data_type: str):
        """Log when admin accesses other users' data for accountability"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "admin": admin_id,
            "accessed_user": target_id,
            "data_type": data_type
        }
        # Store in audit log
        print(f"📝 Admin access logged: {log_entry}")

    def apply_content_filters(self, user_id: str, content: str) -> str:
        """Apply age-appropriate content filters"""

        member = self.family_members.get(user_id)
        if not member:
            return content

        if member.role == UserRole.CHILD:
            # Apply child-safe filters
            blocked_words = ["explicit", "violence", "adult"]
            for word in blocked_words:
                if word in content.lower():
                    return "This content is not appropriate for your age."

            # Limit certain capabilities
            if any(danger in content.lower() for danger in
                   ["delete", "remove", "credit card", "password"]):
                return "Please ask a parent for help with this."

        elif member.role == UserRole.TEEN:
            # Less restrictive but still some filters
            if "explicit" in content.lower():
                return "This content may not be appropriate."

        return content

    def share_with_family(self, user_id: str, resource_type: str,
                         data: Any) -> bool:
        """Share resource with family members"""

        if resource_type == "calendar_event":
            self.family_calendar[datetime.now().isoformat()] = {
                "created_by": user_id,
                "event": data,
                "shared": True
            }
            return True

        elif resource_type == "shopping_item":
            self.shopping_list.append({
                "added_by": user_id,
                "item": data,
                "timestamp": datetime.now().isoformat()
            })
            return True

        elif resource_type == "note":
            self.shared_resources[f"note_{datetime.now().timestamp()}"] = {
                "from": user_id,
                "content": data,
                "shared_with": "family"
            }
            return True

        return False

    def get_family_dashboard(self, user_id: str) -> Dict:
        """Get family dashboard appropriate for user's role"""

        member = self.family_members.get(user_id)
        if not member:
            return {"error": "Unknown user"}

        dashboard = {
            "user": member.name,
            "role": member.role.value,
            "timestamp": datetime.now().isoformat()
        }

        # Everyone sees shared resources
        dashboard["shopping_list"] = self.shopping_list[-10:]  # Last 10 items
        dashboard["calendar_today"] = self._get_todays_events()

        # Role-specific content
        if member.role == UserRole.ADMIN:
            # Admin sees everything
            dashboard["family_members"] = list(self.family_members.keys())
            dashboard["system_status"] = self._get_system_status()
            dashboard["recent_activity"] = self._get_recent_activity()

        elif member.role == UserRole.ADULT:
            # Adults see family info but not system internals
            dashboard["family_members"] = [
                m.name for m in self.family_members.values()
                if m.role != UserRole.GUEST
            ]

        elif member.role == UserRole.CHILD:
            # Children see limited info
            dashboard["my_tasks"] = self._get_child_tasks(user_id)
            dashboard["fun_fact"] = self._get_daily_fun_fact()

        return dashboard

    def _get_todays_events(self) -> List[Dict]:
        """Get today's calendar events"""
        today = date.today().isoformat()
        return [
            event for timestamp, event in self.family_calendar.items()
            if timestamp.startswith(today)
        ]

    def _get_system_status(self) -> Dict:
        """Get system status (admin only)"""
        return {
            "total_users": len(self.family_members),
            "active_services": ["echo", "calendar", "shopping", "notes"],
            "tower_status": "operational"
        }

    def _get_recent_activity(self) -> List[str]:
        """Get recent family activity (admin only)"""
        # This would pull from actual activity logs
        return [
            "Partner added milk to shopping list",
            "Calendar event: Family dinner tomorrow",
            "System backup completed"
        ]

    def _get_child_tasks(self, child_id: str) -> List[str]:
        """Get child's tasks/chores"""
        # This would be customized per child
        return [
            "Homework: Math worksheet",
            "Chore: Clean room",
            "Practice: Piano 30 minutes"
        ]

    def _get_daily_fun_fact(self) -> str:
        """Get a fun fact for children"""
        facts = [
            "Did you know? Octopuses have three hearts!",
            "Fun fact: Honey never spoils!",
            "Cool! The Moon moves away from Earth 1.5 inches per year!"
        ]
        import random
        return random.choice(facts)


class EchoFamilyChat:
    """Family-aware chat system for AI Assist"""

    def __init__(self):
        self.family = EchoFamilySystem()
        self.conversations = {}  # Stored per user

    async def process_message(self, user_id: str, message: str,
                             context: Optional[Dict] = None) -> Dict:
        """Process message with family context awareness"""

        # Authenticate user
        auth = self.family.authenticate_user(user_id)
        if not auth["authenticated"]:
            return {"error": "Please identify yourself"}

        # Check for admin commands
        if auth["is_admin"] and message.startswith("/admin"):
            return await self._handle_admin_command(user_id, message)

        # Check for family sharing commands
        if message.startswith("/family"):
            return await self._handle_family_command(user_id, message)

        # Apply content filters for children
        member = self.family.family_members[user_id]
        if member.role in [UserRole.CHILD, UserRole.TEEN]:
            message = self.family.apply_content_filters(user_id, message)

        # Process with appropriate context
        response = await self._generate_response(user_id, message, member.role)

        # Store conversation (isolated per user)
        if user_id not in self.conversations:
            self.conversations[user_id] = []

        self.conversations[user_id].append({
            "timestamp": datetime.now().isoformat(),
            "message": message,
            "response": response
        })

        return {
            "response": response,
            "user": member.name,
            "role": member.role.value
        }

    async def _handle_admin_command(self, admin_id: str, command: str) -> Dict:
        """Handle admin commands"""

        parts = command.split()

        if len(parts) < 2:
            return {"response": "Invalid admin command"}

        if parts[1] == "help":
            target_user = parts[2] if len(parts) > 2 else None
            if target_user:
                # Admin helping specific user
                user_data = self.family.get_user_data(admin_id, target_user, "support")
                return {
                    "response": f"Accessing {target_user}'s data for support...",
                    "data": user_data,
                    "admin_mode": True
                }

        elif parts[1] == "status":
            dashboard = self.family.get_family_dashboard(admin_id)
            return {
                "response": "System status",
                "dashboard": dashboard,
                "admin_mode": True
            }

        return {"response": "Unknown admin command"}

    async def _handle_family_command(self, user_id: str, command: str) -> Dict:
        """Handle family sharing commands"""

        parts = command.split(maxsplit=2)

        if len(parts) < 3:
            return {"response": "Usage: /family [share|add] [type] [content]"}

        action = parts[1]

        if action == "add":
            # Add to shopping list
            if "shopping" in parts[2]:
                item = parts[2].replace("shopping", "").strip()
                self.family.share_with_family(user_id, "shopping_item", item)
                return {"response": f"Added '{item}' to family shopping list"}

        elif action == "share":
            # Share note with family
            content = parts[2]
            self.family.share_with_family(user_id, "note", content)
            return {"response": "Shared with family"}

        return {"response": "Unknown family command"}

    async def _generate_response(self, user_id: str, message: str,
                                 role: UserRole) -> str:
        """Generate role-appropriate response"""

        # Customize response based on role
        if role == UserRole.CHILD:
            # Friendly, educational responses for children
            return f"Great question! Let me explain that in a fun way..."

        elif role == UserRole.ADMIN:
            # Technical, detailed responses for admin
            return f"Technical analysis: {message}..."

        else:
            # Standard response for adults
            return f"Here's what I found: {message}..."


if __name__ == "__main__":
    print("👨‍👩‍👧‍👦 ECHO BRAIN FAMILY SYSTEM")
    print("=" * 60)

    family = EchoFamilySystem()

    print("\n👥 Family Members:")
    for user_id, member in family.family_members.items():
        print(f"  • {member.name} ({member.role.value})")

    print("\n🔐 Access Control Examples:")

    # Patrick accessing family member data (allowed)
    print("\n1. Patrick helping partner:")
    result = family.get_user_data(os.getenv("TOWER_USER", "patrick"), "partner", "settings")
    print(f"   Result: {result}")

    # Partner accessing Patrick's data (denied)
    print("\n2. Partner accessing Patrick's data:")
    result = family.get_user_data("partner", os.getenv("TOWER_USER", "patrick"), "settings")
    print(f"   Result: {result}")

    # Family shopping list (shared)
    print("\n3. Adding to family shopping list:")
    family.share_with_family("partner", "shopping_item", "Milk")
    family.share_with_family(os.getenv("TOWER_USER", "patrick"), "shopping_item", "Bread")
    print(f"   Shopping list: {family.shopping_list}")

    print("\n✅ Family system configured!")
    print("Features:")
    print("  • User privacy by default")
    print("  • Admin override for support")
    print("  • Parental controls for children")
    print("  • Shared family resources")
    print("  • Age-appropriate content filtering")