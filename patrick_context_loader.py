#!/usr/bin/env python3
"""
Echo Personal Context Loader
Loads Patrick's personal context into Echo responses
"""
import json
import os
from pathlib import Path

class PatrickPersonalContext:
    def __init__(self):
        self.context_file = "/opt/tower-echo-brain/patrick_personal_context.json"
        self.personal_data = self.load_personal_context()
    
    def load_personal_context(self):
        """Load Patrick's personal context data"""
        try:
            if os.path.exists(self.context_file):
                with open(self.context_file, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error loading personal context: {e}")
            return {}
    
    def get_user_profile(self):
        """Get Patrick's professional profile"""
        return self.personal_data.get("user_profile", {})
    
    def get_technical_context(self):
        """Get Patrick's technical background"""
        return self.personal_data.get("technical_intelligence", {})
    
    def get_communication_style(self):
        """Get Patrick's preferred communication style"""
        return self.personal_data.get("communication_preferences", {})
    
    def enhance_response_for_patrick(self, query, base_response):
        """Enhance response based on Patrick's context"""
        profile = self.get_user_profile()
        comm_prefs = self.get_communication_style()
        
        if not profile:
            return base_response
        
        # Enhance based on technical expertise
        if any(tech in query.lower() for tech in ["python", "docker", "fastapi", "kubernetes"]):
            expertise_note = f"Based on your expertise in {', '.join(profile.get('expertise', []))[:3]}, "
            base_response = expertise_note + base_response
        
        # Adjust response style
        if comm_prefs.get("response_style") == "Technical detail matching expertise level":
            base_response += "\n\nNote: Providing senior-level technical detail based on your background."
        
        # Add project context
        current_projects = profile.get("current_projects", [])
        if current_projects and any(proj.lower() in query.lower() for proj in ["tower", "financial", "devops"]):
            base_response += f"\n\nThis relates to your current work on: {', '.join(current_projects)}"
        
        return base_response
    
    def is_patrick_query(self, query):
        """Detect if query is from Patrick"""
        patrick_indicators = ["patrick", "my", "i am", "help me", "what should i"]
        return any(indicator in query.lower() for indicator in patrick_indicators)

# Global instance for Echo to use
patrick_context = PatrickPersonalContext()
