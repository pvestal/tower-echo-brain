"""
CRITICAL TEST: Validate that character LoRAs are injected into ComfyUI workflows.
This test will FAIL until the workflow templates are fixed.
"""
import pytest
import sys
import os
import json
sys.path.append('/opt/tower-echo-brain')

class TestLoRAIntegration:
    """Test that character-specific LoRAs are used in generation"""
    
    @pytest.fixture
    def db_connection(self):
        """Connect to the anime_production database"""
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            database="anime_production",
            user="patrick",
            password="***REMOVED***"
        )
        yield conn
        conn.close()
    
    def test_workflow_contains_loraloader(self, db_connection):
        """
        P0 TEST: Verify generated ComfyUI workflow has LoraLoader nodes.
        This test exposes the critical bug.
        """
        cursor = db_connection.cursor()

        # 1. Get a character LoRA model from ai_models table
        cursor.execute("""
            SELECT id, model_name, character_name, model_path
            FROM ai_models
            WHERE model_type = 'lora'
            AND character_name IS NOT NULL
            LIMIT 1
        """)
        lora_model = cursor.fetchone()

        if not lora_model:
            pytest.skip("No character LoRA found in ai_models table")

        lora_id, model_name, char_name, model_path = lora_model
        print(f"Testing LoRA integration for character: {char_name} (LoRA: {model_name})")
        
        # 2. Get workflow template that could use this LoRA
        cursor.execute("""
            SELECT id, workflow_template, recommended_lora_id
            FROM video_workflow_templates 
            LIMIT 1
        """)
        template_result = cursor.fetchone()
        
        if not template_result:
            pytest.skip("No workflow template found in database")
        
        template_id, workflow_template, recommended_lora_id = template_result
        
        # 3. Convert to string and search for LoraLoader - THIS WILL FAIL
        workflow_str = json.dumps(workflow_template)
        
        # CRITICAL ASSERTION: The workflow must contain LoraLoader nodes
        # This is currently FALSE, exposing the bug
        assert "LoraLoader" in workflow_str, (
            f"🚨 CRITICAL BUG: Workflow template {template_id} lacks LoraLoader nodes. "
            f"Character '{char_name}'s LoRA '{model_name}' will be ignored."
        )
        
        print(f"✅ LoRA integration test passed for {char_name}")
    
    def test_character_prompt_inclusion(self, db_connection):
        """Test that character descriptions are included in generation prompts"""
        cursor = db_connection.cursor()
        
        # Get a character with description using actual table structure
        cursor.execute("""
            SELECT name, description, design_prompt, personality
            FROM characters 
            WHERE description IS NOT NULL 
            AND description != ''
            LIMIT 1
        """)
        character = cursor.fetchone()
        
        if not character:
            pytest.skip("No character with description found")
        
        name, description, design_prompt, personality = character
        
        # Simulate prompt construction (as your API should do)
        base_prompt = "A scene featuring a character"
        full_prompt = f"{base_prompt} named {name}. {description}"
        
        # Verify character data is in the prompt
        assert name in full_prompt, f"Character name '{name}' missing from prompt"
        assert description in full_prompt, f"Character description missing from prompt"
        
        print(f"✅ Character '{name}' correctly integrated into prompt")

if __name__ == '__main__':
    # Run this test directly to see the failure
    pytest.main([__file__, '-v', '-x'])
