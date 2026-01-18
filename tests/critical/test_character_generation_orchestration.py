"""
CRITICAL TEST: Character Generation Orchestration
Tests that characters are properly connected to their LoRAs through generation profiles
"""
import pytest
import sys
sys.path.append('/opt/tower-echo-brain')

class TestCharacterGenerationOrchestration:
    """Test the complete character → LoRA → workflow → generation pipeline"""
    
    @pytest.fixture
    def db_connection(self):
        """Connect to the anime_production database"""
        import psycopg2
        conn = psycopg2.connect(
            host="localhost",
            database="anime_production",
            user="patrick",
            password="tower_echo_brain_secret_key_2025"
        )
        yield conn
        conn.close()
    
    def test_characters_have_generation_profiles(self, db_connection):
        """
        CRITICAL TEST: Every character should have a generation profile
        that connects them to their LoRA and workflow
        """
        cursor = db_connection.cursor()
        
        # Get all characters
        cursor.execute("SELECT id, name FROM characters")
        characters = cursor.fetchall()
        
        assert len(characters) > 0, "No characters found in database"
        
        characters_without_profiles = []
        
        for char_id, char_name in characters:
            # Check if character has a generation profile
            cursor.execute("""
                SELECT COUNT(*) FROM generation_profiles 
                WHERE EXISTS (
                    SELECT 1 FROM ai_models 
                    WHERE character_name = %s 
                    AND model_type = 'lora'
                    AND ai_models.id = generation_profiles.lora_id
                )
            """, (char_name,))
            
            profile_count = cursor.fetchone()[0]
            
            if profile_count == 0:
                characters_without_profiles.append(char_name)
        
        if characters_without_profiles:
            print(f"❌ Characters without proper generation profiles: {characters_without_profiles}")
        
        # This will likely FAIL, exposing the orchestration bug
        assert len(characters_without_profiles) == 0, (
            f"🚨 ORCHESTRATION BUG: {len(characters_without_profiles)} characters lack generation profiles: "
            f"{', '.join(characters_without_profiles)}"
        )
        
        print("✅ All characters have generation profiles")
    
    def test_mei_character_pipeline(self, db_connection):
        """
        SPECIFIC TEST: Mei character should have complete pipeline
        """
        cursor = db_connection.cursor()
        
        # Check Mei exists
        cursor.execute("SELECT id, name FROM characters WHERE name = 'Mei'")
        mei_char = cursor.fetchone()
        
        if not mei_char:
            pytest.skip("Mei character not found")
        
        mei_id, mei_name = mei_char
        print(f"Testing complete pipeline for {mei_name}")
        
        # Check Mei has LoRA
        cursor.execute("""
            SELECT id, model_name, model_path 
            FROM ai_models 
            WHERE character_name = 'Mei' AND model_type = 'lora'
        """)
        mei_lora = cursor.fetchone()
        
        assert mei_lora is not None, (
            "🚨 BUG: Mei character has no LoRA model"
        )
        
        lora_id, lora_name, lora_path = mei_lora
        print(f"Found Mei's LoRA: {lora_name}")
        
        # Check generation profile exists that connects Mei's LoRA to a workflow
        cursor.execute("""
            SELECT gp.id, vwt.name
            FROM generation_profiles gp
            JOIN video_workflow_templates vwt ON gp.video_workflow_template_id = vwt.id
            WHERE gp.lora_id = %s
        """, (lora_id,))
        
        profile_workflow = cursor.fetchone()
        
        assert profile_workflow is not None, (
            f"🚨 CRITICAL BUG: Mei's LoRA {lora_name} is not connected to any workflow via generation_profiles. "
            f"This means generation requests will not use Mei's character-specific LoRA."
        )
        
        profile_id, workflow_name = profile_workflow
        print(f"✅ Mei's pipeline: Character → LoRA '{lora_name}' → Workflow '{workflow_name}'")
        
    def test_generation_api_workflow_selection(self, db_connection):
        """
        TEST: API should select the correct workflow for character generation
        This tests the actual generation logic
        """
        cursor = db_connection.cursor()
        
        # Simulate what the API should do:
        # 1. Get a scene that requires character Mei
        # 2. Find Mei's generation profile
        # 3. Use the correct LoRA and workflow
        
        character_name = "Mei"
        
        # This query simulates the API logic for character generation
        cursor.execute("""
            SELECT 
                c.name as character_name,
                am.model_name as lora_model,
                am.model_path as lora_path,
                vwt.name as workflow_name,
                vwt.workflow_template
            FROM characters c
            JOIN ai_models am ON c.name = am.character_name AND am.model_type = 'lora'
            JOIN generation_profiles gp ON gp.lora_id = am.id
            JOIN video_workflow_templates vwt ON gp.video_workflow_template_id = vwt.id
            WHERE c.name = %s
        """, (character_name,))
        
        generation_config = cursor.fetchone()
        
        assert generation_config is not None, (
            f"🚨 GENERATION API BUG: Cannot get complete generation config for {character_name}. "
            f"This means the API cannot properly generate this character."
        )
        
        char_name, lora_model, lora_path, workflow_name, workflow_template = generation_config
        
        # Verify the workflow actually contains the LoRA model reference
        import json
        workflow_str = json.dumps(workflow_template)
        
        # The workflow should reference the specific LoRA file
        lora_filename = lora_model if lora_model.endswith('.safetensors') else f"{lora_model}.safetensors"
        
        # This is the critical check - does the workflow actually use the character's LoRA?
        assert lora_filename.replace('.safetensors', '') in workflow_str or lora_model in workflow_str, (
            f"🚨 WORKFLOW BUG: Workflow '{workflow_name}' does not reference LoRA '{lora_model}'. "
            f"Generation will produce generic anime instead of {char_name}."
        )
        
        print(f"✅ Complete generation pipeline verified for {char_name}")
        print(f"   LoRA: {lora_model}")
        print(f"   Workflow: {workflow_name}")

if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
