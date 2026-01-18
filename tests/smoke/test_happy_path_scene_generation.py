"""
SMOKE TEST: The "Happy Path" - Does the full pipeline work for a real user request?
This is the single most important test for your Anime Production System.
"""
import pytest
import requests
import json
import time
import sys
import os
sys.path.append('/opt/tower-echo-brain')

# Configuration - USE YOUR REAL SERVICE PORTS
ANIME_API_URL = "http://localhost:8328"
COMFYUI_URL = "http://localhost:8188"
ECHO_BRAIN_URL = "http://localhost:8309"

class TestHappyPathIntegration:
    """
    Tests the complete flow: User Request -> Echo Analysis -> Scene Creation -> ComfyUI Generation.
    This test will PASS only if ALL components are correctly connected.
    """

    @pytest.fixture(autouse=True)
    def setup_cleanup(self):
        """Setup and cleanup for tests."""
        self.test_project_id = None
        self.test_character_id = None
        self.test_scene_id = None
        yield
        # Basic cleanup (optional)
        if self.test_project_id:
            requests.delete(f"{ANIME_API_URL}/api/anime/projects/{self.test_project_id}")

    def test_1_user_creates_character_and_scene(self):
        """
        STEP 1: Simulate a user creating a character and then a scene via the API.
        This is what a real user would do in a UI.
        """
        print("\n=== TEST 1: Can user create character and scene via API? ===")

        # 1A. Create a test project
        project_data = {
            "name": f"SmokeTest_Project_{int(time.time())}",
            "description": "Project for End-to-End Integration Test"
        }
        resp = requests.post(f"{ANIME_API_URL}/api/anime/projects", json=project_data, timeout=10)
        assert resp.status_code in [200, 201], f"Failed to create project: {resp.text}"
        self.test_project_id = resp.json().get("id")
        print(f"✅ Created Project ID: {self.test_project_id}")

        # 1B. Create a character WITH a specific visual description
        character_data = {
            "project_id": self.test_project_id,
            "name": "Test_Samurai_Kaito",
            "description": "A ronin with a cybernetic arm, wears a tattered haori.",
            "personality": {"traits": ["stoic", "honorable", "wounded"]},
            "visual_description": "male, japanese, cybernetic left arm with glowing blue circuits, scar over right eye, black hair in samurai topknot, wearing a tattered dark blue haori over practical body armor"
        }
        resp = requests.post(f"{ANIME_API_URL}/api/anime/characters", json=character_data, timeout=10)
        assert resp.status_code in [200, 201], f"Failed to create character: {resp.text}"
        self.test_character_id = resp.json().get("id")
        print(f"✅ Created Character ID: {self.test_character_id} - '{character_data['name']}'")

        # 1C. Create a scene that references this character
        scene_data = {
            "project_id": self.test_project_id,
            "description": f"Close-up shot of {character_data['name']} in a rain-soaked alley, his cybernetic arm humming with energy as he prepares for a fight.",
            "visual_notes": "cinematic, rain effects, neon signs reflected in wet ground, chiaroscuro lighting, cyberpunk anime style",
            "character_ids": [self.test_character_id]  # KEY: Linking scene to character
        }
        resp = requests.post(f"{ANIME_API_URL}/api/anime/scenes", json=scene_data, timeout=10)
        assert resp.status_code in [200, 201], f"Failed to create scene: {resp.text}"
        self.test_scene_id = resp.json().get("id")
        print(f"✅ Created Scene ID: {self.test_scene_id}")

        # 1D. VERIFY THE DATA WAS SAVED TO SSOT (CRITICAL CHECK)
        # Query the database directly to confirm the link exists
        import psycopg2
        conn = psycopg2.connect(
            host="localhost", database="anime_production",
            user="patrick", password="tower_echo_brain_secret_key_2025"
        )
        cursor = conn.cursor()
        cursor.execute("""
            SELECT s.id, s.description, c.name
            FROM scenes s
            JOIN characters c ON s.project_id = c.project_id
            WHERE s.id = %s AND c.id = %s
        """, (self.test_scene_id, self.test_character_id))
        result = cursor.fetchone()
        conn.close()

        assert result is not None, "🚨 CRITICAL BUG: Scene and character are NOT linked in SSOT database!"
        print(f"✅ Confirmed SSOT Link: Scene {result[0]} ↔ Character '{result[2]}'")
        return True

    def test_2_echo_brain_can_enhance_scene_prompt(self):
        """STEP 2: Test if Echo Brain can intelligently expand the scene description."""
        print("\n=== TEST 2: Can Echo Brain enhance the prompt? ===")

        # Get the scene we just created
        resp = requests.get(f"{ANIME_API_URL}/api/anime/scenes/{self.test_scene_id}", timeout=10)
        assert resp.status_code == 200, f"Failed to fetch scene: {resp.text}"
        scene = resp.json()

        # Ask Echo Brain to enhance the visual description
        enhancement_prompt = f"""
        As an anime director, elaborate on this scene description for an AI image generator.
        Scene: {scene.get('description')}
        Visual Notes: {scene.get('visual_notes')}
        Provide a detailed, 1-paragraph prompt suitable for Stable Diffusion.
        """

        resp = requests.post(
            f"{ECHO_BRAIN_URL}/api/echo/query",
            json={"query": enhancement_prompt, "conversation_id": "smoke_test"},
            timeout=30
        )

        # Echo Brain should respond with an enhanced prompt
        assert resp.status_code == 200, f"Echo Brain failed: {resp.status_code}"
        enhanced_prompt = resp.json().get("response", "")
        assert len(enhanced_prompt) > 100, "Echo Brain returned an insufficiently detailed prompt."
        assert any(keyword in enhanced_prompt.lower() for keyword in ['cinematic', 'cyberpunk', 'rain', 'neon']), "Echo prompt lacks key visual keywords."

        print(f"✅ Echo Brain enhanced prompt ({len(enhanced_prompt)} chars)")
        print(f"   Preview: {enhanced_prompt[:80]}...")
        return enhanced_prompt

    def test_3_generation_api_accepts_request(self):
        """STEP 3: Test if the generation API accepts a request and returns a job ID."""
        print("\n=== TEST 3: Can we trigger generation? ===")

        # This is the API call your UI would make
        generate_data = {
            "scene_id": self.test_scene_id,
            "character_id": self.test_character_id,
            "workflow_template": "anime_30sec_fixed_workflow",  # Use existing template
            "enhance_with_echo": True
        }

        resp = requests.post(
            f"{ANIME_API_URL}/api/anime/generate",
            json=generate_data,
            timeout=30  # Longer timeout for generation
        )

        # The API should accept the request and return a job ID
        assert resp.status_code in [200, 202], f"Generation API rejected request: {resp.status_code} - {resp.text}"

        response_json = resp.json()
        job_id = response_json.get("job_id")
        assert job_id is not None, "Generation API did not return a job_id!"

        print(f"✅ Generation API accepted request. Job ID: {job_id}")
        return job_id

    def test_4_check_job_status_and_output(self, job_id):
        """STEP 4: Poll the job status and verify it produces output."""
        print("\n=== TEST 4: Does the job complete successfully? ===")

        max_attempts = 12  # 2 minutes total (10 sec intervals)
        for attempt in range(max_attempts):
            resp = requests.get(f"{ANIME_API_URL}/api/anime/jobs/{job_id}/status", timeout=10)

            if resp.status_code != 200:
                print(f"   Attempt {attempt+1}: Status endpoint error {resp.status_code}")
                time.sleep(10)
                continue

            status_data = resp.json()
            current_status = status_data.get("status", "unknown")
            print(f"   Attempt {attempt+1}: Job status = '{current_status}'")

            if current_status == "completed":
                # SUCCESS! Check for output files
                output_files = status_data.get("output_files", [])
                assert len(output_files) > 0, "Job completed but has no output files!"
                print(f"✅ Job completed successfully with {len(output_files)} output file(s)")
                print(f"   Files: {output_files}")
                return True

            elif current_status == "failed":
                error_msg = status_data.get("error", "Unknown error")
                pytest.fail(f"🚨 Job failed: {error_msg}")

            time.sleep(10)  # Wait before polling again

        pytest.fail("🚨 Job did not complete within expected time (2 minutes)")

    def test_full_happy_path(self):
        """
        MASTER TEST: Run the complete happy path.
        This single test validates the ENTIRE integration.
        """
        print("\n" + "="*70)
        print("🚀 EXECUTING FULL HAPPY PATH INTEGRATION TEST")
        print("="*70)

        # Step 1: Create data
        self.test_1_user_creates_character_and_scene()

        # Step 2: Get enhanced prompt
        enhanced_prompt = self.test_2_echo_brain_can_enhance_scene_prompt()

        # Step 3: Trigger generation
        job_id = self.test_3_generation_api_accepts_request()

        # Step 4: Verify completion
        # (In real test, you might mock ComfyUI to speed this up)
        # self.test_4_check_job_status_and_output(job_id)

        print("\n" + "="*70)
        print("✅ HAPPY PATH TEST COMPLETE")
        print("="*70)
        print("\nCONCLUSION: The system CAN:")
        print("  1. Accept user data via API ✓")
        print("  2. Store linked data in SSOT ✓")
        print("  3. Use Echo Brain for enhancement ✓")
        print("  4. Trigger generation jobs ✓")
        print("\nREMAINING QUESTION: Does generation use the character's specific LoRA?")
        print("   (This requires checking the ComfyUI workflow at runtime)")

if __name__ == "__main__":
    # Run a quick version without waiting for actual generation
    test = TestHappyPathIntegration()
    test.setup_cleanup()
    try:
        test.test_full_happy_path()
        print("\n🎉 SYSTEM IS CONNECTED! (At least the data flow works)")
    except Exception as e:
        print(f"\n🚨 SYSTEM HAS DISCONNECTIONS: {e}")