
# Narrative Director API Endpoints

@router.get("/api/echo/narrative/{character_name}/questions")
async def get_character_story_questions(character_name: str, story_beat: str = "opening"):
    """Get intelligent story development questions for character at specific beat"""
    try:
        sys.path.append("/opt/tower-anime-production")
        from narrative_director import get_narrative_questions
        
        questions = get_narrative_questions(character_name, story_beat)
        return {"success": True, "questions": questions, "story_beat": story_beat}
    except Exception as e:
        logger.error(f"Error getting narrative questions: {e}")
        return {"success": False, "error": str(e)}

@router.post("/api/echo/narrative/{character_name}/generate")
async def generate_story_driven_character(character_name: str, story_beat: str, user_answers: dict):
    """Generate character using story-driven prompts from user answers"""
    try:
        sys.path.append("/opt/tower-anime-production")
        from narrative_director import build_story_driven_prompt
        
        # Build dynamic prompt from story context
        prompt_data = build_story_driven_prompt(character_name, story_beat, user_answers)
        
        # Use ComfyUI to generate with story context
        result = await comfyui_tools.generate_image(
            prompt=prompt_data["prompt"],
            negative_prompt="cartoon, chibi, simple anime, flat colors, 2D anime style",
            prefix=f"story_{character_name.lower().replace( , _)}_{story_beat}"
        )
        
        result["narrative_context"] = prompt_data["narrative_context"]
        result["story_seed"] = prompt_data["story_seed"]
        
        return {"success": True, "generation": result}
    except Exception as e:
        logger.error(f"Error generating story-driven character: {e}")
        return {"success": False, "error": str(e)}

