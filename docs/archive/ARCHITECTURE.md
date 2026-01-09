
## Database Separation (2026-01-05)

Echo Brain and Anime Production now use separate databases:

| System | Database | Tables | Purpose |
|--------|----------|--------|---------|
| Echo Brain | tower_consolidated | 41 | Model routing, conversations, analysis cache |
| Anime Production | tower_anime | 28 | Character generation, animation workflows |

### Separated Tables:
- **Anime System**: animation_projects, anime_generations, character_*, lora_combinations, generation_*, scene_*, style_*, video_*, workflow_*, music_*, etc.
- **Echo Brain**: echo_*, model_*, intent_*, query_*, past_solutions, codebase_index

This separation eliminates context contamination between systems.

### Benefits:
- ✅ Zero cross-contamination of memory contexts
- ✅ Independent scaling and backup strategies  
- ✅ Clear architectural boundaries
- ✅ Simplified debugging and monitoring

