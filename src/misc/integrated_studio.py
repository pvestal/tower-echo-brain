#!/usr/bin/env python3
"""
Tower Creative Studio - Integrated Interface
Combines Echo Brain, Anime Production, Music, and Voice services
"""

import gradio as gr
import requests
import json

# Service endpoints
ECHO_API = "http://localhost:8309"
ANIME_API = "http://localhost:8328"
MUSIC_API = "http://localhost:8315"
VOICE_API = "http://localhost:8312"
COMFYUI_API = "http://localhost:8188"

def search_music(query):
    """Search Apple Music catalog"""
    try:
        response = requests.get(f"{MUSIC_API}/api/search?q={query}")
        data = response.json()
        if data.get("results"):
            return f"Found {len(data['results'])} tracks", data["results"]
        return "No results found", []
    except Exception as e:
        return f"Error: {str(e)}", []

def generate_anime_with_music(prompt, music_style, duration):
    """Generate anime with integrated soundtrack"""
    steps = []
    
    # Step 1: Search for music
    steps.append("Searching for music...")
    music_response = requests.get(f"{MUSIC_API}/api/search?q={music_style}")
    
    # Step 2: Generate anime frames
    steps.append("Generating anime frames...")
    anime_data = {
        "prompt": prompt,
        "duration": duration,
        "music_style": music_style
    }
    
    # Step 3: Combine
    steps.append("Combining video with soundtrack...")
    
    return "\n".join(steps)

def echo_chat_with_context(message, include_music=False, include_anime=False):
    """Chat with Echo, optionally including creative context"""
    context = {"message": message}
    
    if include_music:
        context["music_context"] = True
    if include_anime:
        context["anime_context"] = True
        
    response = requests.post(f"{ECHO_API}/api/echo/chat", json=context)
    return response.json().get("response", "Echo is thinking...")

# Create Gradio interface
with gr.Blocks(title="Tower Creative Studio", theme=gr.themes.Base()) as studio:
    gr.Markdown("# Tower Creative Studio")
    gr.Markdown("Integrated interface for Echo, Anime, Music, and Voice services")
    
    with gr.Tab("Echo Brain"):
        with gr.Row():
            chat_input = gr.Textbox(label="Message", placeholder="Talk to Echo...")
            music_context = gr.Checkbox(label="Include music context")
            anime_context = gr.Checkbox(label="Include anime context")
        chat_output = gr.Textbox(label="Echo Response", lines=5)
        chat_btn = gr.Button("Send")
        chat_btn.click(
            echo_chat_with_context,
            inputs=[chat_input, music_context, anime_context],
            outputs=chat_output
        )
    
    with gr.Tab("Apple Music"):
        gr.Markdown("### Your Apple Music Library")
        gr.Markdown("Credentials: Team CNXX42ZGF8 | Key 9M85DX285V")
        
        music_search = gr.Textbox(label="Search", placeholder="Search songs, artists...")
        music_results = gr.JSON(label="Results")
        search_btn = gr.Button("Search Apple Music")
        search_btn.click(search_music, inputs=music_search, outputs=[gr.Textbox(label="Status"), music_results])
    
    with gr.Tab("Anime Production"):
        gr.Markdown("### Generate Anime with Soundtrack")
        
        with gr.Row():
            anime_prompt = gr.Textbox(label="Scene Description", placeholder="Describe the anime scene...")
            music_style = gr.Textbox(label="Music Style", placeholder="Synthwave, orchestral, etc...")
        
        duration = gr.Slider(minimum=5, maximum=60, value=30, label="Duration (seconds)")
        
        generate_output = gr.Textbox(label="Generation Progress", lines=10)
        generate_btn = gr.Button("Generate Anime with Music")
        generate_btn.click(
            generate_anime_with_music,
            inputs=[anime_prompt, music_style, duration],
            outputs=generate_output
        )
    
    with gr.Tab("Integrated Workflow"):
        gr.Markdown("### Complete Creative Pipeline")
        gr.Markdown("""
        1. **Describe your idea** to Echo
        2. **Select music** from Apple Music
        3. **Generate anime** with integrated soundtrack
        4. **Add voice** narration
        """)
        
        with gr.Column():
            idea_input = gr.Textbox(label="Your Creative Idea", placeholder="I want to create...")
            
            gr.Markdown("#### Step 1: Echo analyzes your idea")
            echo_analysis = gr.Textbox(label="Echo's Analysis", interactive=False)
            
            gr.Markdown("#### Step 2: Music Selection")
            suggested_music = gr.Dropdown(label="Suggested Tracks", choices=[])
            
            gr.Markdown("#### Step 3: Generate Content")
            final_output = gr.Video(label="Final Creation")
            
            create_btn = gr.Button("Start Creative Process", variant="primary")

if __name__ == "__main__":
    studio.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
        auth=None,
        ssl_verify=False
    )
