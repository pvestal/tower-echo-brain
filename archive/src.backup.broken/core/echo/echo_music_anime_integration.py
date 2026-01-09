#!/usr/bin/env python3
import requests
import json

class MusicAnimeIntegration:
    def __init__(self):
        self.anime_api = 'http://localhost:8328'
        self.apple_music_api = 'http://localhost:8315'
        self.comfyui_api = 'http://localhost:8188'
    
    def generate_from_music(self, genre='electronic'):
        # Map music genres to anime themes
        genre_themes = {
            'electronic': 'cyberpunk neon city, holographic effects',
            'rock': 'intense action scene, dynamic poses',
            'pop': 'colorful magical girl transformation',
            'classical': 'elegant fantasy ballroom scene',
            'hip-hop': 'urban street art graffiti style'
        }
        
        theme = genre_themes.get(genre.lower(), 'dynamic anime scene')
        
        # Create anime generation request
        payload = {
            'prompt': f'anime character in {theme}',
            'music_theme': genre,
            'style': 'high quality anime',
            'character': 'main protagonist'
        }
        
        try:
            response = requests.post(f'{self.anime_api}/api/generate', json=payload)
            return response.json()
        except:
            return {'error': 'Service not available'}
    
    def test_integration(self):
        print('Testing Music → Anime Integration:')
        result = self.generate_from_music('electronic')
        print(f'Result: {json.dumps(result, indent=2)}')
        return result

# Test it
integration = MusicAnimeIntegration()
integration.test_integration()
