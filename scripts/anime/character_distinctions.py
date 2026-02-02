#!/usr/bin/env python3
"""
Character visual distinctions for Tokyo Debt Desire
Ensures each character looks unique
"""

CHARACTER_VISUALS = {
    "tokyo_debt_desire": {
        "Mei_Kobayashi": {
            "age": "24 years old",
            "hair": "shoulder-length black hair with side-swept bangs",
            "face": "soft round face, gentle features, warm smile",
            "body": "petite build, 5'3\", modest curves",
            "style": "casual homewear, apron, simple dress",
            "distinguishing": "beauty mark under left eye, dimples when smiling"
        },
        "Rina_Suzuki": {
            "age": "28 years old",
            "hair": "straight black hair in professional bob cut",
            "face": "sharp angular features, high cheekbones, serious expression",
            "body": "tall athletic build, 5'7\", professional posture",
            "style": "business suit, blazer, pencil skirt, heels",
            "distinguishing": "wears glasses, red lipstick, small ear studs"
        },
        "Yuki_Tanaka": {
            "age": "26 years old",
            "hair": "short messy black hair",
            "face": "nervous expression, masculine jawline, worried eyes",
            "body": "average male build, 5'9\", slightly thin",
            "style": "casual t-shirt, jeans, sometimes hoodie",
            "distinguishing": "dark circles under eyes, slight stubble"
        },
        "Takeshi_Sato": {
            "age": "45 years old",
            "hair": "slicked back black hair with gray streaks",
            "face": "stern expression, sharp eyes, strong jaw",
            "body": "muscular intimidating build, 6'1\", broad shoulders",
            "style": "expensive dark suit, gold watch, dress shoes",
            "distinguishing": "scar on left cheek, always serious expression"
        }
    }
}

def get_character_prompt(project: str, character: str) -> str:
    """Get detailed visual prompt for character"""
    char = CHARACTER_VISUALS.get(project, {}).get(character)
    if not char:
        return ""

    return (f"{char['age']}, {char['hair']}, {char['face']}, "
            f"{char['body']}, {char['style']}, {char['distinguishing']}")

def get_distinguishing_features(character: str) -> str:
    """Get only the unique features to emphasize"""
    if character == "Mei_Kobayashi":
        return "beauty mark under left eye, dimples, soft round face, shoulder-length hair"
    elif character == "Rina_Suzuki":
        return "wearing glasses, sharp angular face, bob haircut, red lipstick, professional suit"
    elif character == "Yuki_Tanaka":
        return "short messy hair, nervous expression, dark circles, slight stubble"
    elif character == "Takeshi_Sato":
        return "scar on left cheek, slicked back hair, gray streaks, intimidating"
    return ""