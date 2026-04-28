"""
instructions.py
Waste category mapping, disposal instructions, and bin color logic.
India-friendly waste segregation (Blue = Dry, Green = Wet).
"""

# ── Category → Bin + Instructions mapping ────────────────────────────────────

WASTE_INSTRUCTIONS = {
    "plastic": {
        "bin": "Blue Bin (Dry Waste)",
        "bin_color": "#2563EB",
        "emoji": "♻️",
        "steps": [
            "Rinse the item to remove food residue.",
            "Crush or flatten to save space.",
            "Place in the Blue Dry Waste bin.",
            "Avoid mixing with wet/food waste.",
        ],
        "tip": "Plastics marked 1–7 are recyclable. Check the recycling symbol on the bottom.",
        "hazard": False,
    },
    "paper": {
        "bin": "Blue Bin (Dry Waste)",
        "bin_color": "#2563EB",
        "emoji": "📄",
        "steps": [
            "Keep paper dry — wet paper cannot be recycled.",
            "Remove staples, tape, or plastic coatings if possible.",
            "Stack or fold neatly and place in the Blue Dry Waste bin.",
        ],
        "tip": "Newspapers, cardboard, and office paper are all recyclable. Tissue paper is NOT.",
        "hazard": False,
    },
    "metal": {
        "bin": "Blue Bin (Dry Waste)",
        "bin_color": "#2563EB",
        "emoji": "🔩",
        "steps": [
            "Rinse cans and tins to remove food residue.",
            "Flatten if possible to save space.",
            "Place in the Blue Dry Waste bin.",
            "Sharp metal edges — handle with care!",
        ],
        "tip": "Aluminium cans are infinitely recyclable. Steel and iron are also accepted.",
        "hazard": True,
    },
    "glass": {
        "bin": "Blue Bin (Dry Waste)",
        "bin_color": "#2563EB",
        "emoji": "🍶",
        "steps": [
            "Handle with extreme care — do NOT break.",
            "Rinse bottles and jars.",
            "Do NOT mix broken glass with regular recycling.",
            "Place intact glass in the Blue Dry Waste bin.",
            "Wrap broken glass in newspaper before disposing.",
        ],
        "tip": "Glass is 100% recyclable indefinitely. Avoid mixing coloured and clear glass.",
        "hazard": True,
    },
    "organic / food": {
        "bin": "Green Bin (Wet Waste)",
        "bin_color": "#16A34A",
        "emoji": "🌿",
        "steps": [
            "Place raw/cooked food scraps in the Green Wet Waste bin.",
            "Drain excess liquids before disposal.",
            "Include vegetable peels, fruit rinds, and eggshells.",
            "Consider home composting to reduce landfill load.",
        ],
        "tip": "Wet waste can be composted into nutrient-rich manure within 30–45 days.",
        "hazard": False,
    },
    "e-waste": {
        "bin": "Special E-Waste Collection",
        "bin_color": "#DC2626",
        "emoji": "💻",
        "steps": [
            "Do NOT place in regular bins.",
            "Take to the nearest authorised e-waste collection centre.",
            "Many brands offer take-back programmes — check the manufacturer's website.",
            "Remove personal data from devices before disposal.",
        ],
        "tip": "E-waste contains toxic metals like lead and mercury. Never burn or bury it.",
        "hazard": True,
    },
    "unknown": {
        "bin": "Unsure — Check local guidelines",
        "bin_color": "#6B7280",
        "emoji": "❓",
        "steps": [
            "Could not confidently identify the waste type.",
            "Check your local municipal waste guidelines.",
            "When in doubt, keep it separate and ask.",
        ],
        "tip": "Contamination ruins entire batches of recyclables. When unsure, keep separate.",
        "hazard": False,
    },
}

# ── ImageNet label → waste category mapping ──────────────────────────────────
# MobileNetV2 is trained on ImageNet; these are common label substrings
# that correspond to each waste category.

IMAGENET_TO_WASTE = {
    # ── Plastic ──────────────────────────────────────────────────────────────
    "water bottle": "plastic",
    "plastic bag": "plastic",
    "bottle": "plastic",
    "pop bottle": "plastic",
    "bucket": "plastic",
    "barrel": "plastic",
    "milk can": "plastic",
    "plastic": "plastic",
    "nylon": "plastic",
    "tray": "plastic",

    # ── Paper ─────────────────────────────────────────────────────────────────
    "book": "paper",
    "newspaper": "paper",
    "paper towel": "paper",
    "envelope": "paper",
    "comic book": "paper",
    "cardboard": "paper",
    "carton": "paper",
    "tissue": "paper",
    "notebook": "paper",
    "menu": "paper",
    "magazine": "paper",

    # ── Metal ─────────────────────────────────────────────────────────────────
    "can opener": "metal",
    "tin can": "metal",
    "steel": "metal",
    "iron": "metal",
    "wrench": "metal",
    "screwdriver": "metal",
    "hammer": "metal",
    "nail": "metal",
    "fork": "metal",
    "spoon": "metal",
    "knife": "metal",
    "spatula": "metal",
    "ladle": "metal",
    "scissors": "metal",
    "chain": "metal",
    "key": "metal",
    "padlock": "metal",
    "coin": "metal",
    "safety pin": "metal",
    "can": "metal",

    # ── Glass ─────────────────────────────────────────────────────────────────
    "wine bottle": "glass",
    "beer bottle": "glass",
    "glass": "glass",
    "goblet": "glass",
    "beer glass": "glass",
    "whiskey jug": "glass",
    "vase": "glass",
    "jar": "glass",
    "jug": "glass",
    "pitcher": "glass",
    "mirror": "glass",
    "lens": "glass",

    # ── Organic / Food ────────────────────────────────────────────────────────
    "banana": "organic / food",
    "apple": "organic / food",
    "orange": "organic / food",
    "lemon": "organic / food",
    "strawberry": "organic / food",
    "pineapple": "organic / food",
    "mango": "organic / food",
    "fig": "organic / food",
    "coconut": "organic / food",
    "cucumber": "organic / food",
    "broccoli": "organic / food",
    "carrot": "organic / food",
    "corn": "organic / food",
    "mushroom": "organic / food",
    "pizza": "organic / food",
    "hotdog": "organic / food",
    "burger": "organic / food",
    "sandwich": "organic / food",
    "pretzel": "organic / food",
    "bagel": "organic / food",
    "bread": "organic / food",
    "cheese": "organic / food",
    "egg": "organic / food",
    "meat": "organic / food",
    "chicken": "organic / food",
    "fish": "organic / food",
    "sushi": "organic / food",
    "food": "organic / food",
    "fruit": "organic / food",
    "vegetable": "organic / food",

    # ── E-Waste ───────────────────────────────────────────────────────────────
    "laptop": "e-waste",
    "computer keyboard": "e-waste",
    "desktop computer": "e-waste",
    "monitor": "e-waste",
    "screen": "e-waste",
    "mouse": "e-waste",
    "remote control": "e-waste",
    "mobile phone": "e-waste",
    "cell phone": "e-waste",
    "telephone": "e-waste",
    "television": "e-waste",
    "radio": "e-waste",
    "headphone": "e-waste",
    "speaker": "e-waste",
    "printer": "e-waste",
    "camera": "e-waste",
    "projector": "e-waste",
    "iPod": "e-waste",
    "electric fan": "e-waste",
    "hair dryer": "e-waste",
    "iron (for clothes)": "e-waste",
    "toaster": "e-waste",
    "microwave": "e-waste",
    "battery": "e-waste",
    "plug": "e-waste",
    "switch": "e-waste",
    "hard disc": "e-waste",
    "disk": "e-waste",
}


def map_label_to_waste(label: str) -> str:
    """
    Maps an ImageNet class label (string) to a waste category.
    Falls back to 'unknown' if no match is found.
    """
    label_lower = label.lower().replace("_", " ")
    for keyword, category in IMAGENET_TO_WASTE.items():
        if keyword in label_lower:
            return category
    return "unknown"


def get_instructions(category: str) -> dict:
    """Returns the full instruction dict for a waste category."""
    return WASTE_INSTRUCTIONS.get(category, WASTE_INSTRUCTIONS["unknown"])
