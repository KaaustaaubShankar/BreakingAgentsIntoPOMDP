import os

replacements = [
    ("Visual Zendo", "Lithic Array"),
    ("VisualZendoEnv", "LithicArrayEnv"),
    ("Master", "Basalt"),
    ("master", "basalt"),
    ("Mondo", "Strata"),
    ("mondo", "strata"),
    ("MONDO", "STRATA"),
    ("Harmonious", "Quartz"),
    ("harmonious", "quartz"),
    ("h_count", "q_count"),
    ("Discordant", "Shale"),
    ("discordant", "shale"),
    ("d_count", "s_count")
]

files = [
    "rules.py",
    "visual_zendo.py",
    "test_agent.py",
    "ablation_study.py"
]

for file in files:
    filepath = os.path.join(".", file)
    with open(filepath, "r") as f:
        content = f.read()
    
    for old, new in replacements:
        content = content.replace(old, new)
        
    with open(filepath, "w") as f:
        f.write(content)

print("Renaming complete!")
