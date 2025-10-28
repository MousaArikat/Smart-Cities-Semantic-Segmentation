from PIL import Image, ImageDraw, ImageFont
import json
import os

with open("assets/colors.json", "r") as f:
    color_map = json.load(f)
color_map = {int(k): tuple(v) for k, v in color_map.items()}

class_names = [
    "Road", "Sidewalk", "Building", "Wall", "Fence", "Pole",
    "Traffic Light", "Traffic Sign", "Vegetation", "Terrain",
    "Sky", "Person", "Rider", "Car", "Truck", "Bus", "Train",
    "Motorcycle", "Bicycle"
]

w, h = 400, 30 * len(class_names)
legend = Image.new("RGB", (w, h), (255, 255, 255))
draw = ImageDraw.Draw(legend)

try:
    font = ImageFont.truetype("arial.ttf", 16)
except:
    font = ImageFont.load_default()

for i, name in enumerate(class_names):
    y = i * 30
    draw.rectangle([10, y + 5, 30, y + 25], fill=color_map[i])
    draw.text((40, y + 5), f"{i}: {name}", fill=(0, 0, 0), font=font)

legend.save("assets/legend.png")
print("✅ Class legend image saved at assets/legend.png")