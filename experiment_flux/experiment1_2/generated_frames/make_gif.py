from PIL import Image
import os

# === Config ===
frames_dir = "./zoom_sail"   # folder containing frames
output_gif = "zoom_sail.gif"       # name of the gif to save
frame_duration = 1000             # duration per frame in ms (lower = faster)

frames = sorted(
    [os.path.join(frames_dir, f) for f in os.listdir(frames_dir)
     if f.lower().endswith((".png", ".jpg", ".jpeg"))]
)

# Open all frames
images = [Image.open(f) for f in frames]

# === Save as GIF ===
images[0].save(
    output_gif,
    save_all=True,
    append_images=images[1:],
    duration=frame_duration,
    loop=0  # 0 means loop forever
)

print(f"✅ GIF saved to {output_gif}")
