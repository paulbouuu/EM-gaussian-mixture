import os
import re
import glob
from PIL import Image

def get_iteration_index(filename):
    """Extract the numeric iteration from a filename like 'MoG_12.png'."""
    match = re.search(r'MoG_(\d+)\.png$', filename)
    if not match:
        raise ValueError(f"Filename does not match pattern: {filename}")
    return int(match.group(1))

def create_gif(duration = 500):
    img_dir = 'images'
    pattern = os.path.join(img_dir, 'MoG_*.png')
    files = glob.glob(pattern)
    
    if not files:
        print(f"No files found matching {pattern}")
        return

    files_sorted = sorted(files, key=get_iteration_index)
    
    print(f"Found {len(files_sorted)} frames.")

    frames = []
    for f in files_sorted:
        try:
            img = Image.open(f)
            frames.append(img.convert('P', palette=Image.ADAPTIVE))
        except Exception as e:
            print(f"Warning: could not open {f}: {e}")

    if not frames:
        print("No valid images to combine. Exiting.")
        return

    # save as animated GIF
    output_path = 'gaussian_mixture.gif'
    try:
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,   # milliseconds per frame; adjust as needed
            loop=0,         # 0 = infinite loop
            optimize=True
        )
        print(f"Saved GIF to {output_path}")
    except Exception as e:
        print(f"Error writing GIF: {e}")
