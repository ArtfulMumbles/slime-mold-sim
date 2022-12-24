import glob

from PIL import Image

def make_gif(frame_folder):
    frames = [Image.open(image) for image in glob.glob(f"{frame_folder}/*.JPG")]
    frame_one = frames[0]
    frame_one.save("gif_name.gif", format="GIF", append_images=frames,
                    save_all=True, duration=100, loop=0)
    