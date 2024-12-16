import os
from PIL import Image

def downscale_image(image_path, scale):
    with Image.open(image_path) as img:
        new_size = (int(img.width * scale), int(img.height * scale))
        downscaled_img = img.resize(new_size, Image.LANCZOS)
        return downscaled_img

def process_images(folder):
    for root, _, files in os.walk(folder):
        for file in files:
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                file_path = os.path.join(root, file)
                base, ext = os.path.splitext(file_path)
                
                # Downscale to 50%
                downscaled_img_50 = downscale_image(file_path, 0.5)
                downscaled_img_50.save(f"{base}_0.5{ext}")
                
                # Downscale to 25%
                downscaled_img_25 = downscale_image(file_path, 0.25)
                downscaled_img_25.save(f"{base}_0.25{ext}")

if __name__ == "__main__":
    folder_path = input("Enter the folder path: ")
    process_images(folder_path)