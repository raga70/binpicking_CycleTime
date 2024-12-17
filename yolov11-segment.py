# region Setup Environment
from ultralytics import YOLO


# nvidia-smi --> if CUDA > 12.4 (also check if CUDA Toolkit is installed)
# then run this command --> pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
# this has been tested on 16/12/2024
# if some error occurs, you might need to go to this link --> https://pytorch.org/get-started/locally/

## --> uncomment this if you are setting up the environment for the first time
# import torch
# print(torch.cuda.is_available()) # check if CUDA is available

# import ultralytics as ul
# ul.checks()
# endregion Setup Environment

# region Train Model
# model = YOLO("models\\InTheBinRGB1.pt") # Load segmentation model
# results = model.train(data="datasets\\InTheBinRGB-4\\data.yaml", imgsz=[1920, 1200], epochs=300, batch_size=8, workers=8, device="0", evolve=True, project="runs/train", name="InTheBinRGB")
# endregion Train Model

# region Validate Model
# model = YOLO("models\\InTheBinRGB1.pt") # Load segmentation model
# my_data = "datasets\\InTheBinRGB-1\\data.yaml"
# metrics = model.val(data=my_data, plots=True, device=0, save_json=True)  # no arguments needed, dataset and settings remembered
# metrics.box.map  # map50-95
# metrics.box.map50  # map50
# metrics.box.map75  # map75
# metrics.box.maps  # a list contains map50-95 of each category
# reference https://docs.ultralytics.com/modes/val/
# endregion Validate Model

# region Predict Testing 2   
# import os
# import time
# import csv
# Load the segmentation model
# model = YOLO("models\\InTheBinRGB1.pt")

# # Define the root dataset folder and output CSV file
# dataset_root = r"C:\Users\moust\OneDrive\Documents\GitHub\binpicking_CycleTime\datasets\Testing 2"
# output_csv = "processing_time_results.csv"

# # Image names to process
# images = [
#     "rgb_image_00000_0.5.png", "rgb_image_00000_0.25.png", "rgb_image_00000.png",
#     "rgb_image_00001_0.5.png", "rgb_image_00001_0.25.png", "rgb_image_00001.png",
#     "rgb_image_00002_0.5.png", "rgb_image_00002_0.25.png", "rgb_image_00002.png"
# ]

# # List to store processing time results
# measurements = []

# # Iterate over each folder
# for folder in os.listdir(dataset_root):
#     folder_path = os.path.join(dataset_root, folder)
#     if os.path.isdir(folder_path):  # Process only directories
#         print(f"Processing folder: {folder}")
#         for image_name in images:
#             image_path = os.path.join(folder_path, image_name)
#             if os.path.exists(image_path):  # Check if the image exists
#                 try:
#                     # Run model prediction and save the processed image
#                     results = model.predict(source=image_path, save=True)

#                     preprocess_time = results[0].speed['preprocess']  # Preprocessing time in seconds
                    
#                     inference_time = results[0].speed['inference']     # Model inference time in seconds
                    
#                     postprocess_time = results[0].speed['postprocess']  # Postprocessing time in seconds

#                     total_time = preprocess_time + inference_time + postprocess_time
#                     # Save results
#                     measurements.append([folder, image_name, f"{total_time:.4f}", f"{preprocess_time:.4f}", f"{inference_time:.4f}", f"{postprocess_time:.4f}"])
#                     print(f"Processed {image_name} in folder {folder}")
#                     print(f"Preprocess time: {preprocess_time} ms")
#                     print(f"Inference time: {inference_time} ms")
#                     print(f"Postprocess time: {postprocess_time} ms")
#                     # print(f"Inference time: {results}")
#                 except Exception as e:
#                     print(f"Error processing {image_name} in folder {folder}: {str(e)}")

# # Write results to a CSV file
# with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
#     writer = csv.writer(file)
#     writer.writerow(["Folder", "Image Name", "Total Time (ms)", "Preprocess Time (ms)", "Inference Time (ms)", "Postprocess Time (ms)"])
#     writer.writerows(measurements)

# print(f"Processing times have been saved to {output_csv}.")

# endregion Test Model

#region Predict General
# Load the segmentation model
model = YOLO("models\\InTheBinRGB1.pt")

# Define the root dataset folder
dataset = r"C:\Users\moust\OneDrive\Documents\GitHub\binpicking_CycleTime\datasets\InTheBinRGB-4\test\images"

# Run model prediction and save the processed image
results = model.predict(source=dataset, save=True)



# endregion Predict General