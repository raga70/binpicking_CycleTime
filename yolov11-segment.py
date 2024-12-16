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
if __name__ == '__main__':
	model = YOLO("models\\InTheBinRGB1.pt") # Load segmentation model
	my_data = "datasets\\InTheBinRGB-1\\data.yaml"
	metrics = model.val(data=my_data, plots=True, device=0)  # no arguments needed, dataset and settings remembered
	metrics.box.map  # map50-95
	metrics.box.map50  # map50
	metrics.box.map75  # map75
	metrics.box.maps  # a list contains map50-95 of each category
	# reference https://docs.ultralytics.com/modes/val/
# endregion Validate Model


