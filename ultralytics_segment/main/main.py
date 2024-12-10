from ultralytics import YOLO

# # Load a model
# model = YOLO("models\yolo11n-seg.pt")  # load a pretrained model (recommended for training)

# # # Train the model
# results = model.train(data="ultralytics_segment\\datasets\\InTheBinRGB-1\\data.yaml", epochs=100, imgsz=640)
from ultralytics import YOLO

# Load a model
model = YOLO("models/InTheBinRGB1.pt", task="detect")

# Predict on images in a folder
results = model.predict(source="ultralytics_segment/datasets/InTheBinRGB-1/test/images")

for result in results:
    print(result.boxes.data)
    result.show()  # uncomment to view each result image