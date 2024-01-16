from ultralytics import YOLO

# Load a model
#model = YOLO("yolov8n.yaml")  # build a new model from scratch
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Use the model
results = model.train(data="config.yaml", epochs=2)  # train the model
metrics = model.val()  # evaluate model performance on the validation set 
print(metrics)
results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
print(results)
path = model.export(format="onnx")  # export the model to ONNX format
print(metrics)
