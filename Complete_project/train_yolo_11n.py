from ultralytics import YOLO

# Load a model
model = YOLO(r"E:\Kuntal\project_idea\my_Projects\Car Plate Detection\yolo11n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data= "E:/Kuntal/project_idea/my_Projects/Car Plate Detection/data.yaml", epochs=50)