import cv2
import os
from ultralytics import YOLO

# Load the YOLO model
model = YOLO(r"D:\Kuntal\project_idea\my_Projects\Car Plate Detection\runs\detect\train\weights\best.pt")

# Load a single image
image_path = r"D:\Kuntal\project_idea\my_Projects\Car Plate Detection\Dataset\test\images\426_png.rf.1fd68a3202b66810d76fd5c04d58307a.jpg"
image = cv2.imread(image_path)

# Run YOLO inference on the image
results = model(image)

# Visualize the results
annotated_image = results[0].plot()

# Create an bounding_outputs folder and a subfolder for cropped bounding boxes
output_path = r"D:\Kuntal\project_idea\my_Projects\Car Plate Detection\bounding_outputs"
cropped_bounding_folder = os.path.join(output_path, "cropped_bounding")
os.makedirs(cropped_bounding_folder, exist_ok=True)

# Save the annotated image
output_file = os.path.join(output_path, "annotated_result2.jpg")
cv2.imwrite(output_file, annotated_image)
print(f"Annotated result saved to {output_file}")

# Crop and save bounding boxes detected by the model
for i, box in enumerate(results[0].boxes.xyxy):  # Extract bounding box coordinates
    x_min, y_min, x_max, y_max = map(int, box)  # Convert to integers
    cropped_image = image[y_min:y_max, x_min:x_max]  # Crop the region
    i+=1
    cropped_file = os.path.join(cropped_bounding_folder, f"cropped_1.jpg")
    cv2.imwrite(cropped_file, cropped_image)
    print(f"Cropped bounding box saved to {cropped_file}")
