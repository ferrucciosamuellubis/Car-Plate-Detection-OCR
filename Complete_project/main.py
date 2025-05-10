import cv2
import os
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR

# ------------------- Define Paths -------------------
model_path = r"D:\Kuntal\project_idea\my_Projects\Car Plate Detection\Complete_project\best.pt"
img_folder = r"D:\Kuntal\project_idea\my_Projects\Car Plate Detection\Complete_project\Sample_Images"

# Automatically generate output paths
base_path = os.path.dirname(img_folder)
annotation_output = os.path.join(base_path, "result/annotation_output")
cropped_img_folder = os.path.join(base_path, "result/cropped_img_folder")
result_folder = os.path.join(base_path, "result/result_folder")
final_output_folder = os.path.join(base_path, "result/final_output_folder")
save_path = os.path.join(result_folder, "ocr_results.csv")

# Ensure directories exist
os.makedirs(annotation_output, exist_ok=True)
os.makedirs(cropped_img_folder, exist_ok=True)
os.makedirs(result_folder, exist_ok=True)
os.makedirs(final_output_folder, exist_ok=True)

# ------------------- YOLO Model Detection -------------------

# Load YOLO model
model = YOLO(model_path)
cropped_files = []  # To store paths of cropped images
extracted_texts = {}  # To map images to extracted texts

# Process each image in the folder
for img_name in os.listdir(img_folder):
    img_path = os.path.join(img_folder, img_name)
    image = cv2.imread(img_path)

    # Run YOLO inference on the image
    results = model(image)

    # Visualize results
    annotated_image = results[0].plot()
    annotated_file = os.path.join(annotation_output, f"annotated_{img_name}")
    cv2.imwrite(annotated_file, annotated_image)
    print(f"Annotated result saved to {annotated_file}")

    # Crop and save bounding boxes detected by the model
    for i, box in enumerate(results[0].boxes.xyxy):
        x_min, y_min, x_max, y_max = map(int, box)
        cropped_image = image[y_min:y_max, x_min:x_max]
        cropped_file = os.path.join(cropped_img_folder, f"cropped_{img_name}_{i+1}.jpg")
        cv2.imwrite(cropped_file, cropped_image)
        cropped_files.append((img_name, cropped_file, box))
        print(f"Cropped bounding box saved to {cropped_file}")

# ------------------- OCR Processing with PaddleOCR -------------------

# Initialize PaddleOCR
ocr = PaddleOCR(use_angle_cls=True)

# Prepare a list to store OCR results
data = []

# Process each cropped image
for img_name, cropped_file, bbox in cropped_files:
    extracted_text = ""

    # Perform OCR with error handling
    try:
        result = ocr.ocr(cropped_file, cls=True)

        # Ensure result is valid before processing
        if result and isinstance(result, list):
            extracted_text = " ".join([line[1][0] for res in result if res for line in res if line])
        else:
            extracted_text = ""  # Default to empty string if OCR fails

    except Exception as e:
        print(f"Error processing {img_name}: {e}")
        extracted_text = ""  # Store empty string for failed OCR attempts

    # Append to data list
    data.append([img_name, extracted_text])
    extracted_texts[img_name] = extracted_text  # Store text for overlaying later

# Convert to DataFrame and save as CSV
df = pd.DataFrame(data, columns=["Image Name", "Extracted Text"])
df.to_csv(save_path, index=False)

print(f"OCR processing completed! Results saved in '{save_path}'.")

# ------------------- Final Output Generation -------------------

for img_name in extracted_texts.keys():
    img_path = os.path.join(img_folder, img_name)
    image = cv2.imread(img_path)

    if img_name in extracted_texts:
        text = extracted_texts[img_name]

        # Overlay text at the top right corner
        cv2.putText(image, text, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)

    # Save final output image
    final_output_file = os.path.join(final_output_folder, f"final_{img_name}")
    cv2.imwrite(final_output_file, image)
    print(f"Final highlighted result saved to {final_output_file}")