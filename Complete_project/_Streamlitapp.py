import streamlit as st
import cv2
import os
import pandas as pd
from ultralytics import YOLO
from paddleocr import PaddleOCR
import numpy as np

# ------------------- Streamlit Interface -------------------
st.title("Car Plate Detection and OCR Extraction")

# User Inputs
model_path = st.file_uploader("Upload YOLO Model File", type=["pt"])
img_folder = st.text_input("Enter path to the image folder")

# Ensure user has provided valid inputs
if model_path and img_folder:
    # Save model temporarily
    temp_model_path = "temp_model.pt"
    with open(temp_model_path, "wb") as f:
        f.write(model_path.getvalue())

    # Define output paths
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

    # Load YOLO model
    model = YOLO(temp_model_path)
    cropped_files = []  
    extracted_texts = {}  

    # Process images
    st.write("Processing Images...")
    for img_name in os.listdir(img_folder):
        img_path = os.path.join(img_folder, img_name)
        image = cv2.imread(img_path)

        results = model(image)

        # Annotate & display image
        annotated_image = results[0].plot()
        annotated_file = os.path.join(annotation_output, f"annotated_{img_name}")
        cv2.imwrite(annotated_file, annotated_image)

        # Display result in Streamlit
        st.image(annotated_image, caption=f"Annotated: {img_name}", channels="BGR")

        # Crop bounding boxes
        for i, box in enumerate(results[0].boxes.xyxy):
            x_min, y_min, x_max, y_max = map(int, box)
            cropped_image = image[y_min:y_max, x_min:x_max]
            cropped_file = os.path.join(cropped_img_folder, f"cropped_{img_name}_{i+1}.jpg")
            cv2.imwrite(cropped_file, cropped_image)
            cropped_files.append((img_name, cropped_file, box))

    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True)

    data = []
    for img_name, cropped_file, bbox in cropped_files:
        extracted_text = ""

        try:
            result = ocr.ocr(cropped_file, cls=True)
            if result and isinstance(result, list):
                extracted_text = " ".join([line[1][0] for res in result if res for line in res if line])
            else:
                extracted_text = ""
        except Exception as e:
            extracted_text = ""

        data.append([img_name, extracted_text])
        extracted_texts[img_name] = extracted_text

    # Save results as CSV
    df = pd.DataFrame(data, columns=["Image Name", "Extracted Text"])
    df.to_csv(save_path, index=False)
    
    # Display OCR Results
    st.write("OCR Results")
    st.dataframe(df)

    # Overlay text & display final image
    for img_name in extracted_texts.keys():
        img_path = os.path.join(img_folder, img_name)
        image = cv2.imread(img_path)

        if img_name in extracted_texts:
            text = extracted_texts[img_name]
            cv2.putText(image, text, (150, 150), cv2.FONT_HERSHEY_SIMPLEX, 5, (0, 0, 255), 10)

        final_output_file = os.path.join(final_output_folder, f"final_{img_name}")
        cv2.imwrite(final_output_file, image)

        st.image(image, caption=f"Final: {img_name}", channels="BGR")

    st.success("Processing Completed!")
