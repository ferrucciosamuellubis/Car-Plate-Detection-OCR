

https://github.com/user-attachments/assets/7c9ac2ab-a9ab-418e-a9e3-469acf15df1d



# Car Plate Detection and OCR Extraction

A Streamlit-based application that detects car plates using YOLO and extracts text from the plates using PaddleOCR.

## Features
- **YOLO-based car plate detection**: Detects number plates from images.
- **OCR extraction**: Reads text from detected plates using PaddleOCR.
- **Annotated image display**: Shows processed images with bounding boxes.
- **Text overlay on images**: Displays extracted text on images.
- **Results export**: Saves OCR results as a CSV file.

## Installation
Ensure you have Python installed, then install dependencies:

```bash
pip install streamlit opencv-python pandas ultralytics paddleocr numpy
```

## Usage
Run the Streamlit app:

```bash
streamlit run app.py
```

### Steps:
1. Upload the YOLO model (`.pt` file).
2. Provide the image folder path.
3. Process images to detect plates and extract text.
4. View annotated images and extracted text.
5. Export results as a CSV file.

## Project Structure
```
├── app.py                    # Streamlit app
├── result
│   ├── annotation_output      # Annotated images
│   ├── cropped_img_folder     # Cropped plate images
│   ├── result_folder          # Processed results
│   ├── final_output_folder    # Final output images
│   ├── ocr_results.csv        # Extracted text results
```

## Example Output
Annotated images with bounding boxes:

![Annotated Example](example_annotated.jpg)

Extracted text overlaid on image:

![Final Output Example](example_final.jpg)

## Contributing
Feel free to fork, improve, and submit pull requests!

## License
This project is licensed under the MIT License.

---

This README should help users understand how to install, use, and contribute to your project! Would you like any modifications? 🚀
