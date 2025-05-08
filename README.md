Car Plate Detection & OCR

📌 Project Overview
This project uses YOLO (You Only Look Once) for car plate detection and PaddleOCR for optical character recognition (OCR) to extract text from detected license plates. It provides a Streamlit-based web interface for users to upload images, detect plates, and extract text.

🔥 Features
🚗 Car Plate Detection using YOLOv8 model.

🔍 OCR Processing using PaddleOCR.

🖼 Annotated & Cropped Image Saving.

🌐 Streamlit Web App for user interaction.

📄 CSV Output of extracted text.

🛠 Installation & Setup
1️⃣ Clone the Repository
bash
git clone https://github.com/yourusername/car-plate-detection.git
cd car-plate-detection
2️⃣ Create Virtual Environment
bash
python -m venv myenv
source myenv/bin/activate  # On macOS/Linux
myenv\Scripts\activate  # On Windows
3️⃣ Install Dependencies
bash
pip install -r requirements.txt
4️⃣ Download YOLO Model
Download your trained YOLO model best.pt and place it in the project directory.

🚀 Running the Application
Run the Streamlit web app:

bash
streamlit run app.py
Once started, open the localhost link displayed in the terminal and upload an image for plate detection.

📂 Project Structure
car-plate-detection/
│── best.pt                # YOLO model file
│── requirements.txt        # Dependencies
│── app.py                  # Streamlit Web App
│── Sample_Images/         # Test images for detection
│── results/
│   ├── annotation_output/   # YOLO detection annotations
│   ├── cropped_img_folder/  # Cropped plates
│   ├── result_folder/       # OCR results
│   ├── final_output_folder/ # Final processed images
└── README.md               # Project Documentation
⚡ Example Output
Image Name	Extracted Text
car1.jpg	WB04A1234
car2.jpg	MH12AB3456
💡 Future Improvements
Enhance OCR accuracy with better preprocessing.

Add multiple language support for license plates.

Integrate a database to store vehicle details.

🏆 Credits & Acknowledgments
YOLOv8 for car plate detection.

PaddleOCR for text recognition.

Streamlit for the web interface.
