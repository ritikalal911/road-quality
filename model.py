import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the YOLOv8 model
@st.cache_resource
def load_model():
    return YOLO('yolov8.pt')

model = load_model()

# Streamlit app
st.title('Potholes and Cracks Detection in Road')

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Convert image to 3 channels (RGB) to handle 4-channel images
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)

    if st.button('Detect Cracks'):
        # Convert PIL Image to numpy array
        image_np = np.array(image)

        # Perform inference
        results = model(image_np)

        # Get the names of the detected classes
        names = model.names

        # Initialize flags for crack types
        pothole_detected = False
        alligator_detected = False
        traversal_detected = False
        longitudinal_detected = False

        # Filter detections by confidence >= 80%
        filtered_boxes = []
        crack_classes = ["pothole", "alligator", "traversal", "longitudinal"]  # Define the crack-related classes

        for result in results:
            boxes = result.boxes
            filtered_result_boxes = []
            for box in boxes:
                cls = int(box.cls[0])
                conf = box.conf[0]  # Confidence score
                if names[cls] in crack_classes and conf >= 0.50:
                    filtered_result_boxes.append(box)
                    if names[cls] == "pothole":
                        pothole_detected = True
                    elif names[cls] == "alligator":
                        alligator_detected = True
                    elif names[cls] == "traversal":
                        traversal_detected = True
                    elif names[cls] == "longitudinal":
                        longitudinal_detected = True
            if filtered_result_boxes:
                filtered_boxes.extend(filtered_result_boxes)

        # Display detection summary
        st.subheader('Detection Summary')

        if pothole_detected or alligator_detected or traversal_detected or longitudinal_detected:
            # Construct the message
            detection_message = "Detected: "
            if pothole_detected:
                detection_message += "Pothole "
            if alligator_detected:
                detection_message += "Alligator Crack "
            if traversal_detected:
                detection_message += "Traversal Crack "
            if longitudinal_detected:
                detection_message += "Longitudinal Crack "

            st.error(detection_message.strip())

            # Plot the image with all filtered bounding boxes
            results[0].boxes = filtered_boxes
            plot_image = results[0].plot()  # This draws all filtered boxes on the image
            st.image(plot_image, caption='Detected Cracks and Potholes', use_column_width=True)
        else:
            st.success("No pothole or crack detected in the image.")
