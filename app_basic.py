import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from database import get_plant_info

# --- 1. CONFIGURATION ---
MODEL_PATH = r'C:\Users\SBC-IT\PycharmProjects\Plant-tf-test\apple_model_final.tflite'
IMG_SIZE = (128, 128)
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy']

# --- 2. PROCESSING FUNCTIONS ---

@st.cache_resource
def load_tflite_model(model_path):
    """Loads the TFLite model and initializes the interpreter."""
    try:
        interpreter = tf.lite.Interpreter(model_path=model_path)
        interpreter.allocate_tensors()
        return interpreter
    except Exception as e:
        st.error(f"Error loading TFLite model: {e}")
        return None


def predict_with_tflite(interpreter, image):
    """Takes an interpreter and image, and returns the prediction results."""
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    img = image.resize(IMG_SIZE)
    img_array = np.array(img, dtype=np.float32) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    predicted_class_index = np.argmax(output_data)
    confidence = float(np.max(output_data))
    predicted_class_name = CLASS_NAMES[predicted_class_index]

    return predicted_class_name, confidence


# --- 3. STREAMLIT APPLICATION UI ---

st.set_page_config(page_title="Apple Disease Diagnosis", layout="wide")

st.title("🍎 Apple Leaf Disease Diagnosis Assistant")
st.markdown("Upload one or more images of apple leaves for the system to analyze and diagnose automatically.")

# Load the model
interpreter = load_tflite_model(MODEL_PATH)

if interpreter is not None:
    # Allow uploading multiple files at once
    uploaded_files = st.file_uploader(
        "Choose one or more image files...",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        # Create a suitable number of columns, e.g., 3 columns
        num_columns = 3
        cols = st.columns(num_columns)

        # Process and display each uploaded file
        for index, uploaded_file in enumerate(uploaded_files):
            # Determine the column to display the current image in
            col = cols[index % num_columns]

            with col:
                st.subheader(f"Results for: {uploaded_file.name}")
                image = Image.open(uploaded_file)

                st.image(image, caption='Uploaded Image.', use_container_width=True)

                with st.spinner('Analyzing...'):
                    predicted_label, confidence = predict_with_tflite(interpreter, image)

                # Extract the disease name from the predicted label
                # disease_name = predicted_label.split('___')[1].replace('_', ' ')

                st.success(f"**Diagnosis:** {predicted_label}")
                st.info(f"**Confidence:** {confidence:.2%}")

                # --- DATABASE INTEGRATION ---
                plant_info = get_plant_info(predicted_label)
                if plant_info:
                    with st.expander("View more information about the disease"):
                        st.markdown(f"**Description:** {plant_info[0]}")
                        st.markdown(f"**Treatment:** {plant_info[1]}")
                else:
                    st.warning("No detailed information found for this disease in the database.")
                st.markdown("---")

else:
    st.error("Could not load the model. Please check the file path in the MODEL_PATH variable.")