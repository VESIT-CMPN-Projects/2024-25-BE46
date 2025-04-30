import os
import json
from PIL import Image, ImageEnhance
import numpy as np
import tensorflow as tf
import streamlit as st

# --- Paths ---
working_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(working_dir, "trained_model", "tomato_leaf_cnn_prediction_model.h5")
class_indices_path = os.path.join(working_dir, "class_indices.json")

# --- Load model and class indices ---
model = tf.keras.models.load_model(model_path)
class_indices = json.load(open(class_indices_path))
index_to_class = class_indices  # Assuming no reversal needed

# --- Disease Guidance ---
disease_guidance = {
    "Tomato___Bacterial_spot": {
        "disease" :"small, dark, often water-soaked spots on leaves, stems, and fruit.",
        "prevention": "Avoid overhead watering, use disease-free seeds, apply copper-based bactericides, and ensure rainfall does not exceed 15 mm/week.",
        "removal": "Remove infected leaves, use copper sprays, clean tools, and avoid working with wet plants."
    },
    "Tomato___Early_blight": {
        "disease" :"dark brown to black spots with concentric rings, often described as a bullseye pattern",
        "prevention": "Rotate crops, remove infected debris, avoid rain accumulation beyond 20 mm/week, and use fungicides.",
        "removal": "Prune infected leaves, apply chlorothalonil/mancozeb, and maintain plant spacing."
    },
    "Tomato___Late_blight": {
        "disease": "dark, water-soaked lesions on leaves, stems, and fruit",
        "prevention": "Use resistant varieties, avoid high rainfall periods (>10 mm/day), ensure proper airflow, and apply appropriate fungicides.",
        "removal": "Destroy infected plants, apply fungicides immediately, and avoid wetting foliage."
    },
    "Tomato___Leaf_Mold": {
        "disease":"pale green to yellowish spots on the upper leaf surface that eventually turn yellow.",
        "prevention": "Increase ventilation, reduce humidity, keep rainfall under 12 mm/week in greenhouses, and use fungicides like chlorothalonil.",
        "removal": "Improve airflow, remove infected leaves, and apply sulfur-based fungicides."
    },
    "Tomato___Septoria_leaf_spot": {
        "disease": "circular, tan to gray spots with dark brown margins, often with tiny black dots in the center",
        "prevention": "Remove affected leaves, avoid wetting foliage, maintain rainfall under 18 mm/week, and use preventive fungicides.",
        "removal": "Remove infected leaves, apply fungicides (chlorothalonil), and rotate crops."
    },
    "Tomato___Spider_mites Two-spotted_spider_mite": {
        "disease": "tiny, barely visible with the naked eye, with two dark spots on their body",
        "prevention": "Use insecticidal soap, maintain humidity, and ensure low rainfall to prevent fungal misdiagnosis.",
        "removal": "Spray with neem oil, isolate infected plants, and increase humidity."
    },
    "Tomato___Target_Spot": {
        "disease": "small, dark spots on leaves that enlarge to form concentric, light brown lesions with dark margins and a yellow halo",
        "prevention": "Remove infected debris, keep rainfall < 15 mm/week, eliminate weed hosts, water at base, and improve spacing.",
        "removal": "Burn infected leaves, water early at the base, prune lower leaves, apply fungicide, and rotate crops next season."
    },
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": {
        "disease": "an overall crumpled appearance of the leaves, and upward and downward leaf-curling.",
        "prevention": "Control whiteflies, use resistant varieties, and reduce rainfall-induced vector activity.",
        "removal": "Remove infected plants immediately, and control whiteflies with traps/insecticides."
    },
    "Tomato___Tomato_mosaic_virus": {
        "disease": "a mottled or mosaic pattern on the leaves, with alternating light and dark green patches or streaks",
        "prevention": "Avoid tobacco handling, disinfect tools, use resistant plants, and manage rainfall to limit stress.",
        "removal": "Remove and destroy infected plants, disinfect tools, and avoid spreading."
    },
    "Tomato___healthy": {
        "disease": "firm, heavy for its size, blemish-free, and has a strong, pleasant aroma",
        "prevention": "Maintain hygiene, balance watering, and avoid rainfall above 20 mm/week.",
        "removal": " No disease found! Keep practicing good care."
    }
}

# --- Preprocessing function ---
def load_and_preprocess_image(img, target_size=(224, 224)):
    img = Image.open(img).convert("RGB")
    img = ImageEnhance.Brightness(img).enhance(1.1)
    img = ImageEnhance.Contrast(img).enhance(1.2)
    img = img.resize(target_size)
    img_array = np.array(img).astype('float32') / 255.0
    img_array = (img_array - np.mean(img_array)) / (np.std(img_array) + 1e-7)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- Prediction function ---
def predict_image_class(model, img, index_to_class):
    preprocessed_img = load_and_preprocess_image(img)
    predictions = model.predict(preprocessed_img)
    predicted_index = np.argmax(predictions, axis=1)[0]
    confidence = np.max(predictions)
    predicted_class = index_to_class[str(predicted_index)]
    return predicted_class, confidence

# --- Streamlit App ---
st.set_page_config(page_title="Tomato Leaf Disease Detection", page_icon="🍅", layout="centered")

st.title("🍅 Tomato Leaf Disease Detection App")
st.caption("Upload an image to detect leaf disease on RANDOM Tomato Leaf Image using a trained CNN model.")

# Upload image
uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image_display = Image.open(uploaded_image)
    st.image(image_display, caption="Uploaded Image", use_column_width=True)

    if st.button('Predict'):
        with st.spinner("Model is predicting..."):
            prediction, confidence = predict_image_class(model, uploaded_image, index_to_class)

        st.success(f"Prediction: **{prediction}**")
        st.info(f"Confidence: **{confidence:.2f}**")

        tips = disease_guidance.get(prediction, {"prevention": "No tips available.", "removal": "No removal steps found."})
        st.info(f"Disease Information:** {tips['prevention']}")
        st.info(f"**Prevention Tip:** {tips['prevention']}")
        st.warning(f"**After Occurrence - Removal Tip:** {tips['removal']}")

else:
    st.warning("No image uploaded. Please upload an image!")
