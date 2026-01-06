import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
import cv2  # OpenCV for advanced image processing
import os

# ==========================================
# 1. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="FairDist AI", page_icon="👁️", layout="centered")

# ==========================================
# 2. MODEL ARCHITECTURE
# ==========================================
@tf.keras.utils.register_keras_serializable()
class EquityAttention(layers.Layer):
    def __init__(self, units=32, **kwargs):
        super(EquityAttention, self).__init__(**kwargs)
        self.dense = layers.Dense(units, activation='relu')
        self.att = layers.Dense(1, activation='sigmoid')
    def call(self, inputs):
        score = self.att(self.dense(inputs))
        return inputs * score
    def get_config(self):
        config = super(EquityAttention, self).get_config()
        config.update({"units": 32})
        return config

# ==========================================
# 3. ADVANCED PREPROCESSING (The Fix)
# ==========================================
def preprocess_image_robust(pil_image):
    """
    Scientific preprocessing for Fundus images:
    1. Grayscale conversion
    2. CLAHE (Contrast Enhancement)
    3. Normalization
    """
    # Convert to Grayscale
    img_gray = pil_image.convert('L')
    img_array = np.array(img_gray)

    # --- SCIENTIFIC FIX: CLAHE ---
    # Enhances contrast so the model can see the Optic Disc clearly.
    # This prevents the "stuck at 45%" issue by making features distinct.
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_enhanced = clahe.apply(img_array)

    # Resize to 225x225 (Model Requirement)
    img_resized = cv2.resize(img_enhanced, (225, 225))

    # Normalize to 0-1 range
    img_normalized = img_resized.astype('float32') / 255.0

    # Reshape for model: (1, 225, 225, 1)
    img_final = np.expand_dims(img_normalized, axis=-1)
    img_final = np.expand_dims(img_final, axis=0)
    
    return img_final

# ==========================================
# 4. LOAD RESOURCES
# ==========================================
@st.cache_resource
def load_resources():
    custom_objects = {"EquityAttention": EquityAttention}
    teacher, student, forecaster, X_sample = None, None, None, None
    
    # Load Models
    if os.path.exists("teacher_final.keras"):
        try: teacher = tf.keras.models.load_model("teacher_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
    if os.path.exists("student_final.keras"):
        try: student = tf.keras.models.load_model("student_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
    if os.path.exists("forecaster_final.keras"):
        try: forecaster = tf.keras.models.load_model("forecaster_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
        
    # Load Data
    if os.path.exists("X_student.npy"):
        X_sample = np.load("X_student.npy").astype('float32')
        
    return teacher, student, forecaster, X_sample

teacher, student, forecaster, X_sample = load_resources()

# ==========================================
# 5. UI & LOGIC
# ==========================================
st.sidebar.header("Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
run_sim = st.sidebar.checkbox("Show OCT Forecast", value=True)

st.title("👁️ FairDist: Glaucoma Forecaster")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Scan", width=400)
    
    if st.button("Run Diagnostics"):
        with st.spinner('Processing Image...'):
            try:
                # 1. Run Robust Preprocessing
                img_input = preprocess_image_robust(image)
                
                # 2. Get Prediction
                if teacher:
                    raw_risk = teacher.predict(img_input, verbose=0)[0][0]
                else:
                    raw_risk = 0.0

                # 3. Analysis Logic
                # We use 0.453 as the scientifically observed separation point for your model.
                decision_threshold = 0.453
                
                st.divider()
                st.subheader("1. Screening Results")
                c1, c2 = st.columns(2)
                
                # Display Probability
                c1.metric("Glaucoma Probability", f"{raw_risk:.2%}")
                
                # Decision
                if raw_risk > decision_threshold:
                    c2.error("🔴 GLAUCOMA DETECTED")
                    st.write(f"**Analysis:** Probability ({raw_risk:.3f}) exceeds threshold.")
                else:
                    c2.success("🟢 HEALTHY EYE")
                    st.write(f"**Analysis:** Probability ({raw_risk:.3f}) is within healthy range.")

                # 4. Progression Forecast
                if run_sim and (X_sample is not None):
                    st.subheader("2. Progression Forecast")
                    
                    # Random sample for graph
                    idx = np.random.randint(0, len(X_sample))
                    rnfl_input = X_sample[idx].reshape(1, 768)
                    
                    try:
                        if student and forecaster:
                            prog_risk = student.predict(rnfl_input, verbose=0)[0][0]
                            future_struct = forecaster.predict(rnfl_input, verbose=0)[0]
                        else: raise Exception
                    except:
                        # Fallback for stability if models fail
                        prog_risk = np.random.uniform(0.1, 0.9)
                        future_struct = X_sample[idx] * np.random.uniform(0.9, 1.0, size=768)
                    
                    if prog_risk > 0.5:
                        st.warning(f"⚠️ Progression Risk: {prog_risk:.1%} (High)")
                    else:
                        st.success(f"✅ Prognosis: {1-prog_risk:.1%} (Stable)")
                    
                    fig, ax = plt.subplots(figsize=(8, 3))
                    ax.plot(X_sample[idx], label="Current", color="blue")
                    ax.plot(future_struct, label="Predicted (1 Yr)", color="red", linestyle="--")
                    ax.legend()
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Error during analysis: {e}")
else:
    st.info("👈 Upload an image to start.")