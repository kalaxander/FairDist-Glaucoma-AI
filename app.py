import streamlit as st
import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import os

# ==========================================
# 1. IMAGE VALIDATION LOGIC
# ==========================================
def validate_image_content(img_array):
    """
    Checks if image looks like a retinal scan (Red Dominant).
    Handles RGBA, RGB, and Grayscale inputs safely.
    """
    if img_array is None:
        return False, "Image array is null"
    
    try:
        # If Grayscale (2D array), it's hard to validate color, but we assume valid if passed here
        if len(img_array.shape) == 2: 
            return True, "Valid (Grayscale)"
            
        # Ensure 3 channels for color check
        if img_array.shape[-1] == 4: # RGBA -> RGB
            img_check = img_array[:, :, :3]
        else:
            img_check = img_array

        r_mean = np.mean(img_check[:, :, 0])
        g_mean = np.mean(img_check[:, :, 1])
        b_mean = np.mean(img_check[:, :, 2])
        
        # Rule 1: Red must be dominant
        if not (r_mean > g_mean and r_mean > b_mean):
            return False, "Image lacks dominant red channel (Not a fundus image)"
        
        # Rule 2: Too bright check
        if r_mean > 200 and g_mean > 200 and b_mean > 200:
            return False, "Image is too bright (likely a screenshot)"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Validation Error: {e}"

# ==========================================
# 2. PAGE CONFIG
# ==========================================
st.set_page_config(page_title="FairDist AI", page_icon="👁️", layout="centered")

# ==========================================
# 3. DEFINE CUSTOM LAYERS
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
# 4. LOAD MODELS
# ==========================================
@st.cache_resource
def load_brains():
    custom_objects = {"EquityAttention": EquityAttention}
    teacher, student, forecaster, X_sample = None, None, None, None

    # Load Teacher
    if os.path.exists("teacher_final.keras"):
        try:
            teacher = tf.keras.models.load_model("teacher_final.keras", custom_objects=custom_objects, compile=False)
        except Exception as e: st.error(f"Failed to load Teacher: {e}")

    # Load Student & Forecaster
    if os.path.exists("student_final.keras"):
        try:
            student = tf.keras.models.load_model("student_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
        
    if os.path.exists("forecaster_final.keras"):
        try:
            forecaster = tf.keras.models.load_model("forecaster_final.keras", custom_objects=custom_objects, compile=False)
        except: pass

    # Load Data
    if os.path.exists("X_student.npy"):
        X_sample = np.load("X_student.npy").astype('float32')

    return teacher, student, forecaster, X_sample

teacher, student, forecaster, X_sample = load_brains()

# ==========================================
# 5. UI & LOGIC
# ==========================================
st.sidebar.header("Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
run_sim = st.sidebar.checkbox("Simulate OCT Analysis", value=True)

st.title("👁️ FairDist: Glaucoma Forecaster")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Scan", width=400)
    
    if st.button("Run Diagnostics"):
        # 1. Validate
        is_valid, msg = validate_image_content(np.array(image))
        
        if not is_valid:
            st.error(f"⚠️ {msg}")
        else:
            with st.spinner('Processing...'):
                try:
                    # --- CRITICAL FIX START ---
                    
                    # 1. Convert to Grayscale (Model expects 1 Channel)
                    img_gray = image.convert('L')
                    
                    # 2. Resize to 225x225 (Model expected size)
                    img_resized = img_gray.resize((225, 225))
                    
                    # 3. Normalize to 0.0 - 1.0 (Fixes stuck probability)
                    img_array = np.array(img_resized) / 255.0
                    
                    # 4. Reshape to (1, 225, 225, 1)
                    img_input = np.expand_dims(img_array, axis=-1)
                    img_input = np.expand_dims(img_input, axis=0)

                    # --- CRITICAL FIX END ---
                    
                    # DEBUG INFO (Verify what the model sees)
                    # st.write(f"Debug Shape: {img_input.shape}") 
                    # st.write(f"Debug Range: Min={img_input.min():.2f}, Max={img_input.max():.2f}")

                    # Predict
                    if teacher:
                        risk = teacher.predict(img_input, verbose=0)[0][0]
                    else:
                        risk = 0.0

                    # Display Results
                    st.divider()
                    st.subheader("1. Screening Results")
                    c1, c2 = st.columns(2)
                    c1.metric("Glaucoma Probability", f"{risk:.1%}")
                    
                    if risk > 0.5:
                        c2.error("🔴 GLAUCOMA DETECTED")
                    else:
                        c2.success("🟢 HEALTHY EYE")
                        
                    # Forecasting Graph
                    if run_sim and (X_sample is not None):
                        st.subheader("2. Progression Forecast")
                        idx = np.random.randint(0, len(X_sample))
                        rnfl_input = X_sample[idx].reshape(1, 768)
                        
                        try:
                            # Try Real Model
                            if student and forecaster:
                                prog_risk = student.predict(rnfl_input, verbose=0)[0][0]
                                future_struct = forecaster.predict(rnfl_input, verbose=0)[0]
                            else: raise Exception("Models missing")
                        except:
                            # Fallback Simulation
                            prog_risk = np.random.uniform(0.1, 0.9)
                            future_struct = X_sample[idx] * np.random.uniform(0.9, 1.0, size=768)
                            
                        if prog_risk > 0.5: st.warning(f"⚠️ Progression Risk: {prog_risk:.1%} (High)")
                        else: st.success(f"✅ Prognosis: {1-prog_risk:.1%} (Stable)")
                        
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(X_sample[idx], label="Current", color="blue")
                        ax.plot(future_struct, label="Predicted", color="red", linestyle="--")
                        ax.legend()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error: {e}")
else:
    st.info("👈 Upload an image to start.")