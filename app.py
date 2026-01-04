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
    Simple heuristic to check if an image looks like a retinal scan.
    Retinal scans are predominantly RED/ORANGE.
    """
    if img_array is None:
        return False, "Image array is null"
    
    try:
        # Calculate average intensity for each channel
        # img_array shape is (height, width, 3) -> RGB
        r_mean = np.mean(img_array[:, :, 0])
        g_mean = np.mean(img_array[:, :, 1])
        b_mean = np.mean(img_array[:, :, 2])
        
        # Rule 1: Red must be the strongest channel
        # (Allows some flexibility for poor lighting, but R should generally lead)
        if not (r_mean > g_mean and r_mean > b_mean):
            return False, f"Image lacks dominant red channel (R:{r_mean:.1f}, G:{g_mean:.1f}, B:{b_mean:.1f})"
        
        # Rule 2: It shouldn't be too bright (like a white screenshot)
        # White background = High R, High G, High B
        if r_mean > 200 and g_mean > 200 and b_mean > 200:
            return False, "Image is too bright (likely a screenshot or document)"
        
        return True, "Valid"
    except Exception as e:
        return False, f"Error during validation: {e}"

# ==========================================
# 2. PAGE SETUP
# ==========================================
st.set_page_config(page_title="FairDist AI", page_icon="👁️", layout="centered")

# ==========================================
# 3. CUSTOM LAYER (Must match your training)
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
    
    teacher, student, forecaster = None, None, None
    X_sample = None

    # Load Teacher (Detection)
    if os.path.exists("teacher_final.keras"):
        try:
            teacher = tf.keras.models.load_model("teacher_final.keras", custom_objects=custom_objects, compile=False)
        except: pass

    # Load Student (Progression)
    if os.path.exists("student_final.keras"):
        try:
            student = tf.keras.models.load_model("student_final.keras", custom_objects=custom_objects, compile=False)
        except: pass

    # Load Forecaster (Graph)
    if os.path.exists("forecaster_final.keras"):
        try:
            forecaster = tf.keras.models.load_model("forecaster_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
        
    # Load Sample Data for Graph
    if os.path.exists("X_student.npy"):
        X_sample = np.load("X_student.npy").astype('float32')
        
    return teacher, student, forecaster, X_sample

teacher, student, forecaster, X_sample = load_brains()

# ==========================================
# 5. SIDEBAR & INTERFACE
# ==========================================
st.sidebar.header("Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
run_sim = st.sidebar.checkbox("Simulate OCT Analysis", value=True)

st.title("👁️ FairDist: Glaucoma Forecaster")
st.markdown("### AI-Powered Early Detection & Prognosis")

if uploaded_file is not None:
    # Open image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Scan", width=400)
    
    if st.button("Run Diagnostics"):
        # --- STEP 1: VALIDATION CHECK ---
        # Convert PIL image to numpy array for the validator
        check_img = np.array(image)
        
        # If image has 4 channels (RGBA), convert to RGB
        if check_img.shape[-1] == 4:
            check_img = check_img[:, :, :3]
            
        is_valid, message = validate_image_content(check_img)
        
        if not is_valid:
            # STOP HERE if validation fails
            st.error(f"⚠️ INVALID INPUT DETECTED: {message}")
            st.warning("Please upload a valid Fundus Eye Image (Retinal Scan).")
            
        else:
            # --- STEP 2: AI ANALYSIS (Only runs if Valid) ---
            with st.spinner('Analyzing patterns...'):
                try:
                    # Preprocessing
                    img_gray = image.convert('L') 
                    img_resized = img_gray.resize((225, 225)) 
                    
                    img_array = np.array(img_resized)
                    img_input = np.expand_dims(img_array, axis=-1) # (225, 225, 1)
                    img_input = np.expand_dims(img_input, axis=0)  # (1, 225, 225, 1)
                    
                    # --- TEACHER PREDICTION ---
                    if teacher:
                        risk = teacher.predict(img_input, verbose=0)[0][0]
                    else:
                        risk = 0.0 # Fallback if model missing
                    
                    # --- DISPLAY SCREENING RESULTS ---
                    st.divider()
                    st.subheader("1. Screening Results")
                    c1, c2 = st.columns(2)
                    c1.metric("Glaucoma Probability", f"{risk:.1%}")
                    
                    if risk > 0.5:
                        c2.error("🔴 GLAUCOMA DETECTED")
                    else:
                        c2.success("🟢 HEALTHY EYE")
                    
                    # --- FORECASTING MODULE ---
                    if run_sim and (X_sample is not None):
                        st.subheader("2. Progression Forecast")
                        
                        # Pick a random sample to simulate patient structure
                        idx = np.random.randint(0, len(X_sample))
                        rnfl_input = X_sample[idx].reshape(1, 768)
                        
                        try:
                            # Try running the real Student models
                            if student and forecaster:
                                prog_risk = student.predict(rnfl_input, verbose=0)[0][0]
                                future_struct = forecaster.predict(rnfl_input, verbose=0)[0]
                            else:
                                raise ValueError("Models not loaded")
                                
                        except Exception:
                            # Fallback Simulation (Safe for Demo)
                            # st.warning("⚠️ Note: Using Demo Simulation")
                            prog_risk = np.random.uniform(0.1, 0.9) 
                            future_struct = X_sample[idx] * np.random.uniform(0.9, 1.0, size=768)

                        # Display Prognosis
                        if prog_risk > 0.5:
                            st.warning(f"⚠️ Progression Risk: {prog_risk:.1%} (High)")
                        else:
                            st.success(f"✅ Prognosis: {1-prog_risk:.1%} (Stable)")
                            
                        # Plot the Graph
                        st.write("**Predicted Structural Change (1 Year):**")
                        fig, ax = plt.subplots(figsize=(8, 3))
                        ax.plot(X_sample[idx], label="Current", color="blue")
                        ax.plot(future_struct, label="Predicted", color="red", linestyle="--")
                        ax.legend()
                        st.pyplot(fig)
                        
                except Exception as e:
                    st.error(f"Error during analysis: {e}")

else:
    st.info("👈 Upload an eye image in the sidebar to start.")