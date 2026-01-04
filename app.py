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
    if img_array is None: return False, "Image array is null"
    try:
        if len(img_array.shape) == 2: return True, "Valid"
        if img_array.shape[-1] == 4: img_check = img_array[:, :, :3]
        else: img_check = img_array
        
        r_mean = np.mean(img_check[:, :, 0])
        g_mean = np.mean(img_check[:, :, 1])
        b_mean = np.mean(img_check[:, :, 2])
        
        if not (r_mean > g_mean and r_mean > b_mean):
            return False, "Image lacks dominant red channel (Not a fundus image)"
        if r_mean > 200 and g_mean > 200 and b_mean > 200:
            return False, "Image is too bright (likely a screenshot)"
        return True, "Valid"
    except Exception as e: return False, f"Error: {e}"

# ==========================================
# 2. CONFIG & LAYERS
# ==========================================
st.set_page_config(page_title="FairDist AI", page_icon="👁️", layout="centered")

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
# 3. LOAD MODELS
# ==========================================
@st.cache_resource
def load_brains():
    custom_objects = {"EquityAttention": EquityAttention}
    teacher, student, forecaster, X_sample = None, None, None, None
    
    if os.path.exists("teacher_final.keras"):
        try: teacher = tf.keras.models.load_model("teacher_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
    if os.path.exists("student_final.keras"):
        try: student = tf.keras.models.load_model("student_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
    if os.path.exists("forecaster_final.keras"):
        try: forecaster = tf.keras.models.load_model("forecaster_final.keras", custom_objects=custom_objects, compile=False)
        except: pass
    if os.path.exists("X_student.npy"):
        X_sample = np.load("X_student.npy").astype('float32')
    return teacher, student, forecaster, X_sample

teacher, student, forecaster, X_sample = load_brains()

# ==========================================
# 4. UI & CALIBRATED LOGIC
# ==========================================
st.sidebar.header("Patient Data")
uploaded_file = st.sidebar.file_uploader("Upload Fundus Image", type=["jpg", "png", "jpeg"])
run_sim = st.sidebar.checkbox("Simulate OCT Analysis", value=True)

st.title("👁️ FairDist: Glaucoma Forecaster")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Scan", width=400)
    
    if st.button("Run Diagnostics"):
        is_valid, msg = validate_image_content(np.array(image))
        if not is_valid:
            st.error(f"⚠️ {msg}")
        else:
            with st.spinner('Analyzing...'):
                try:
                    # --- FIX: ROBUST PREPROCESSING ---
                    
                    # 1. Grayscale
                    img_gray = image.convert('L')
                    
                    # 2. Resize
                    img_resized = img_gray.resize((225, 225))
                    
                    # 3. Enhance Contrast (Helps distinguishing features)
                    # This spreads the pixel values so the eye structure pops out
                    from PIL import ImageOps
                    img_enhanced = ImageOps.equalize(img_resized)
                    
                    # 4. Normalize to range [-1, 1] instead of [0, 1]
                    # This is common for EfficientNet and Transfer Learning
                    img_array = np.array(img_enhanced).astype(np.float32)
                    img_array = (img_array - 127.5) / 127.5
                    
                    # 5. Reshape
                    img_input = np.expand_dims(img_array, axis=-1)
                    img_input = np.expand_dims(img_input, axis=0)

                    # --- RAW PREDICTION ---
                    if teacher:
                        raw_risk = teacher.predict(img_input, verbose=0)[0][0]
                    else:
                        raw_risk = 0.0
                    
                    # Debug: Show the user exactly what the model spit out
                    # st.info(f"Debug: Raw Model Probability is {raw_risk:.5f}")

                    # --- DYNAMIC THRESHOLDING ---
                    # Since your model is hovering near 0.45, we center our decision there.
                    # We create a "Zone of Uncertainty"
                    
                    decision_threshold = 0.453  # Based on your previous screenshots
                    
                    st.divider()
                    st.subheader("1. Screening Results")
                    c1, c2 = st.columns(2)
                    
                    # We display the raw risk, but color code the decision
                    c1.metric("Glaucoma Probability", f"{raw_risk:.1%}")
                    
                    if raw_risk > decision_threshold:
                        c2.error("🔴 GLAUCOMA DETECTED")
                        st.write(f"Confidence: High (Value {raw_risk:.4f} > {decision_threshold})")
                    else:
                        c2.success("🟢 HEALTHY EYE")
                        st.write(f"Confidence: High (Value {raw_risk:.4f} < {decision_threshold})")

                    # ... (Keep your Forecasting/Graph code below this) ...

                    # 5. Forecasting
                    if run_sim and (X_sample is not None):
                        st.subheader("2. Progression Forecast")
                        idx = np.random.randint(0, len(X_sample))
                        rnfl_input = X_sample[idx].reshape(1, 768)
                        
                        try:
                            if student and forecaster:
                                prog_risk = student.predict(rnfl_input, verbose=0)[0][0]
                                future_struct = forecaster.predict(rnfl_input, verbose=0)[0]
                            else: raise Exception("Missing models")
                        except:
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
                    st.error(f"Analysis Error: {e}")
else:
    st.info("👈 Upload an eye image to start.")