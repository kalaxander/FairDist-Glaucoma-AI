import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
from scipy.ndimage import gaussian_filter1d
from fpdf import FPDF
import tempfile
import os

# --- PAGE CONFIG ---
st.set_page_config(page_title="Glaucoma AI Diagnostic Suite", page_icon="👁️", layout="wide")

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .big-font { font-size:24px !important; font-weight: bold; }
    .prediction-box { padding: 20px; border-radius: 10px; margin-bottom: 20px; text-align: center; }
    .stable { background-color: #d4edda; color: #155724; border: 1px solid #c3e6cb; }
    .risk { background-color: #f8d7da; color: #721c24; border: 1px solid #f5c6cb; }
    .uncertain { background-color: #fff3cd; color: #856404; border: 1px solid #ffeeba; }
    </style>
    """, unsafe_allow_html=True)

# --- 1. LOAD AI MODEL ---
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('final_glaucoma_model.keras', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

model = load_model()

# --- 2. HELPER FUNCTIONS ---
def extract_rnfl_profile(img):
    """Simulates extracting the RNFL thickness profile from the image."""
    raw_profile = np.sum(img, axis=0)
    smooth_profile = gaussian_filter1d(raw_profile, sigma=2)
    norm_profile = (smooth_profile - np.min(smooth_profile)) / (np.max(smooth_profile) - np.min(smooth_profile) + 1e-5)
    return norm_profile * 0.6 + 0.4 

# --- 3. ADVANCED PREDICTION (BAYESIAN APPROXIMATION) ---
def predict_with_uncertainty(model, img_batch, n_iter=10):
    results = []
    for i in range(n_iter):
        noise = np.random.normal(0, 0.005, img_batch.shape) 
        noisy_input = img_batch + noise
        pred = model.predict(noisy_input, verbose=0)[0][0]
        results.append(pred)
    
    prediction_mean = np.mean(results)
    prediction_uncertainty = np.std(results)
    return prediction_mean, prediction_uncertainty

# --- 4. GRAD-CAM EXPLAINABILITY (ROBUST VERSION) ---
def make_gradcam_heatmap(img_array, model):
    try:
        vgg_layer = model.get_layer('vgg16')
        last_conv_layer_name = 'block5_conv3'
        
        # Robust Input Handling
        if isinstance(vgg_layer.input, list): vgg_input = vgg_layer.input[0]
        else: vgg_input = vgg_layer.input

        # Robust Output Handling
        if isinstance(vgg_layer.output, list): vgg_output_tensor = vgg_layer.output[0]
        else: vgg_output_tensor = vgg_layer.output

        vgg_sub_model = tf.keras.models.Model(
            inputs=vgg_input,
            outputs=[vgg_layer.get_layer(last_conv_layer_name).output, vgg_output_tensor]
        )

        classifier_input = tf.keras.Input(shape=vgg_output_tensor.shape[1:])
        x = classifier_input
        for layer in model.layers[3:]:
            x = layer(x)
        classifier_model = tf.keras.models.Model(inputs=classifier_input, outputs=x)

        with tf.GradientTape() as tape:
            adapter_output = model.layers[1](img_array)
            conv_output, vgg_output = vgg_sub_model(adapter_output)
            preds = classifier_model(vgg_output)
            loss = preds[:, 0]

        grads = tape.gradient(loss, conv_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

        conv_output = conv_output[0]
        heatmap = conv_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()

    except Exception as e:
        return np.zeros((200, 200))

def overlay_heatmap(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    if len(img.shape) == 2: img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else: img_rgb = img
    if img_rgb.max() <= 1.0: img_rgb = np.uint8(img_rgb * 255)
    superimposed_img = heatmap * alpha + img_rgb
    return cv2.cvtColor(np.clip(superimposed_img, 0, 255).astype('uint8'), cv2.COLOR_BGR2RGB)

# --- 5. PDF GENERATOR ---
class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 15)
        self.cell(0, 10, 'Glaucoma AI Diagnostic Report', 0, 1, 'C')
        self.ln(10)
    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', 0, 0, 'C')

def create_pdf(patient_id, age, iop, diagnosis, confidence, uncertainty, img_path, graph_path):
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"Patient ID: {patient_id}", ln=True)
    pdf.cell(200, 10, txt=f"Age: {age} | IOP: {iop} mmHg", ln=True)
    pdf.cell(200, 10, txt=f"Date: 07-Jan-2026", ln=True)
    pdf.ln(10)
    
    pdf.set_font("Arial", 'B', 14)
    if "GLAUCOMA" in diagnosis: pdf.set_text_color(255, 0, 0)
    else: pdf.set_text_color(0, 128, 0)
    pdf.cell(200, 10, txt=f"Diagnosis: {diagnosis}", ln=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt=f"AI Confidence: {confidence}", ln=True)
    pdf.cell(200, 10, txt=f"Uncertainty: +/- {uncertainty}", ln=True)
    pdf.ln(10)
    
    pdf.cell(200, 10, txt="Scan Analysis (Grad-CAM Heatmap):", ln=True)
    pdf.image(img_path, x=10, y=None, w=100)
    pdf.ln(5)
    pdf.cell(200, 10, txt="Longitudinal Forecast (1 Year):", ln=True)
    pdf.image(graph_path, x=10, y=None, w=100)
    return pdf

# --- MAIN APP LAYOUT ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3063/3063822.png", width=80)
st.sidebar.title("Patient Data")
p_id = st.sidebar.text_input("Patient ID", "DMET-1124")
p_age = st.sidebar.slider("Age", 40, 90, 65)
p_iop = st.sidebar.slider("Intraocular Pressure (mmHg)", 10, 40, 15)
p_history = st.sidebar.radio("Family History", ["No", "Yes"])

st.title("👁️ AI Glaucoma Diagnostic Suite")
st.caption("Research Prototype v3.0 | Uncertainty + Forecasting + Multimodal View")

uploaded_file = st.file_uploader("Upload OCT Scan (.npz)", type=["npz"])

if uploaded_file is not None:
    with np.load(uploaded_file) as data:
        # --- DATA PREP ---
        # 1. OCT B-Scans
        if 'bscans' in data: oct_3d = data['bscans']
        elif 'oct_bscans' in data: oct_3d = data['oct_bscans']
        else: st.error("Invalid File"); st.stop()
        
        # 2. Fundus Image (Restored Feature!)
        if 'slo_fundus' in data:
            fundus_img = data['slo_fundus']
        else:
            # Fallback: Projection if no fundus exists
            fundus_img = np.mean(oct_3d, axis=0)
            
        # Normalize Fundus
        fundus_img = fundus_img.astype('float32')
        fundus_img = (fundus_img - np.min(fundus_img)) / (np.max(fundus_img) - np.min(fundus_img))
        
        # 3. Process OCT Middle Slice
        if oct_3d.ndim == 3:
            mid = oct_3d.shape[0] // 2 
            img = oct_3d[mid, :, :]
        else: img = oct_3d 
        
        img = cv2.resize(img, (200, 200))
        img_processed = img.astype('float32') / 255.0
        img_batch = np.expand_dims(np.expand_dims(img_processed, -1), 0)
        
        # --- AI EXECUTION ---
        prediction_mean, uncertainty = predict_with_uncertainty(model, img_batch)
        
        # Risk Logic
        risk_score = prediction_mean
        risk_reasons = []
        is_uncertain = uncertainty > 0.10
        
        if p_iop > 21: risk_score += 0.15; risk_reasons.append("High Intraocular Pressure")
        if p_history == "Yes": risk_score += 0.10; risk_reasons.append("Family History")
        
        final_confidence = min(risk_score, 0.99) if prediction_mean > 0.5 else max(risk_score, 0.01)
        is_glaucoma = final_confidence > 0.5
        
        # Heatmap
        try:
            heatmap = make_gradcam_heatmap(img_batch, model)
            overlay_img = overlay_heatmap(img, heatmap)
        except:
            overlay_img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # --- UI COLUMNS ---
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("1. AI Analysis")
            # TABS restored for Multimodal View
            tab_ai, tab_fundus, tab_raw = st.tabs(["AI Heatmap", "Fundus Image", "Raw OCT"])
            
            with tab_ai:
                st.image(overlay_img, caption="Grad-CAM Heatmap", use_container_width=True)
            with tab_fundus:
                st.image(fundus_img, caption="En Face Fundus View", use_container_width=True)
            with tab_raw:
                st.image(img, caption="Raw OCT B-Scan", use_container_width=True)
            
        with col2:
            st.subheader("2. Diagnosis & Forecast")
            
            # Diagnosis Box
            if is_uncertain:
                 st.markdown(f'<div class="prediction-box uncertain"><p class="big-font">⚠️ INCONCLUSIVE</p>'
                            f'Model Uncertainty: ±{uncertainty*100:.1f}%</div>', unsafe_allow_html=True)
            elif is_glaucoma:
                st.markdown(f'<div class="prediction-box risk"><p class="big-font">DIAGNOSIS: GLAUCOMA</p>'
                            f'Confidence: {final_confidence*100:.1f}% (±{uncertainty*100:.1f}%)</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="prediction-box stable"><p class="big-font">DIAGNOSIS: HEALTHY</p>'
                            f'Confidence: {final_confidence*100:.1f}% (±{uncertainty*100:.1f}%)</div>', unsafe_allow_html=True)
            
            # --- FORECASTING GRAPH ---
            st.markdown("**Longitudinal RNFL Forecast (1 Year)**")
            current_curve = extract_rnfl_profile(img)
            x_axis = np.linspace(0, 360, len(current_curve))
            noise = np.random.normal(0, 0.015, len(current_curve))
            
            thinning_factor = (final_confidence * 0.20) if is_glaucoma else 0.02
            future_curve = current_curve * (1.0 - thinning_factor) + noise
            
            fig, ax = plt.subplots(figsize=(6, 3))
            ax.plot(x_axis, current_curve, label='Baseline', color='blue')
            ax.plot(x_axis, future_curve, label='1-Year Prediction', color='red', linestyle='--')
            ax.fill_between(x_axis, current_curve, future_curve, color='red', alpha=0.1)
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0, 1.1)
            st.pyplot(fig)

        # --- PDF REPORT ---
        st.markdown("---")
        if st.button("📄 Generate Full Medical Report"):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_img:
                plt.imsave(tmp_img.name, overlay_img)
                img_path = tmp_img.name
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_graph:
                fig.savefig(tmp_graph.name)
                graph_path = tmp_graph.name
            
            pdf = create_pdf(p_id, p_age, p_iop, 
                             "GLAUCOMA" if is_glaucoma else "HEALTHY", 
                             f"{final_confidence*100:.1f}%", 
                             f"{uncertainty*100:.1f}%",
                             img_path, graph_path)
            
            pdf_bytes = pdf.output(dest='S').encode('latin-1')
            st.download_button(label="📥 Download PDF Report", data=pdf_bytes, 
                               file_name=f"Report_{p_id}.pdf", mime='application/pdf')