import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from pathlib import Path

st.set_page_config(page_title='Waste Segregation Demo', page_icon='♻️')
st.title('♻️ Waste Segregation Classifier')
st.write('Upload a picture of waste to classify it as Biodegradable, Nonbiodegradable, or Recyclable.')

MODEL_PATH = Path(__file__).parent / 'best_model_v4_fixed.keras'
CLASS_NAMES = ['Biodegradable', 'Nonbiodegradable', 'Recyclable']

@st.cache_resource
def load_classifier():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)

if not MODEL_PATH.exists():
    st.error(f'Missing {MODEL_PATH.name}. Put the trained V4 checkpoint alongside this script.')
    st.stop()

try:
    model = load_classifier()
except Exception as exc:
    st.error(f'Cannot load model: {exc}')
    st.stop()

uploaded = st.file_uploader('Choose a waste image', type=['jpg', 'jpeg', 'png', 'webp'])
if uploaded is not None:
    img = Image.open(uploaded).convert('RGB')
    st.image(img, caption='Uploaded waste image', width=350)
    # Match V4 training: RGB, bilinear resize, and float32 pixels in 0-255.
    # EfficientNetB0 includes its own rescaling inside the saved model.
    arr = np.asarray(
        img.resize((224, 224), Image.Resampling.BILINEAR), dtype=np.float32
    )
    output = model.predict(np.expand_dims(arr, axis=0), verbose=0)[0]
    if len(output) != len(CLASS_NAMES):
        st.error(f'Expected three outputs, received {len(output)}. Check the checkpoint.')
        st.stop()
    idx = int(np.argmax(output))
    st.success(f'Predicted waste type: {CLASS_NAMES[idx]}')
    st.write('Model scores (not necessarily calibrated probabilities):')
    for name, score in zip(CLASS_NAMES, output):
        st.write(f'{name}: {float(score) * 100:.1f}%')
        st.progress(float(np.clip(score, 0, 1)))
    st.caption('Prediction is experimental. Confirm disposal rules locally before acting on the result.')
