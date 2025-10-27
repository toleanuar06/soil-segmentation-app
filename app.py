import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import gdown # Google Drive-тан жүктеу үшін

# --- Негізгі параметрлер ---
IMG_WIDTH = 256
IMG_HEIGHT = 256
NUM_CLASSES = 7
MODEL_FILENAME = 'unet_segmentation_model.keras' # Жүктелген файлдың аты

# --- GOOGLE DRIVE FILE ID ---
# Осы жерге Google Drive-тан көшіріп алған FILE_ID-ді қойыңыз!!!
GDRIVE_FILE_ID = '1WtTyXAagScTIYo63_cFUm81_wv6eynIU'
# ----------------------------

# Түстерді анықтау... (қалған код өзгеріссіз)
class_colors = np.array([
    [0, 255, 255], [255, 255, 0], [255, 0, 255], [0, 255, 0],
    [0, 0, 255], [255, 255, 255], [0, 0, 0]
], dtype=np.uint8)

# --- Модельді жүктеу функциясы ---
@st.cache_resource
def load_keras_model(file_id, output_path):
    # Егер модель файлы жергілікті жерде жоқ болса, Google Drive-тан жүктеу
    if not os.path.exists(output_path):
        st.info(f"Модель Google Drive-тан жүктелуде... Бұл біраз уақыт алуы мүмкін.")
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output_path, quiet=False)
            st.success("Модель сәтті жүктелді!")
        except Exception as e:
            st.error(f"Google Drive-тан жүктеу кезінде қате: {e}")
            return None

    # Модельді жүктеу
    try:
        model = tf.keras.models.load_model(output_path)
        st.success("Модель жадыға сәтті жүктелді.")
        return model
    except Exception as e:
        st.error(f"Модель файлын оқу кезінде қате: {e}")
        return None

# --- Болжалды масканы түрлі-түсті суретке айналдыру функциясы ---
# ... (бұл функция өзгеріссіз қалады) ...
def mask_to_rgb(pred_mask):
    rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_index in range(NUM_CLASSES):
        rgb_mask[pred_mask == class_index] = class_colors[class_index]
    return rgb_mask

# --- Негізгі Streamlit қосымшасы ---
st.set_page_config(layout="wide")
st.title("🛰️ Спутниктік суреттерді семантикалық сегменттеу")
st.write("Спутниктік суретті жүктеңіз, ал біз жердің бетін жіктеп береміз (U-Net + MobileNetV2).")

# Модельді Google Drive-тан жүктеу
model = load_keras_model(GDRIVE_FILE_ID, MODEL_FILENAME)

uploaded_file = st.file_uploader("Спутниктік суретті таңдаңыз (.jpg, .png)", type=["jpg", "png"])

if model is None:
    st.warning("Модельді жүктеу мүмкін болмады. Файл ID немесе бөлісу рұқсаттарын тексеріңіз.")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption='Жүктелген сурет', use_column_width=True)

    img_array = img_to_array(image)
    img_resized = tf.image.resize(img_array, [IMG_HEIGHT, IMG_WIDTH])
    img_normalized = img_resized / 255.0
    img_batch = tf.expand_dims(img_normalized, axis=0)

    with st.spinner('Сегменттеу орындалуда...'):
        pred_mask_probs = model.predict(img_batch)
        pred_mask = tf.argmax(pred_mask_probs, axis=-1)
        pred_mask = tf.squeeze(pred_mask).numpy()

    pred_mask_rgb = mask_to_rgb(pred_mask)

    with col2:
        st.image(pred_mask_rgb, caption='Болжалды сегменттеу маскасы', use_column_width=True)

    st.subheader("Түстердің түсіндірмесі (Легенда):")
    # ... (Легенда коды өзгеріссіз) ...
    legend_html = ""
    class_names = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']
    for i, name in enumerate(class_names):
        color = class_colors[i]
        hex_color = '#%02x%02x%02x' % (color[0], color[1], color[2])
        legend_html += f'<span style="display:inline-block; width:20px; height:20px; background-color:{hex_color}; margin-right:5px; vertical-align:middle;"></span> {name} &nbsp;&nbsp;'
    st.markdown(legend_html, unsafe_allow_html=True)
