import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from PIL import Image
import os
import gdown

# --- Негізгі параметрлер ---
# Модельді осы өлшеммен оқыттық
MODEL_IMG_SIZE = 256
NUM_CLASSES = 7
PROJECT_PATH = '/content/drive/MyDrive/DeepGlobe_Segmentation'
MODEL_FILENAME = 'unet_segmentation_model.keras'
MODEL_SAVE_PATH = os.path.join(PROJECT_PATH, MODEL_FILENAME)

# --- GOOGLE DRIVE FILE ID ---
# Google Drive-тағы модель файлының ID-і.
# Бұл модельді қайта оқытпас үшін қажет.
GDRIVE_FILE_ID = '1WtTyXAagScTIYo63_cFUm81_wv6eynIU' # <<<--- ОСЫ ЖЕРГЕ ФАЙЛ ID-ІҢІЗДІ ҚОЙЫҢЫЗ!

# Түстерді анықтау (Легенда үшін)
class_colors = np.array([
    [0, 255, 255],  # Urban
    [255, 255, 0],  # Agriculture
    [255, 0, 255],  # Rangeland
    [0, 255, 0],    # Forest
    [0, 0, 255],    # Water
    [255, 255, 255],# Barren
    [0, 0, 0]       # Unknown
], dtype=np.uint8)
class_names = ['Urban', 'Agriculture', 'Rangeland', 'Forest', 'Water', 'Barren', 'Unknown']

# --- Модельді жүктеу функциясы (өзгеріссіз) ---
@st.cache_resource
def load_keras_model(file_id, output_path):
    if not os.path.exists(output_path):
        st.info(f"Модель Google Drive-тан жүктелуде...")
        url = f'https://drive.google.com/uc?id={file_id}'
        try:
            gdown.download(url, output_path, quiet=False)
            st.success("Модель сәтті жүктелді!")
        except Exception as e:
            st.error(f"Google Drive-тан жүктеу қатесі: {e}")
            return None
    try:
        model = tf.keras.models.load_model(output_path)
        st.success("Модель жадыға сәтті жүктелді.")
        return model
    except Exception as e:
        st.error(f"Модель файлын оқу қатесі: {e}")
        return None

# --- Болжалды масканы түрлі-түсті суретке айналдыру (өзгеріссіз) ---
def mask_to_rgb(pred_mask):
    rgb_mask = np.zeros((pred_mask.shape[0], pred_mask.shape[1], 3), dtype=np.uint8)
    for class_index in range(NUM_CLASSES):
        rgb_mask[pred_mask == class_index] = class_colors[class_index]
    return rgb_mask

# --- ЖАҢА ФУНКЦИЯ: Бөліктерге бөліп болжау ---
def predict_tiled(image, model):
    """
    Үлкен суретті MODEL_IMG_SIZE өлшемді плиткаларға бөліп, болжам жасап,
    нәтижелерді қайта біріктіреді.
    """
    img_height, img_width, _ = image.shape
    tile_size = MODEL_IMG_SIZE
    
    # Нәтижені сақтайтын бос массив (маска)
    full_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    
    # Суретті плиткаларға бөліп өңдеу
    for y in range(0, img_height, tile_size):
        for x in range(0, img_width, tile_size):
            # Плитканың шекараларын анықтау
            y1, y2 = y, min(y + tile_size, img_height)
            x1, x2 = x, min(x + tile_size, img_width)
            
            # Плитканы кесіп алу
            tile = image[y1:y2, x1:x2]
            
            # Егер плитканың өлшемі модельге сәйкес келмесе (шеткі плиткалар)
            # Оны 0-мен толықтырып, 256x256-ға жеткізу
            tile_padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            tile_padded[0:(y2-y1), 0:(x2-x1)] = tile
            
            # Модельге дайындау
            tile_preprocessed = tf.keras.applications.mobilenet_v2.preprocess_input(tile_padded)
            tile_batch = tf.expand_dims(tile_preprocessed, axis=0)
            
            # Болжам жасау
            pred_mask_probs = model.predict(tile_batch, verbose=0)
            pred_mask = tf.argmax(pred_mask_probs, axis=-1)
            pred_mask = tf.squeeze(pred_mask).numpy() # (256, 256)
            
            # Нәтиженің қажетті бөлігін толық маскаға "желімдеу"
            full_mask[y1:y2, x1:x2] = pred_mask[0:(y2-y1), 0:(x2-x1)]
            
    return full_mask

# --- Негізгі Streamlit қосымшасы (ЖАҢАРТЫЛДЫ) ---
st.set_page_config(layout="wide")
st.title("🛰️ Спутниктік суреттерді семантикалық сегменттеу (Жетілдірілген)")
st.write("Кез келген өлшемді суретті жүктеңіз. Модель оны бөліктерге бөліп, жоғары сапада талдайды.")

model = load_keras_model(GDRIVE_FILE_ID, MODEL_FILENAME)
uploaded_file = st.file_uploader("Спутниктік суретті таңдаңыз (.jpg, .png)", type=["jpg", "png"])

if model is None:
    st.warning("Модельді жүктеу мүмкін болмады. Файл ID-ін тексеріңіз.")
elif uploaded_file is not None:
    image = Image.open(uploaded_file).convert('RGB')
    image_np = np.array(image) # Суретті numpy массивіне айналдыру
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(image, caption=f'Жүктелген сурет ({image_np.shape[0]}x{image_np.shape[1]})', use_column_width=True)

    with st.spinner('Сурет бөліктерге бөлініп, талдануда... Бұл үлкен суреттер үшін уақыт алады...'):
        # Кішірейтудің орнына, жаңа 'predict_tiled' функциясын шақыру
        pred_mask_final = predict_tiled(image_np, model)
        
    # Болжалды масканы түрлі-түсті суретке айналдыру
    pred_mask_rgb = mask_to_rgb(pred_mask_final)

    with col2:
        st.image(pred_mask_rgb, caption='Болжалды сегменттеу (Жоғары сапа)', use_column_width=True)

    # Легенда (өзгеріссіз)
    st.subheader("Түстердің түсіндірмесі (Легенда):")
    legend_html = ""
    for i, name in enumerate(class_names):
        color = class_colors[i]
        hex_color = '#%02x%02x%02x' % (color[0], color[1], color[2])
        legend_html += f'<span style="display:inline-block; width:20px; height:20px; background-color:{hex_color}; margin-right:5px; vertical-align:middle;"></span> {name} &nbsp;&nbsp;'
    st.markdown(legend_html, unsafe_allow_html=True)
