import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# --- Configuración de la página ---
st.set_page_config(page_title="Clasificador de Leucocitos", page_icon="🩸")

# --- Carga del modelo (se hace una sola vez y se guarda en caché) ---
@st.cache_resource
def load_model():
    # El nombre del archivo debe coincidir con el que subas a GitHub
    return tf.keras.models.load_model("best_cnn_leucocitos.keras")

model = load_model()

class_names = ['Basophil', 'Eosinophil', 'Lymphocyte', 'Monocyte', 'Neutrophil']
IMG_SIZE = (96, 96)

# --- Función de predicción ---
def predict(image):
    # Convertir la imagen de PIL a un array de NumPy
    img_array = np.array(image)
    # El modelo espera un batch, así que añadimos una dimensión
    img_array = np.expand_dims(img_array, axis=0)
    # Asegurar el tipo de dato correcto
    img_array = tf.convert_to_tensor(img_array, dtype=tf.float32)
    # Redimensionar si es necesario (aunque el modelo ya lo hace)
    img_array = tf.image.resize(img_array, IMG_SIZE)
    
    probs = model.predict(img_array, verbose=0)[0]
    return {class_names[i]: float(probs[i]) for i in range(len(class_names))}

# --- Interfaz de usuario ---
st.title("Clasificador de Leucocitos 🩸")
st.write("Sube una imagen de un glóbulo blanco para clasificarla.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Imagen subida", use_column_width=True)
    
    if st.button("Clasificar"):
        with st.spinner("Analizando..."):
            try:
                predictions = predict(image)
                st.success("¡Listo!")
                # Mostrar las probabilidades
                for class_name, prob in sorted(predictions.items(), key=lambda x: x[1], reverse=True):
                    st.write(f"**{class_name}:** {prob:.2%}")
            except Exception as e:
                st.error(f"Ocurrió un error al predecir: {e}")