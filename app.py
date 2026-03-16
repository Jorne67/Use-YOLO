import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Konfiguration der Seite
st.set_page_config(page_title="KI-Kleidungs-Detektor", layout="centered")

st.title("👕 YOLO Kleidungserkennung")
st.write("Lade ein Bild hoch und die KI sagt dir, welche Kleidungsstücke sie sieht.")

# Modell laden 
# Du kannst 'yolov8n.pt' (Standard) oder dein eigenes trainiertes Modell 'best.pt' nutzen
@st.cache_resource
def load_model():
    # Hinweis: Das Standard-YOLO-Modell erkennt bereits "tie", "handbag" etc. 
    # Für spezifische Kleidung brauchst du meist ein speziell trainiertes Modell.
    model = YOLO('yolov8n.pt') 
    return model

model = load_model()

# Dateiupload
uploaded_file = st.file_uploader("Wähle ein Bild aus...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Bild öffnen
    image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)

    with col1:
        st.image(image, caption="Hochgeladenes Bild", use_container_width=True)

    with col2:
        with st.spinner('Analysiere...'):
            # Vorhersage treffen
            results = model(image)
            
            # Ergebnis-Bild rendern (mit Boxen)
            res_plotted = results[0].plot()
            st.image(res_plotted, caption="Erkennung", use_container_width=True)

    # Gefundene Objekte auflisten
    st.subheader("Gefundene Objekte:")
    detections = results[0].boxes.cls.tolist()
    names = model.names

    if len(detections) > 0:
        for class_id in detections:
            st.write(f"- ✅ **{names[int(class_id)]}**")
    else:
        st.write("Keine Objekte erkannt.")
