import streamlit as st
import cv2
import numpy as np
import pandas as pd
from inference_sdk import InferenceHTTPClient
from collections import Counter

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Contador de Trigo AI", layout="wide")

st.title("🌾 Detector y Contador de Trigo con AI")
st.markdown("Sube una imagen para detectar espigas, filtrar por confianza y ver estadísticas.")

default_key = ""
if "ROBOFLOW_KEY" in st.secrets:
    default_key = st.secrets["ROBOFLOW_KEY"]

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("Configuración")
api_key = st.sidebar.text_input(
    "Roboflow API Key", 
    value=default_key, 
    type="password",
    help="Si no tienes una, deja la que viene por defecto (usa la cuenta del creador)."
)
workspace_name = st.sidebar.text_input("Workspace", value="enricthings")
workflow_id = st.sidebar.text_input("Workflow ID", value="detect-count-and-visualize")

st.sidebar.markdown("---")
st.sidebar.header("Filtros")
conf_threshold = st.sidebar.slider("Nivel de Confianza (Confidence)", 0.0, 1.0, 0.40, 0.05)
iou_threshold = st.sidebar.slider("Superposición (Overlap / IoU)", 0.0, 1.0, 0.50, 0.05)

# --- FUNCIONES ---

# Usamos @st.cache_data para NO llamar a la API cada vez que mueves un slider
@st.cache_data
def get_roboflow_predictions(image_bytes, _api_key, _workspace, _workflow):
    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=_api_key)
    
    # Guardamos temporalmente para enviar (la librería inference pide archivo o ruta)
    # Un truco para enviar bytes directamente sería usar base64, pero guardarlo es seguro en local
    with open("temp_image.jpg", "wb") as f:
        f.write(image_bytes)

    result = client.run_workflow(
        workspace_name=_workspace,
        workflow_id=_workflow,
        images={"image": "temp_image.jpg"},
        use_cache=True
    )
    
    # Limpieza de datos (tu lógica probada)
    data = result[0]
    predictions = []
    
    # Extracción robusta
    raw_preds = data.get('predictions', [])
    if isinstance(raw_preds, dict) and 'predictions' in raw_preds:
        predictions = raw_preds['predictions']
    elif isinstance(raw_preds, list):
        predictions = raw_preds
        
    return predictions

def draw_and_count(image, predictions, conf_th, iou_th):
    img_copy = image.copy()
    
    boxes = []
    confidences = []
    labels = []
    
    # 1. Filtrar por confianza
    for p in predictions:
        conf = p.get('confidence', 0)
        if conf >= conf_th:
            w = p.get('width', 0)
            h = p.get('height', 0)
            x = p.get('x', 0)
            y = p.get('y', 0)
            
            x_min = int(x - w/2)
            y_min = int(y - h/2)
            
            boxes.append([x_min, y_min, int(w), int(h)])
            confidences.append(float(conf))
            labels.append(p.get('class', 'Obj'))
            
    # 2. NMS (Overlap)
    indices = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, iou_th)
        if hasattr(indices, 'flatten'):
            indices = indices.flatten()
            
    # 3. Dibujar y contar finales
    final_labels = []
    for i in indices:
        box = boxes[i]
        lbl = labels[i]
        
        final_labels.append(lbl)
        
        # Dibujar rectángulo verde
        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 4)
        # Texto
        cv2.putText(img_copy, lbl, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    return img_copy, final_labels

# --- INTERFAZ PRINCIPAL ---

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Leer imagen para OpenCV
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB) # Corregir color
    
    # Resetear el puntero del archivo para leer bytes de nuevo si es necesario
    uploaded_file.seek(0)
    
    with st.spinner('Consultando a Roboflow...'):
        try:
            # Llamada a la API (Solo ocurre una vez por imagen gracias al caché)
            preds = get_roboflow_predictions(uploaded_file.getvalue(), api_key, workspace_name, workflow_id)
            
            # Procesado local (Ocurre cada vez que mueves el slider)
            result_img, final_labels = draw_and_count(opencv_image, preds, conf_threshold, iou_threshold)
            
            # COLUMNAS: Izquierda (Imagen), Derecha (Datos)
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(result_img, caption=f"Detección Visual ({len(final_labels)} objetos)", use_container_width=True)
                
            with col2:
                st.subheader("📊 Estadísticas")
                st.metric(label="Total Detectado", value=len(final_labels))
                
                if final_labels:
                    counts = Counter(final_labels)
                    df = pd.DataFrame(counts.items(), columns=['Clase', 'Cantidad']).set_index('Clase')
                    st.dataframe(df, use_container_width=True)
                else:
                    st.warning("No hay objetos con estos filtros.")
                    
        except Exception as e:
            st.error(f"Ocurrió un error: {e}")
