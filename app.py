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

# --- GESTIÓN DE SECRETOS (LOGICA MEJORADA) ---
# Esta función busca la clave en la raíz O dentro de la sección [general]
def get_secret_key():
    # 1. Buscar directamente (Formato: CLAVE = "valor")
    if "ROBOFLOW_KEY" in st.secrets:
        return st.secrets["ROBOFLOW_KEY"]
    
    # 2. Buscar dentro de 'general' (Formato: [general] \n CLAVE = "valor")
    if "general" in st.secrets and "ROBOFLOW_KEY" in st.secrets["general"]:
        return st.secrets["general"]["ROBOFLOW_KEY"]
        
    return ""

default_key = get_secret_key()

# --- BARRA LATERAL (SIDEBAR) ---
st.sidebar.header("Configuración")

# DEBUG: ESTO TE DIRÁ SI STREAMLIT ESTÁ LEYENDO EL SECRETO O NO
# (Borra estas 3 líneas cuando ya funcione)
if default_key:
    st.sidebar.success(f"✅ Secreto detectado (Longitud: {len(default_key)})")
else:
    st.sidebar.error("❌ No se detecta el secreto en st.secrets")
# ---------------------------------------------------------

api_key = st.sidebar.text_input(
    "Roboflow API Key", 
    value=default_key, 
    type="password",
    help="Si no tienes una, deja la que viene por defecto (usa la cuenta del creador)."
)

# Si el usuario borra el campo manualmente, usamos la default
if not api_key:
    api_key = default_key

workspace_name = st.sidebar.text_input("Workspace", value="enricthings")
workflow_id = st.sidebar.text_input("Workflow ID", value="detect-count-and-visualize")

st.sidebar.markdown("---")
st.sidebar.header("Filtros")
conf_threshold = st.sidebar.slider("Nivel de Confianza (Confidence)", 0.0, 1.0, 0.40, 0.05)
iou_threshold = st.sidebar.slider("Superposición (Overlap / IoU)", 0.0, 1.0, 0.50, 0.05)

# --- FUNCIONES ---

@st.cache_data
def get_roboflow_predictions(image_bytes, _api_key, _workspace, _workflow):
    # Protección extra: Si la clave llega vacía, lanzamos error antes de llamar a la API
    if not _api_key:
        raise ValueError("La API Key está vacía. Revisa los 'Secrets' o escribe una manualmente.")

    client = InferenceHTTPClient(api_url="https://serverless.roboflow.com", api_key=_api_key)
    
    with open("temp_image.jpg", "wb") as f:
        f.write(image_bytes)

    result = client.run_workflow(
        workspace_name=_workspace,
        workflow_id=_workflow,
        images={"image": "temp_image.jpg"},
        use_cache=True
    )
    
    data = result[0]
    predictions = []
    
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
            
    indices = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, iou_th)
        if hasattr(indices, 'flatten'):
            indices = indices.flatten()
            
    final_labels = []
    for i in indices:
        box = boxes[i]
        lbl = labels[i]
        final_labels.append(lbl)
        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), (0, 255, 0), 4)
        cv2.putText(img_copy, lbl, (box[0], box[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
        
    return img_copy, final_labels

# --- INTERFAZ PRINCIPAL ---

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    uploaded_file.seek(0)
    
    if st.button("🔍 Analizar Imagen"):
        with st.spinner('Consultando a Roboflow...'):
            try:
                preds = get_roboflow_predictions(uploaded_file.getvalue(), api_key, workspace_name, workflow_id)
                result_img, final_labels = draw_and_count(opencv_image, preds, conf_threshold, iou_threshold)
                
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
