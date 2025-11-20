import streamlit as st
import cv2
import numpy as np
import pandas as pd
from inference_sdk import InferenceHTTPClient
from collections import Counter

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(page_title="Contador de Trigo AI", layout="wide")

st.title("🌾 Contador y Clasificador de Semillas de Trigo con AI 🤖")
st.markdown("Sube una imagen para detectar espigas, filtrar por confianza y ver estadísticas.")

# --- CONFIGURACIÓN INTERNA (INVISIBLE PARA EL USUARIO) ---
# Estos datos ahora son fijos. El usuario no puede cambiarlos ni verlos.
WORKSPACE_NAME = "enricthings"
WORKFLOW_ID = "detect-count-and-visualize"

# Recuperamos la API Key de forma silenciosa desde los Secrets
api_key = None
if "ROBOFLOW_KEY" in st.secrets:
    api_key = st.secrets["ROBOFLOW_KEY"]
elif "general" in st.secrets and "ROBOFLOW_KEY" in st.secrets["general"]:
    api_key = st.secrets["general"]["ROBOFLOW_KEY"]

# Si por alguna razón no encuentra la clave, mostramos un error discreto y paramos.
if not api_key:
    st.error("🚨 Error de configuración: No se encontró la API Key en los secretos del servidor.")
    st.stop()

# --- BARRA LATERAL (SOLO FILTROS) ---
st.sidebar.header("Ajustes de Detección")
# Solo dejamos los controles que le interesan al usuario final
conf_threshold = st.sidebar.slider("Nivel de Confianza", 0.0, 1.0, 0.40, 0.05)
iou_threshold = st.sidebar.slider("Superposición (Overlap)", 0.0, 1.0, 0.50, 0.05)

st.sidebar.info("💡 Ajusta la confianza si ves demasiados o muy pocos objetos.")

# --- FUNCIONES ---

@st.cache_data
def get_roboflow_predictions(image_bytes, _api_key, _workspace, _workflow):
    # Conectamos con Roboflow
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
    
    # Extracción de datos
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
    
    # 1. Filtrar
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
            
    # 2. NMS
    indices = []
    if boxes:
        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.0, iou_th)
        if hasattr(indices, 'flatten'):
            indices = indices.flatten()
            
    # 3. Dibujar
    final_labels = []
    for i in indices:
        box = boxes[i]
        lbl = labels[i]
        final_labels.append(lbl)
        
        # Color Verde Roboflow
        color = (0, 255, 0)
        cv2.rectangle(img_copy, (box[0], box[1]), (box[0]+box[2], box[1]+box[3]), color, 4)
        
        # Texto con fondo para que se lea mejor
        label_text = f"{lbl}"
        (w_text, h_text), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.rectangle(img_copy, (box[0], box[1]-h_text-10), (box[0]+w_text, box[1]), color, -1)
        cv2.putText(img_copy, label_text, (box[0], box[1]-5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,0), 2)
        
    return img_copy, final_labels

# --- INTERFAZ PRINCIPAL ---

uploaded_file = st.file_uploader("Arrastra tu imagen aquí...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Previsualizar imagen pequeña antes de analizar (opcional, da feedback rápido)
    # st.image(uploaded_file, caption="Imagen original", width=200)

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    opencv_image = cv2.imdecode(file_bytes, 1)
    opencv_image = cv2.cvtColor(opencv_image, cv2.COLOR_BGR2RGB)
    
    uploaded_file.seek(0)
    
    # Botón de acción
    if st.button("🔍 Analizar Imagen", type="primary"):
        with st.spinner('Procesando con IA...'):
            try:
                # Usamos las variables FIJAS (api_key, WORKSPACE_NAME...)
                preds = get_roboflow_predictions(uploaded_file.getvalue(), api_key, WORKSPACE_NAME, WORKFLOW_ID)
                result_img, final_labels = draw_and_count(opencv_image, preds, conf_threshold, iou_threshold)
                
                st.markdown("---")
                
                # Layout de resultados
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.image(result_img, caption="Resultado del análisis", use_container_width=True)
                    
                with col2:
                    st.subheader("📊 Resultados")
                    st.metric(label="Total Objetos", value=len(final_labels))
                    
                    if final_labels:
                        counts = Counter(final_labels)
                        df = pd.DataFrame(counts.items(), columns=['Clase', 'Cantidad']).set_index('Clase')
                        st.dataframe(df, use_container_width=True)
                    else:
                        st.warning("Intenta bajar la confianza en el menú lateral.")
                        
            except Exception as e:
                st.error(f"Error: {e}")
