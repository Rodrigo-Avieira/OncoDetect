import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# --- CONFIGURAÇÃO DA IA ---
CONF_THRESHOLD = 0.20
NMS_THRESHOLD = 0.40

# --- CONFIGURAÇÃO DE PÁGINA ---
st.set_page_config(
    page_title="OncoDetect - AI",
    layout="wide",
    initial_sidebar_state="collapsed"
)

#estilização
st.markdown("""
<style>
    /* Fundo da tela inteira com gradiente sutil */
    .stApp {
        background-color: #0e0e11;
        background-image: radial-gradient(circle at 50% 0%, #2a0a18 0%, #0e0e11 60%);
    }
    
    /* Esconde o menu padrão do Streamlit para ficar mais limpo */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Top Bar falsa igual a da foto */
    .top-bar {
        text-align: center;
        color: #ffffff;
        font-size: 1.2rem;
        font-weight: 500;
        margin-top: -40px;
        margin-bottom: 40px;
        letter-spacing: 1px;
    }
    
    /* Título com Gradiente Rosa/Neon */
    .pink-title {
        text-align: center;
        background: linear-gradient(90deg, #ff4da6, #ff8cb3);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 5px;
    }
    
    /* Subtítulo */
    .sub-title {
        text-align: center;
        color: #a0a0a0;
        font-size: 1rem;
        margin-bottom: 40px;
    }

    /* Estilizando a caixa de Upload (Glassmorphism) */
    [data-testid="stFileUploader"] {
        background: rgba(255, 255, 255, 0.02);
        border: 1px solid rgba(255, 77, 166, 0.2);
        border-radius: 20px;
        padding: 25px;
        box-shadow: 0 8px 32px 0 rgba(255, 77, 166, 0.05);
        backdrop-filter: blur(10px);
    }
    
    /* A área tracejada de arrastar arquivo */
    [data-testid="stFileUploadDropzone"] {
        border: 2px dashed rgba(255, 77, 166, 0.4) !important;
        border-radius: 15px !important;
        background-color: transparent !important;
    }
    
    /* O Botão de "Browse files" */
    [data-testid="stFileUploadDropzone"] button {
        background: linear-gradient(90deg, #ff1a8c, #ff4da6) !important;
        color: white !important;
        border-radius: 30px !important;
        border: none !important;
        padding: 10px 25px !important;
        font-weight: bold !important;
        transition: 0.3s;
    }
    [data-testid="stFileUploadDropzone"] button:hover {
        transform: scale(1.05);
        box-shadow: 0 0 15px rgba(255, 77, 166, 0.6);
    }

    /* Caixas de laudo */
    .report-box {
        background: rgba(20, 20, 25, 0.8);
        border: 1px solid #333;
        border-radius: 15px;
        padding: 20px;
        margin-top: 20px;
    }
    .alert-text {color: #ff4da6; font-weight: bold;}
    .safe-text {color: #00e676; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# --- ESTRUTURA VISUAL ---
st.markdown('<div class="top-bar">OncoDetect &nbsp;&nbsp; 🔍</div>', unsafe_allow_html=True)
st.markdown('<div class="pink-title">Sistema de Diagnóstico Oncológico - OncoDetect</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">Capstone Project: Detecção automática de lesões mamárias utilizando Deep Learning</div>', unsafe_allow_html=True)

# --- CARREGAR MODELO ---
try:
    model = YOLO('runs/detect/train/weights/best.pt')
except:
    st.error("Erro Crítico: Modelo 'best.pt' não encontrado.")
    st.stop()

# --- UPLOAD ---
arquivo = st.file_uploader(" ", type=['png', 'jpg', 'jpeg'])

if arquivo:
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img_orig = cv2.imdecode(file_bytes, 1)
    img_rgb = cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    col_e, col_d = st.columns(2)
    with col_e:
        st.markdown('<p style="color:#ff4da6; font-weight:bold; font-size:1.2rem;">Exame Original</p>', unsafe_allow_html=True)
        st.image(img_rgb, use_container_width=True)

    # --- INFERÊNCIA ---
    with st.spinner("Analisando matriz de pixels via YOLOv8..."):
        results = model(img_orig, conf=CONF_THRESHOLD, verbose=False)
        
        boxes_list, confidences_list, class_ids_list = [], [], []

        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                
                boxes_list.append([x1, y1, x2 - x1, y2 - y1])
                confidences_list.append(conf)
                class_ids_list.append(cls_id)

        indices = cv2.dnn.NMSBoxes(boxes_list, confidences_list, CONF_THRESHOLD, NMS_THRESHOLD)

        img_final = img_rgb.copy()
        n_lesoes = 0
        maior_conf = 0.0

        if len(indices) > 0:
            for i in indices.flatten():
                if class_ids_list[i] == 1: # Apenas câncer
                    n_lesoes += 1
                    conf = confidences_list[i]
                    if conf > maior_conf: maior_conf = conf
                    
                    x, y, w, h = boxes_list[i]
                    
                    # Desenho com a cor Rosa da interface (BGR para o OpenCV fica estranho, então no OpenCV usamos RGB já convertido)
                    cor_neon = (255, 77, 166) # Rosa Neon
                    cv2.rectangle(img_final, (x, y), (x + w, y + h), cor_neon, 3)
                    
                    label = f"SUSPEITO ({conf*100:.0f}%)"
                    (w_text, h_text), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    cv2.rectangle(img_final, (x, y - h_text - 10), (x + w_text, y), cor_neon, -1)
                    cv2.putText(img_final, label, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    with col_d:
        st.markdown('<p style="color:#ff4da6; font-weight:bold; font-size:1.2rem;">Processamento OncoDetect</p>', unsafe_allow_html=True)
        st.image(img_final, use_container_width=True)

    # --- LAUDO ---
    if n_lesoes > 0:
        st.markdown(f"""
        <div class="report-box" style="border-left: 5px solid #ff4da6;">
            <h4 style="color: #ff4da6;">⚠️ Relatório de Detecção IA</h4>
            <p>O algoritmo detectou <span class="alert-text">{n_lesoes} região(ões)</span> de interesse compatíveis com lesões suspeitas.</p>
            <ul>
                <li><b>Confiança do Modelo:</b> {maior_conf*100:.1f}%</li>
                <li><b>Classificação IA:</b> Padrão compatível com neoplasia.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="report-box" style="border-left: 5px solid #00e676;">
            <h4 style="color: #00e676;">✅ Rastreamento Negativo</h4>
            <p>A análise não detectou estruturas compatíveis com nódulos malignos na região processada.</p>
        </div>
        """, unsafe_allow_html=True)