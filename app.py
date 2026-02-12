import streamlit as st
from ultralytics import YOLO
from PIL import Image
import cv2
import numpy as np

# --- Configuração da Página ---
st.set_page_config(page_title="Laudo IA - Final", layout="wide")

st.title("Sistema de Diagnóstico Oncológico - OncoDetect")
st.markdown("**Capstone Project:** Detecção automática de lesões mamárias utilizando Deep Learning.")

# --- Carregar Modelo ---
try:
    model = YOLO('runs/detect/train/weights/best.pt')
except:
    st.error("Erro: Modelo não encontrado.")
    st.stop()

# --- Mapeamento de Classes (Tradução) ---
# Baseado no seu teste: 1 = Tumor, 0 = Normal
NOMES_CLASSES = {
    0: "Tecido Mamario (Normal)",
    1: "Nodulo / Lesao Suspeita"
}

st.divider()
arquivo = st.file_uploader("Anexar Mamografia", type=['png', 'jpg', 'jpeg'])

if arquivo:
    # Converter para formato que o OpenCV gosta
    file_bytes = np.asarray(bytearray(arquivo.read()), dtype=np.uint8)
    img_cv2 = cv2.imdecode(file_bytes, 1)
    img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Exame Original")
        st.image(img_cv2, use_container_width=True)

    # --- INFERÊNCIA ---
    with st.spinner("Processando..."):
        results = model(img_cv2, conf=0.25)
        
        # Vamos desenhar nós mesmos para ter controle total das cores
        img_final = img_cv2.copy()
        alertas = []

        for box in results[0].boxes:
            classe_id = int(box.cls[0])
            confianca = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            nome_classe = NOMES_CLASSES.get(classe_id, "Desconhecido")
            
            # LÓGICA DE CORES E FILTRO
            if classe_id == 1: # CÂNCER (Mostrar em Vermelho)
                cor = (255, 0, 0) # Vermelho RGB
                cv2.rectangle(img_final, (x1, y1), (x2, y2), cor, 4)
                
                # Etiqueta com fundo vermelho
                texto = f"{nome_classe} ({confianca*100:.1f}%)"
                (w, h), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                cv2.rectangle(img_final, (x1, y1 - 30), (x1 + w, y1), cor, -1)
                cv2.putText(img_final, texto, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
                
                alertas.append((nome_classe, confianca))
            
            elif classe_id == 0: # NORMAL (Mostrar em Verde discreto ou Ignorar)
                # Vamos desenhar fininho em verde só pra saber que a IA viu a mama
                cor = (0, 255, 0) 
                cv2.rectangle(img_final, (x1, y1), (x2, y2), cor, 2)
                # Não adicionamos na lista de 'alertas' para não assustar o usuário

    with col2:
        st.subheader("Detecção Computacional")
        st.image(img_final, use_container_width=True)

    # --- LAUDO MÉDICO FINAL ---
    st.divider()
    st.subheader("📋 Laudo Técnico Preliminar")

    if len(alertas) > 0:
        for nome, conf in alertas:
            st.error(f"""
            ### ⚠️ DETECÇÃO POSITIVA: {nome}
            **Nível de Confiança da IA:** {conf*100:.1f}%
            
            **Análise:** O sistema identificou um padrão de alta densidade compatível com lesão maligna.
            Recomenda-se encaminhamento imediato para biópsia ou ultrassonografia complementar.
            """)
    else:
        st.success("""
        ### ✅ NEGATIVO PARA LESÕES SUSPEITAS
        O sistema analisou a imagem e identificou apenas tecido mamário dentro dos padrões de normalidade.
        **Observação:** Detectou-se estrutura mamária (Classe 0), mas nenhum nódulo isolado.
        """)