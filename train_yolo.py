from ultralytics import YOLO
from roboflow import Roboflow

print("⬇Iniciando download do Dataset...")

rf = Roboflow(api_key="E2jPBNHRFqOb240NE0BP")
project = rf.workspace("study-rjibi").project("breastcancer-rvcxu-0md8g")
version = project.version(1)
dataset = version.download("yolov8")
                


print("Carregando o modelo YOLOv8n (Nano)...")
# Carrega o modelo pré-treinado 
model = YOLO('yolov8n.pt') 

print("🔥 Começando o treinamento... (Isso pode demorar um pouco)")

# Inicia o treino
# data: aponta para o arquivo yaml baixado pelo Roboflow
# epochs: 50 rodadas de estudo 
# imgsz: tamanho da imagem 640 é o padrão da YOLO
try:
    results = model.train(
        data=f"{dataset.location}/data.yaml", 
        epochs=50, 
        imgsz=640,
        plots=True # Gera gráficos de performance no final
    )
    print("\nTreino finalizado com SUCESSO!")
    print(f"-> O modelo final está salvo em: runs/detect/train/weights/best.pt")
    
except Exception as e:
    print("\nOcorreu um erro durante o treino:")
    print(e)
    print("Dica: Verifique se o código do Roboflow foi colado corretamente.")