import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import json
import os

# --- Funções Auxiliares ---

@st.cache_resource
def carregar_modelo_e_classes():
    """
    Carrega o modelo Keras treinado e os nomes das classes.
    Usa o cache do Streamlit para evitar recarregar a cada interação.
    """
    model_path = 'saved_model/meu_modelo_gestos.keras'
    class_names_path = 'saved_model/class_names.json'

    if not os.path.exists(model_path) or not os.path.exists(class_names_path):
        st.error("Erro: Arquivo do modelo ou das classes não encontrado!")
        st.info("Por favor, execute o script 'train_model.py' primeiro para treinar e salvar o modelo.")
        return None, None

    try:
        modelo = tf.keras.models.load_model(model_path)
        with open(class_names_path, 'r') as f:
            nomes_classes = json.load(f)
        return modelo, nomes_classes
    except Exception as e:
        st.error(f"Ocorreu um erro ao carregar o modelo: {e}")
        return None, None

def preprocessar_imagem(image, target_size=(50, 50)):
    """
    Preprocessa a imagem carregada para o formato esperado pelo modelo.
    """
    # Garante que a imagem tenha 3 canais (RGB)
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Redimensiona a imagem
    image = image.resize(target_size)
    
    # Converte para array numpy e adiciona uma dimensão de batch
    img_array = np.array(image)
    img_array = np.expand_dims(img_array, axis=0) # Cria um batch de 1 imagem
    
    return img_array

# --- Interface da Aplicação ---

st.set_page_config(page_title="Reconhecimento de Gestos", page_icon="👋")

st.title("👋 Reconhecimento de Gestos com as Mãos")
st.write(
    "Faça o upload de uma imagem de um gesto com a mão e a inteligência artificial "
    "tentará adivinhar qual é. O modelo foi treinado para reconhecer 37 gestos diferentes."
)

# Carrega o modelo e as classes
modelo, nomes_classes = carregar_modelo_e_classes()

# Se o modelo foi carregado com sucesso, exibe o uploader de arquivo
if modelo is not None and nomes_classes is not None:
    uploaded_file = st.file_uploader(
        "Escolha uma imagem...",
        type=["jpg", "jpeg", "png"]
    )

    if uploaded_file is not None:
        # Exibe a imagem carregada
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(image, caption="Imagem Carregada", use_column_width=True)

        # Preprocessa a imagem e faz a predição
        with st.spinner('Analisando a imagem...'):
            img_array = preprocessar_imagem(image)
            predicao = modelo.predict(img_array)
            
            # Obtém a classe prevista e a confiança
            score = tf.nn.softmax(predicao[0])
            classe_prevista = nomes_classes[np.argmax(score)]
            confianca = 100 * np.max(score)

        with col2:
            st.subheader("Resultado da Análise")
            st.metric(label="Gesto Previsto", value=f"{classe_prevista}")
            st.metric(label="Nível de Confiança", value=f"{confianca:.2f}%")

            if confianca < 60:
                st.warning("A confiança da predição é baixa. O modelo pode estar incerto.")
            else:
                st.success("Predição realizada com sucesso!")
else:
    st.warning("A aplicação não pode iniciar até que o modelo seja treinado e salvo.")
