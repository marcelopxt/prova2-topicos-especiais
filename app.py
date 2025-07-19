import streamlit as st
import gdown
import tensorflow as tf
import io
from PIL import Image
import numpy as np
import pandas as pd
import plotly.express as px


# armazenar em cache pra não precisar baixar sempre que der refresh
@st.cache_resource

def carrega_modelo():
    # https://drive.google.com/file/d/1YAMjfUQ608juD1Ojc0qb08XMrToU5BWR/view?usp=sharing
    url = "https://drive.google.com/file/d/1-QRtWkdYJdUmU6xZyZ2PkwXSb7Q6PGZ0/view?usp=sharing"
    gdown.download(url, 'modelo_quantizado16bits.tflite') # baixa o arquivo (modelo)

    # carrega o modelo
    interpreter = tf.lite.Interpreter(model_path='modelo_quantizado16bits.tflite')

    # disponibiliza para uso
    interpreter.allocate_tensors()

    return interpreter

def carrega_imagem():
    uploaded_file = st.file_uploader("Arraste e solte a imagem ou clique para selecionar uma", type=['png', 'jpg', 'jpeg'])

    if uploaded_file is not None:
        # ler a imagem
        image_data = uploaded_file.read()

        # abrir a imagem
        image = Image.open(io.BytesIO(image_data))

        # exibir a imagem na página
        st.image(image)
        st.success("Imagem foi carregada com sucesso")

        # converter a imagem em ponto flutuante
        image = np.array(image, dtype=np.float32)

        # normalizar a imagem
        image = image/255.0

        # adicionar uma dimensão extra
        image = np.expand_dims(image, axis=0)
        
        return image
    
def previsao(interpreter, image):
    # Obtém os detalhes da entrada do modelo (por exemplo: shape, índice do tensor)
    input_details = interpreter.get_input_details()
    
    # Obtém os detalhes da saída do modelo
    output_details = interpreter.get_output_details()
    
    # Define a imagem (pré-processada) como entrada no modelo
    interpreter.set_tensor(input_details[0]['index'], image)
    
    # Executa a inferência (processo de predição) com o modelo TFLite
    interpreter.invoke()
    
    # Obtém a saída do modelo — geralmente, as probabilidades para cada classe
    output_data = Interpreter.get_tensor(output_details[0]['index'])

    # Lista com os nomes das classes, correspondentes à ordem da saída do modelo
    classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', '_']
    
    # Cria um DataFrame para exibir as classes e suas probabilidades
    df = pd.DataFrame()
    df['classes'] = classes
    df['probabilidades (%)'] = 100 * output_data[0]  # Converte para porcentagem
    
    # Cria um gráfico de barras horizontal com Plotly para exibir as probabilidades
    fig = px.bar(
        df,
        y='classes',
        x='probabilidades (%)',
        orientation='h',
        text='probabilidades (%)',
        title='Probabilidade de Gestos'
    )
    
    # Exibe o gráfico interativo na interface do Streamlit
    st.plotly_chart(fig)

def main():
    st.set_page_config(
        page_title="Classifica gestos"
    )
    st.write("# Classifica gestos")
    # carregar o modelo
    interpreter = carrega_modelo()


    # carregar a imagem
    image = carrega_imagem()

    # classificar a imagem
    if image is not None:
        previsao(interpreter, image)

if __name__ == "__main__":
    main()