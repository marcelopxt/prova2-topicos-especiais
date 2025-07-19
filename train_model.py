import tensorflow as tf
import pathlib
import json
import os
import numpy as np
import matplotlib.pyplot as plt
import zipfile

# --- 1. Montagem do Google Drive e Descompactação dos Dados ---
print("--- Iniciando o script de treinamento ---")

# Nota: As linhas a seguir são para uso em um ambiente Google Colab.
# Elas montarão seu Google Drive para acessar o arquivo zip.
try:
    from google.colab import drive
    drive.mount('/content/drive')
    
    # Caminho para o seu arquivo ZIP no Google Drive
    # AJUSTE ESTE CAMINHO se o seu arquivo estiver em outro lugar.
    zip_path = "/content/drive/MyDrive/Grad. Sistemas de Informação/5° Período/Tópicos Especiais em Desenvolvimento de Software/gestos.zip"
    
    # Diretório onde os arquivos serão extraídos
    extraction_dir = "gestos_extraidos"
    
    print(f"\n--- Descompactando {zip_path} ---")
    if not os.path.exists(zip_path):
        print(f"ERRO: O arquivo zip não foi encontrado em: {zip_path}")
        print("Por favor, verifique se o caminho para o arquivo no Google Drive está correto.")
        exit()

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extraction_dir)
    print(f"Arquivos extraídos com sucesso para a pasta '{extraction_dir}'")
    
    # O caminho para os dados de imagem, dentro da pasta extraída
    data_dir = pathlib.Path(extraction_dir) / 'Gesture Image Data'

except ImportError:
    # Bloco para ser executado se não estiver no Colab (execução local)
    print("\nAVISO: Não estamos em um ambiente Google Colab.")
    print("O script tentará usar a pasta 'gestos/Gesture Image Data' localmente.")
    data_dir = pathlib.Path('gestos/Gesture Image Data')


# --- 2. Configuração de Parâmetros e Caminhos ---
# Parâmetros para o treinamento
batch_size = 64
altura = 50
largura = 50
epocas = 50 # Aumentado para melhor convergência, mas o callback pode parar antes
validation_split = 0.2
seed = 568

# Criar diretório para salvar o modelo, se não existir
os.makedirs('saved_model', exist_ok=True)
print(f"Diretório de dados sendo usado: {data_dir}")
if not data_dir.exists():
    print(f"\nERRO: O diretório de dados '{data_dir}' não foi encontrado.")
    print("Por favor, certifique-se de que o arquivo zip foi extraído corretamente e contém a pasta 'Gesture Image Data'.")
    exit()


# --- 3. Carregamento e Preparação dos Dados ---
print("\n--- Carregando e preparando os datasets ---")

# Carregando o conjunto de dados de treino
treino = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset='training',
    seed=seed,
    image_size=(altura, largura),
    batch_size=batch_size
)

# Carregando o conjunto de dados de validação
validacao = tf.keras.utils.image_dataset_from_directory(
    data_dir,
    validation_split=validation_split,
    subset='validation',
    seed=seed,
    image_size=(altura, largura),
    batch_size=batch_size
)

# Obtendo os nomes das classes e o número de classes
class_names = treino.class_names
num_classes = len(class_names)
print(f"Classes encontradas: {class_names}")
print(f"Número de classes: {num_classes}")

# Salvando os nomes das classes em um arquivo JSON para uso posterior na aplicação
with open('saved_model/class_names.json', 'w') as f:
    json.dump(class_names, f)
print("Nomes das classes salvos em 'saved_model/class_names.json'")

# Otimizando a performance de carregamento dos dados
AUTOTUNE = tf.data.AUTOTUNE
treino = treino.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
validacao = validacao.cache().prefetch(buffer_size=AUTOTUNE)


# --- 4. Definição do Modelo (Arquitetura da CNN) ---
print("\n--- Construindo o modelo de CNN ---")

# Camada de aumento de dados (Data Augmentation) para evitar overfitting
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1)
])

# Construção do modelo sequencial
modelo = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(altura, largura, 3)),
    data_augmentation,
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compilando o modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

modelo.summary()


# --- 5. Treinamento do Modelo ---
print("\n--- Iniciando o treinamento ---")

# Callback para parar o treinamento se a acurácia de validação não melhorar
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10,
    restore_best_weights=True
)

history = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=epocas,
    callbacks=[early_stopping]
)


# --- 6. Avaliação e Salvamento do Modelo ---
print("\n--- Treinamento concluído. Avaliando e salvando o modelo. ---")

loss, acc = modelo.evaluate(validacao)
print(f"\nAcurácia final de validação: {acc:.4f}")

modelo.save('saved_model/meu_modelo_gestos.keras')
print("Modelo salvo com sucesso em 'saved_model/meu_modelo_gestos.keras'")


# --- 7. Visualização dos Resultados (Opcional) ---
def plota_resultados(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    epochs_range = range(len(acc))

    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Acurácia do Treino')
    plt.plot(epochs_range, val_acc, label='Acurácia da Validação')
    plt.legend(loc='lower right')
    plt.title('Acurácia de Treino e Validação')
    
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Custo do Treino')
    plt.plot(epochs_range, val_loss, label='Custo da Validação')
    plt.legend(loc='upper right')
    plt.title('Custo de Treino e Validação')
    
    plt.savefig('training_results.png')
    plt.show()

plota_resultados(history)
print("\nGráficos de resultado salvos em 'training_results.png'")
print("--- Script de treinamento finalizado ---")