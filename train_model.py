
import tensorflow as tf
import pathlib
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# --- 1. Configuração de Parâmetros e Caminhos ---
print("--- Iniciando o script de treinamento ---")

# Caminho para a pasta de dados descompactada
data_dir = pathlib.Path('gestos/Gesture Image Data')

# Parâmetros para o treinamento
batch_size = 64
altura = 50
largura = 50
epocas = 50 # Aumentado para melhor convergência, mas o callback pode parar antes
validation_split = 0.2
seed = 568

# Criar diretório para salvar o modelo, se não existir
os.makedirs('saved_model', exist_ok=True)
print(f"Diretório de dados: {data_dir}")
if not data_dir.exists():
    print("\nERRO: O diretório de dados 'gestos/Gesture Image Data' não foi encontrado.")
    print("Por favor, certifique-se de que você descompactou o arquivo 'gestos.zip' na pasta correta.")
    exit()


# --- 2. Carregamento e Preparação dos Dados ---
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


# --- 3. Definição do Modelo (Arquitetura da CNN) ---
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
    # Camada de entrada com o formato das imagens (50x50 pixels, 3 canais de cor)
    tf.keras.layers.Input(shape=(altura, largura, 3)),
    
    # Aplicando o aumento de dados
    data_augmentation,
    
    # Normalizando os valores dos pixels de [0, 255] para [0, 1]
    tf.keras.layers.Rescaling(1./255),
    
    # Primeira camada de convolução e max pooling
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Segunda camada de convolução e max pooling
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Terceira camada de convolução e max pooling
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    
    # Achatando os mapas de características para uma dimensão
    tf.keras.layers.Flatten(),
    
    # Camada densa (totalmente conectada) com regularização Dropout
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    
    # Camada de saída com neurônios para cada classe e ativação softmax
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Compilando o modelo
modelo.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Exibindo um resumo da arquitetura do modelo
modelo.summary()


# --- 4. Treinamento do Modelo ---
print("\n--- Iniciando o treinamento ---")

# Callback para parar o treinamento se a acurácia de validação não melhorar
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=10, # Número de épocas sem melhora antes de parar
    restore_best_weights=True # Restaura os pesos da melhor época
)

history = modelo.fit(
    treino,
    validation_data=validacao,
    epochs=epocas,
    callbacks=[early_stopping] # Adicionando o callback
)


# --- 5. Avaliação e Salvamento do Modelo ---
print("\n--- Treinamento concluído. Avaliando e salvando o modelo. ---")

# Avaliando o modelo com os dados de validação
loss, acc = modelo.evaluate(validacao)
print(f"\nAcurácia final de validação: {acc:.4f}")

# Salvando o modelo treinado no formato Keras
modelo.save('saved_model/meu_modelo_gestos.keras')
print("Modelo salvo com sucesso em 'saved_model/meu_modelo_gestos.keras'")


# --- 6. Visualização dos Resultados (Opcional) ---
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
```python