import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt

# Obtendo imagens para treino
def obter_imagens_treino(caminho_imagens_treino, forma_img, tam_lote):
    # Cria um gerador de imagens com aumento de dados (data augmentation)
    # vertical_flip e horizontal_flip permitem que as imagens sejam viradas
    # vertical e horizontalmente, o que ajuda a criar mais variação
    # nos dados de treino. rescale=1./255 normaliza os valores de
    # pixels para o intervalo [0, 1]
    gerador_imagens = keras.preprocessing.image.ImageDataGenerator(
        vertical_flip=True,
        horizontal_flip=True,
        rescale=1./255
    )

    # Carrega as imagens do diretório especificado e aplica o gerador de imagens
    # target_size redimensiona as imagens para o tamanho especificado 
    # (altura, largura).
    # batch_size define o número de imagens a serem carregadas por vez.
    # class_mode='categorical' significa que as classes serão representadas 
    # por vetores one-hot.
    # shuffle=True embaralha as imagens a cada época para evitar dependências 
    # entre as amostras.
    # seed=42 garante que o embaralhamento seja reproduzível.
    imagens_treino = gerador_imagens.flow_from_directory(
        caminho_imagens_treino,
        target_size=forma_img[:2],
        batch_size=tam_lote,
        class_mode='categorical',
        shuffle=True,
        seed=42
    )

    return imagens_treino

# Obtendo imagens de validação
def obter_imagens_validacao(caminho_imagens_validacao, forma_img, tam_lote):
    # Cria um gerador de imagens para a validação sem aumento de dados,
    # apenas com normalização dos valores dos pixels para o intervalo [0, 1].
    gerador_imagens = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255
    )

    # Carrega as imagens do diretório de validação e aplica o gerador de imagens
    # target_size redimensiona as imagens para o tamanho especificado 
    # (altura, largura).
    # batch_size define o número de imagens a serem carregadas por vez.
    # class_mode='categorical' significa que as classes serão representadas 
    # por vetores one-hot.
    # shuffle=True embaralha as imagens a cada época para evitar dependências entre as amostras.
    imagens_validacao = gerador_imagens.flow_from_directory(
        caminho_imagens_validacao,
        target_size=forma_img[:2],
        batch_size=tam_lote,
        class_mode='categorical',
        shuffle=True
    )

    return imagens_validacao

# Obtendo modelo de CNN
def obter_modelo_cnn(num_classes):
    # Define um seed para garantir que os resultados sejam reproduzíveis.
    tf.random.set_seed(42)

    # Define a forma das imagens de entrada: 224x224 pixels com 3 canais (RGB).
    forma_img=(224,224,3)

    # Carrega o modelo MobileNetV2 pré-treinado com pesos do ImageNet.
    # include_top=False significa que a camada final de classificação (top) 
    # não é incluída, permitindo que o modelo seja usado para extração de 
    # características ou para ser adaptado a uma nova tarefa de classificação.
    modelo_base = keras.applications.MobileNetV2(input_shape=forma_img,
                                                weights='imagenet',
                                                include_top=False)

    # Congela as camadas do modelo base, impedindo que os pesos do MobileNetV2 
    # sejam atualizados durante o treinamento.
    modelo_base.trainable=False

    # Define a camada de entrada que aceita imagens com a forma 
    # especificada (224x224x3).
    camada_entrada = keras.Input(shape=forma_img)

    # Passa as imagens de entrada através do modelo base, 
    # gerando uma saída de características.
    x = modelo_base(camada_entrada)

    # Aplica Batch Normalization, que normaliza a ativação da camada anterior.
    # Isso ajuda a acelerar o treinamento e a estabilidade do modelo.
    x = keras.layers.BatchNormalization()(x)

    # Aplica o GlobalAveragePooling2D, que reduz a dimensão espacial 
    # (altura e largura) da saída para um vetor, calculando a média dos 
    # valores em cada canal. Isso resulta em uma representação compacta 
    # da imagem, mantendo informações relevantes.
    x = keras.layers.GlobalAveragePooling2D()(x)

    # Adiciona uma camada de Dropout com taxa de 30%, que desativa 
    # aleatoriamente 30% dos neurônios durante o treinamento, ajudando a 
    # prevenir overfitting, onde o modelo se ajusta excessivamente
    # aos dados de treinamento.
    x = keras.layers.Dropout(0.3)(x)

    # Adiciona uma camada densa totalmente conectada com 64 neurônios.
    # Os parâmetros de regularização ajudam a penalizar pesos grandes, 
    # promovendo um modelo mais simples:
    # - kernel_regularizer=L2: Aplica regularização L2 nos pesos da camada.
    # A ativação 'relu' é usada para introduzir não-linearidade.
    x = keras.layers.Dense(64, kernel_regularizer=keras.regularizers.L2(0.001), activation='relu')(x)

    # Aplica outra camada de Dropout, desta vez com uma taxa de 50%, 
    # para maior regularização.
    x = keras.layers.Dropout(0.5)(x)

    # Adiciona a camada de saída com 'num_classes' neurônios, onde 
    # 'num_classes' é o número de classes para a tarefa de classificação.
    # A ativação 'softmax' é usada para converter as saídas em 
    # probabilidades de cada classe.
    camada_saida = keras.layers.Dense(num_classes, activation='softmax')(x)

    # Cria o modelo final, conectando a camada de entrada à camada de saída.
    modelo = keras.Model(camada_entrada, camada_saida)

    # Exibe um resumo do modelo, mostrando as camadas e o número de parâmetros treináveis.
    modelo.summary()

    return modelo

# Compilando modelo de cnn
def compilar_modelo_cnn(modelo, metrica):
    # define o otimizar para o cálculo de ajuste dos parâmetros/pesos do modelo
    otimizador = keras.optimizers.Adam(learning_rate=0.0001)
    # define a função para o cálculo do loss (erro entre o valor previsto e o esperado)
    loss=keras.losses.CategoricalCrossentropy()
    # o modelo é configurado com essas informações e está pronto para ser treinado
    modelo.compile(optimizer=otimizador, loss=loss, metrics=metrica)
    return modelo

# Treinando o modelo
def treinar_modelo(modelo, imagens_treino, imanges_validacao, num_epocas, caminho_checkpoints, lista_callbacks):
    # realiza o treinamento do modelo utilizando os dados de treinamento para o aprendizado
    # e os dados de validação para auxiliar no cálculos dos pesos do modelo a cada época
    # e executa as funções presentes na lista de callbacks
    historico = modelo.fit(imagens_treino, epochs=num_epocas, verbose=1, validation_data=imanges_validacao,
                        callbacks=lista_callbacks)
    # Os melhores pesos armazenados nos checkpoints são carregados no modelo
    modelo.load_weights(caminho_checkpoints)
    return modelo, historico

# Plotagem historico
def plot_historico(historico, metrica, caminho_resultados):
    # plota a evolução da acurácia e loss ao longo das épocas de treinamento
    plt.figure(1)
    # accuracy
    plt.subplot(211)
    plt.plot(historico.history[metrica])
    plt.plot(historico.history['val_'+metrica])
    plt.title('Acurácia do Modelo')
    plt.ylabel('Acurácia')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='lower right')
    # loss
    plt.subplot(212)
    plt.plot(historico.history['loss'])
    plt.plot(historico.history['val_loss'])
    plt.title('Perda do Modelo')
    plt.ylabel('Perda')
    plt.xlabel('Época')
    plt.legend(['Treinamento', 'Validação'], loc='upper right')
    plt.tight_layout()
    plt.savefig(caminho_resultados+'Historico_Treinamento', dpi=300)
    print(f'\nHistórico salvo na pasta: {caminho_resultados}\n')
    #plt.show()
    plt.close()

# Salvando modelo
def salvar_modelo(modelo, caminho_dest_modelo):
    modelo.save(caminho_dest_modelo)
    print(f'\nO modelo treinado foi salvo na pasta {caminho_dest_modelo}\n')

# Obtendo os callbacks
def obter_checkpoints_callback(caminho_checkpoints):
    # Criação do callback para pontos de verificação
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        filepath=caminho_checkpoints,
        save_weights_only=True,
        monitor='val_accuracy',
        mode='max',
        save_best_only=True
    )

    return checkpoint_callback

# Obtendo log dos callbacks
def obter_log_callbacks(caminho_log):
    # Criação do callback para log dos valores de acurácia e loss a cada época
    log_callback = keras.callbacks.CSVLogger(caminho_log, separator=',', append=False)
    return log_callback

# Reduce lr callback
def obter_reduce_lr_callbacks():
    # Reduz a tx de aprend se a mudança for menor do q o factor em val_los
    reduce_lr_callback = keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=1, #quantidade de épocas que espera para reduzir a tx de aprend
        verbose=1
    )
    return reduce_lr_callback

# Early stop
def obter_early_stop_callback():
    # Para o treinamento se val_loss parar de diminuir por 3 épocas consecutivas
    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='auto',
        patience=5,
        verbose=1,
        restore_best_weights=True
    )
    return early_stop