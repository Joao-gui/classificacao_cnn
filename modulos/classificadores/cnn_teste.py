import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import numpy as np
import seaborn as sns
import os
import sys

# Obtendo imagens de teste
def obter_imagens_teste(caminho_imagens_teste, forma_img, tam_lote):
    # Cria um gerador de imagens para o conjunto de teste
    # rescale=1./255 normaliza os valores dos pixels para o intervalo de [0, 1]
    gerador_imagens = keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

    # Carrega as imagens do diretório de teste e aplica o gerador de imagens
    # target_size redimensiona as imagens para o tamanho especifícado
    # (altura, largura)
    # batch_size indica quantas imagens serão processadas por vez
    # se o teste for realizado com uma única imagem, o batch_size pode
    # ser igual a 1
    # class_mode='categorical' significa que as classes serão representadas
    # por vetores one-hot.
    # shuffle=False garante que a ordem dos rótulos verdadeiros correspondem à
    # dos rótulos previstos
    imagens_teste = gerador_imagens.flow_from_directory(
        caminho_imagens_teste,
        target_size=forma_img[:2],
        batch_size=tam_lote,
        class_mode='categorical',
        shuffle=False
    )

    return imagens_teste

# Carregando modelo treinado cnn
def carregar_modelo_cnn_treinado(caminho_modelo_cnn_treinado):
    # Verifica se o arquivo do modelo existe
    if not os.path.exists(caminho_modelo_cnn_treinado):
        # Se o arquivo não for encontrado, imprime uma mensagem de erro e interrompe a execução do notebook
        print(f'Arquivo não encontrado: {caminho_modelo_cnn_treinado}')
        sys.exit("Execução interrompida: Arquivo do modelo não encontrado.")

    # Carrega o modelo salvo em arquivo no formato .keras
    modelo = keras.models.load_model(caminho_modelo_cnn_treinado)
    return modelo

# Testando o modelo
def testar_modelo(modelo, imagens_teste):
    # Gera previsões usando modelo
    previsoes = modelo.predict(imagens_teste, verbose=2)

    # Converte as previsões em rótulos. Cada linha da previsão contém
    # probabilidades para cada classe, e np.argmax() retorna o ínidice da
    # classe com maior probabilidade
    rotulos_previstos = np.argmax(previsoes, axis=1)

    return rotulos_previstos

# Matriz de confusão
def matriz_confusao(rotulos_verdadeiros, rotulos_previstos, nomes_das_classes, caminho_resultados):
    # Calcula a matriz de confusão
    conf_matrix = confusion_matrix(rotulos_verdadeiros, rotulos_previstos)

    # Calcula a acurácia do modelo em percentual
    acuracia = accuracy_score(rotulos_verdadeiros, rotulos_previstos)*100

    # Gera a figura da matriz de confusão usando Seaborn
    plt.figure(figsize=(10,8))
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')

    # Adiciona os nomes das classes, rótulos, título, subtítulo
    plt.xticks(np.arange(len(nomes_das_classes))+0.5, nomes_das_classes, rotation=0, ha='right')
    plt.yticks(np.arange(len(nomes_das_classes))+0.5, nomes_das_classes, rotation=0, va='center')
    plt.xlabel('Rótulos Previstos')
    plt.ylabel('Rótulos Reais')
    plt.title('Matriz de confusão', fontsize=18, weight='bold', x=0.5, y=1.05)
    plt.suptitle(f"Acurácia do Modelo: {acuracia:.2f}%", fontsize=14, x=0.435, y=0.92)
    plt.tight_layout()
    plt.savefig(caminho_resultados /'Matriz_Confusao', dpi=300)
    print(f'\nMatriz salva na pasta {caminho_resultados}\n')
    #plt.show()
    plt.close()

# Função para criar e salvar um gráfico para uma métrica escolhida
def plot_metrica(rotulos_verdadeiros, rotulos_previstos, nome_metrica, nomes_das_classes, caminho_resultados):
    # Obtém um relatório da classificação da scikit-learn como um dicionário
    report_dict = classification_report(rotulos_verdadeiros, rotulos_previstos,
                                        target_names=nomes_das_classes, output_dict=True)
    # cria um dicionário para a métrica de interesse (precision, recall ou f1-score)
    metric_data = {
        # cahve-valo = nome da classe: valor da métrica
        class_name: metrics[nome_metrica] 
        for class_name, metrics in report_dict.items()
        # apenas as aclasses da lista class_names
        # serão incluídas no dicionário metric_data
        if class_name in nomes_das_classes
    }
    sns.barplot(x=list(metric_data.keys()), y=list(metric_data.values()))
    plt.xlabel('Classes')
    plt.ylabel(nome_metrica.capitalize())
    plt.title(f'{nome_metrica.capitalize()} por classe')
    plt.xticks(rotation=0)
    # Adiciona rótulos de texto para os valores máximos em cada barra
    for index, value in enumerate(metric_data.values()):
        plt.text(index, value, str(round(value, 2)), ha='center', va='bottom')
    # Salvar figura
    plt.savefig(caminho_resultados / nome_metrica, dpi=300)
    print(f'\nMétrica salva na pasta: {caminho_resultados}\n')
    #plt.show()
    plt.close()