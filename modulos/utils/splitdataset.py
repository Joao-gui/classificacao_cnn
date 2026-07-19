# https://pypi.org/project/split-folders/
import splitfolders

input_path = 'pasta_de_entrada'

# A pasta de entrada deve ter o seguinte formato:
'''
entrada/
    classe1/
        img1.jpg
        img2.jpg
        ...
    classe2/
        imgX.jpg
        ...
    ...
'''

output_path = 'pasta_destino'

# A pasta de sáida vai ficar neste formato:
'''
saida/
    train/
        classe1/
            img1.jpg
            ...
        classe2/
            imgU.jpg
            ...
    val/
        classe1/
            imgV.jpg
            ...
        classe2/
            imgX.jpg
            ...
    test/
        classe1/
            imgY.jpg
            ...
        classe2/
            imgZ.jpg
            ...
'''

# A função 'splitfolders.ratio' é utilizada para dividir um conjunto de dados
# de uma pasta de entrada, com imagens por exemplo, em múltiplas pastas para,
# treinamento, validação e teste, com base em uma proporção especificada.

splitfolders.ratio(
    # 'input': Caminho para a pasta que contem as imagens a serem divididas
    input=input_path,
    # 'output': Caminho onde as pastas resultantes serão salvas
    output=output_path,
    # 'seed': Semente para o gerador de números aleatórios, garantido
    # a repordutibilidade da divisão dos dados.
    seed=42,
    # 'ratio': Proporção da divisão dos dados. Neste exemplo,
    # treinamento = 80% das imagens
    # validação = 10% das imagens
    # teste = 10% das imagens
    ratio=(0.8, 0.1, 0.1),
    # 'group_prefix': Se definido, agrupa arquivos que compartilham
    # o mesmo prefixo em uma única divisão. Isso é útil para garantir
    # que dados relacionados (como imagens da mesma classe) fiquem
    # juntos.
    group_prefix=None,
    # 'move': Se definido como 'True', move o arquivos em vez de
    # copialos.
    move=False
)