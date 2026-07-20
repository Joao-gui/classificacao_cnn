
# ☕ Classificação de Grãos de Café utilizando CNN e Transfer Learning

## 📌 Sobre o projeto

Este projeto implementa um sistema de classificação de imagens utilizando **Redes Neurais Convolucionais (CNN)** e a técnica de **Transfer Learning**.

O objetivo é classificar imagens de grãos de café em diferentes categorias utilizando uma rede neural pré-treinada, a **MobileNetV2**, adaptada para uma nova tarefa de classificação.

A aplicação possui uma interface desenvolvida em **Streamlit**, permitindo:

- Seleção do dataset;
- Treinamento do modelo CNN;
- Visualização da arquitetura da rede;
- Acompanhamento das métricas de treinamento;
- Avaliação do modelo em dados de teste;
- Visualização de métricas de classificação e matriz de confusão.

---

# 🎯 Objetivo

Construir um pipeline completo de classificação de imagens contendo:

- Preparação do dataset;
- Carregamento das imagens;
- Data augmentation;
- Treinamento utilizando CNN;
- Avaliação do modelo;
- Geração de métricas;
- Interface interativa para utilização do modelo.

---

# 🧠 Modelo utilizado

O modelo base utilizado foi a:

## MobileNetV2

A MobileNetV2 foi treinada originalmente no dataset ImageNet e utilizada como extratora de características.

A arquitetura utilizada:

**Entrada:** 224x224x3

**MobileNetV2:** (include_top=False)

**Batch Normalization**

**Global Average Pooling**

**Dropout**

**Dense (64 neurônios)**

**Dropout**

**Camada de saída Softmax**

**Número de classes:** 4 classes

**Classes:**

* Dark
* Green
* Light
* Medium

---

# 📂 Estrutura do projeto

```
classificacao_cnn
│
├── checkpoints/                      # Arquivo .h5 com os checkpoints do modelo
│
├── datasets/                         # Pasta com os datasets separados por train, test e validation nas proporções (80%, 10%, 10%)
│   ├── coffee_bean/
│   │   ├── test/
│   │   │   ├── Dark/
│   │   │   ├── Green/
│   │   │   ├── Light/
│   │   │   ├── Medium/
```
