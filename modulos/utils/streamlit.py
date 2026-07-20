# ==========================================================
# Módulos do projeto
# ==========================================================
import os
import streamlit as st
import pandas as pd
import tensorflow as tf
from pathlib import Path

from modulos.classificadores import cnn_teste, cnn_treino

# Função para execução do projeto
def run():
    # ==========================================================
    # CONFIGURAÇÃO DA PÁGINA
    # ==========================================================

    st.set_page_config(
        page_title="Sistema de Classificação por CNN",
        page_icon="🖼️",
        layout='wide',
        initial_sidebar_state='expanded'
    )

    # ==========================================================
    # TÍTULO
    # ==========================================================

    st.title('Sistema de Classificação de Imagens com CNN')

    st.markdown('''
    Este sistema permite utilizar um sistema de CNN já pré-treinada, neste caso utilizaremos
    a MobileNetV2, e utilizando a técnica de Transfer Learning faremos a extração de caracteristica 
    e a classificação das imagens.
    ''')

    # ==========================================================
    # CAMINNHOS E VARIAVEIS GLOBAIS
    # ==========================================================

    BASE_DIR = Path(__file__).resolve().parents[2]
    ALL_DATASETS_DIR = BASE_DIR / 'datasets'
    # Listar datasets
    datasets = sorted([p.name for p in ALL_DATASETS_DIR.iterdir() if p.is_dir()])
    SELECTED_DATASET = st.sidebar.selectbox(
        "Dataset",
        datasets,
        #index=datasets.index('coffee_bean') if "coffee_bean" in datasets else 0  # Caso tenha o coffee_bean ele setara como primeiro, se não ele pegara o priemiro da lista
    )    
    IMG_TRAIN_DIR = ALL_DATASETS_DIR / SELECTED_DATASET / 'train'
    IMG_VAL_DIR = ALL_DATASETS_DIR / SELECTED_DATASET / 'val'
    IMG_TEST_DIR = ALL_DATASETS_DIR / SELECTED_DATASET / 'test'
    LOG_DIR = BASE_DIR / 'log' / 'logger.csv'
    CHECKPOINTS_DIR = BASE_DIR / 'checkpoints' / 'cp.weights.h5'
    MODEL_CNN_DIR = BASE_DIR / 'modelos' / 'my_mobile_net_v2.keras'
    RESULTS_DIR = BASE_DIR / 'resultados'
    IMG_FORM = (224,224,3)
    METRIC = 'accuracy'
    BATCH_SIZE = 32
    EPOCHS = 20

    # ==========================================================
    # CONFIGURAÇÃO DAS ABAS
    # ==========================================================
    tab_treinamento, tab_teste = st.tabs(
        [
            'Treinamento',
            'Teste'
        ]
    )

    # ==========================================================
    # ABA - TREINAMENTO
    # ==========================================================
    with tab_treinamento:
        st.subheader('Treinamento CNN')

        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=BATCH_SIZE,
            value=BATCH_SIZE
        )

        epochs = st.number_input(
            "Número de épocas",
            min_value=1,
            max_value=EPOCHS,
            value=EPOCHS
        )

        metrica = st.selectbox(
            "Métrica",
            ['accuracy'],
            index=0
        )

        treinar = st.button(
            "Treinar modelo",
            type='primary',
            width='stretch'
        )

        # ==========================================================
        # TREINANDO MODELO
        # ==========================================================
        if treinar:
            with st.spinner("Treinando Modelo..."):
                # 1 - Obtendo imagens de treino
                imagens_treino = cnn_treino.obter_imagens_treino(
                    IMG_TRAIN_DIR,
                    IMG_FORM,
                    batch_size
                )

                # 2 - Obtendo imagens de validação
                imagens_validacao = cnn_treino.obter_imagens_validacao(
                    IMG_VAL_DIR,
                    IMG_FORM,
                    batch_size
                )

                # 3 - Obtendo modelo de CNN
                modelo, resumo_modelo = cnn_treino.obter_modelo_cnn(
                    imagens_treino.num_classes,
                    IMG_FORM
                )

                # 4 - Compilando o modelo da CNN
                modelo = cnn_treino.compilar_modelo_cnn(
                    modelo,
                    metrica
                )

                # 5 - Lista de Callbacks
                callbacks = [
                    cnn_treino.obter_checkpoints_callback(
                        CHECKPOINTS_DIR
                    ),

                    cnn_treino.obter_log_callbacks(
                        LOG_DIR
                    ),

                    cnn_treino.obter_reduce_lr_callbacks(),

                    cnn_treino.obter_early_stop_callback()
                ]

                # 6 - Treinando o Modelo
                modelo, historico = cnn_treino.treinar_modelo(
                    modelo,
                    imagens_treino,
                    imagens_validacao,
                    epochs,
                    CHECKPOINTS_DIR,
                    callbacks
                )

                # 7 - Salvando o modelo treinado
                cnn_treino.salvar_modelo(
                    modelo,
                    MODEL_CNN_DIR
                )

                # 8 - Gerando o gráfico
                cnn_treino.plot_historico(
                    historico,
                    metrica,
                    RESULTS_DIR
                )

                # 9 - Exibindo métricas no Streamlit
                st.success("Treinamento concluido!")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Accuracy Treino",
                        f"{historico.history['accuracy'][-1]*100:.2f}%"
                    )

                with col2:
                    st.metric(
                        "Accuracy Validação",
                        f"{historico.history['val_accuracy'][-1]*100:.2f}%"
                    )

                # 10 - Histórico de épocas
                st.subheader('Histórico de Treinamento')
                df = pd.read_csv(LOG_DIR)
                st.dataframe(
                    df,
                    width='stretch'
                )

                # 11 - Resumo do modelo
                st.subheader("Arquitetura da CNN")
                st.code(
                    resumo_modelo,
                    language='text'
                )
                

                # 11 - Abrir o gráfico
                arquivo = RESULTS_DIR / "Historico_Treinamento.png"
                if arquivo.exists():
                    st.image(
                        arquivo,
                        caption="Histórico do treinamento",
                        width=800
                    )