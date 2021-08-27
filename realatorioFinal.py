import streamlit as st 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF



st.set_page_config(
    page_title="Relatório Final",
    page_icon=":computer:",
    layout="centered",
    initial_sidebar_state="expanded",
)


# plt.rcParams['figure.figsize']  = (5, 5)
fig, ax = plt.subplots(figsize=(5,3))

dataset = pd.read_csv('./databases/adult_data.csv')

column_new_names = {}
for c in dataset.columns[1:]:
    column_new_names[c] = c.split(' ')[1].replace('-', '_')
    
dataset = dataset.rename(columns=column_new_names)


st.markdown("""

    # Relatório Final ICD - Adult Income

    A renda anual de um indivíduo pode resultar de vários fatores. 
    Intuitivamente, é influenciada pelo nível de educação do indivíduo, idade, sexo, ocupação e etc.
    O conjuto de dados escolhido, extraído do banco de dados do censo Estadunidense, 
    é uma boa fonte para desenvolver técnicas de processamento de dados, 
    aprendizado de máquina e visualização através 
    de dados demográficos e outros recursos para descrever uma pessoa.

    *Database*:
    Contém 16 colunas sendo 15 atributos e o Alvo da predição.

    Alvo: Renda, que é dividida em duas classes: <= 50K e > 50K. As pessoas que ganham mais de 50 mil dólares anuais e as que ganham menos.
""")
st.dataframe(dataset.head())

st.markdown("""
    ##
    Podemos explorar a possibilidade de prever o nível de renda com base nas informações pessoais do indivíduo e fazer
    as seguintes perguntas...
    """)

st.markdown(""" ## Perguntas
    - Existe algum atributo que é fortemente correlacionado com outros?
    - Os atributos "imutáveis" (sexo, raça, país de origem...) do trabalhador são menos ou mais determinantes que os outros (como escolaridade) para predizer se ele ganha acima de 50 mil dólares anuais?
    - Com qual precisão conseguimos prever o salário de um trabalhador, baseado em seus atributos, a partir do dataset selecionado? E baseado apenas nos grupos de atributos das perguntas anteriores?
    - De modo semelhante, qual a menor combinação de atributos que melhor prediz se  um trabalhador ganha mais de 50 mil dólares anuais?
""")

st.markdown("""
    ## Distribuição dos dados
""")

selecao = "age"
selecao = st.selectbox("Selecione um atributo:",["age", "workclass", "fnlwgt", "education", "education_num", "marital_status","occupation", "relationship", "race", "sex", "capital_gain", "capital_loss", "hours_per_week", "salary"])

try: 
    dataset[selecao].plot.hist(bins = 73)
    st.pyplot(plt)
    plt.clf()

except:
    plt.barh(dataset[selecao].value_counts().index, dataset[selecao].value_counts().values)
    st.pyplot(plt)
    plt.clf()


st.markdown(""" ## Variação observada nos grupos
    Os seguintes Gráficos visam observar a relação da renda com alguns dos atributos, através da distribuição.
""")

def plot_ECDF(sample, label):
    ecdf = ECDF(sample)
    plt.plot(ecdf.x, ecdf.y, label=label)

above = dataset.drop(dataset[(dataset.salary == ' >50K')].index)
below = dataset.drop(dataset[(dataset.salary == ' <=50K')].index)

col1,col2, col3=st.columns(3)

with col1:
    plot_ECDF(above['education_num'], ">50K")
    plot_ECDF(below['education_num'], "<=50K")
    plt.legend()
    plt.xlabel("Nível de educação")
    plt.ylabel("Percentual")
    plt.title("Educação")
    st.pyplot(plt)
    plt.clf()

with col2:

    plot_ECDF(above['age'], ">50K")
    plot_ECDF(below['age'], "<=50K")
    plt.legend()
    plt.xlabel("Anos")
    plt.ylabel("Percentual")
    plt.title("Idade")
    st.pyplot(plt)
    plt.clf()

with col3:
    plot_ECDF(above['hours_per_week'], ">50K")
    plot_ECDF(below['hours_per_week'], "<=50K")
    plt.legend()
    plt.xlabel("Horas Trabalhadas")
    plt.ylabel("Percentual")
    plt.title("Horas por semana")
    st.pyplot(plt)
    plt.clf()




