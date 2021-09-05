import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import math
from random import sample
import streamlit as st 
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.distributions.empirical_distribution import ECDF
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import zscore
import copy
from imblearn.under_sampling import RandomUnderSampler

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

st.markdown("""Pergunta:

- Existe algum atributo que é fortemente correlacionado com outros?



""")


# st.image('LogoD3.jpeg')

# st.markdown(""" Com base na imagem, observamos que não.""")

st.markdown("""# Teste de Hipóteses

Podemos ver no primeiro gráfico que existe uma diferença de escolaridade entre os grupos de pessoas que ganham mais ou menos que 50k por ano. 


Uma pergunta que pode ser feita é: "Será que a diferença no nível de educação do grupo que ganha mais de 50k pode ser explicada pelo acaso?"

Com isso podemos definir uma "Hipótese" e tentar verificá-la.


* Hipótese Nula: O acaso justifica a diferença de npiveis de educação nos grupos que ganham mais ou menos de 50k

""")

# Quantidade de pessoas com salário alto e baixo
print(pd.DataFrame({'count': dataset.salary.value_counts(), '%': dataset.salary.value_counts(normalize = True)}))

def total_variation(p, q):
    return np.sum(np.abs(p - q)) / 2


def sample_proportion100(pop_size, prop, n=10000):
    assert(prop >= 0)
    assert(prop <= 1)
    
    grupo = pop_size * prop
    # print(grupo)
    resultados = np.zeros(n)
    for i in range(n):
        sample = np.random.randint(0, pop_size, 100)
        resultados[i] = np.sum(sample < grupo)
    return resultados

if 1:
    column_name = 'education'
    secondary = 'education_num'
    
    #Prepara o dataset
    new_dataset = dataset.groupby([column_name,secondary , 'salary'])["education_num"].count().reset_index(name="count")

    educations = dict(zip(new_dataset['education'], new_dataset['education_num']))

    all_educations = list()
    for i in sorted(educations, key = educations.get):
        all_educations.append(i)

    key_list = all_educations
    value_list = np.zeros(len(all_educations))

    education_less_dict = dict(zip(key_list, value_list))
    education_more_dict = dict(zip(key_list, value_list))
    education_all_dict = dict(zip(key_list, value_list))

    for item in new_dataset.itertuples():
        if item[3] == " <=50K":
            education_less_dict[item[1]] = item[4]
        else:
            education_more_dict[item[1]] = item[4]
        education_all_dict[item[1]] = education_all_dict[item[1]] + item[4]

    fraction_less = dict(zip(key_list, value_list))
    fraction_more = dict(zip(key_list, value_list))
    for key in education_all_dict:
        fraction_less[key] = (education_less_dict[key] / education_all_dict[key])*100
        fraction_more[key] = (education_more_dict[key] / education_all_dict[key])*100

    idx = list()
    for i in education_all_dict.keys():
        idx.append(i)
    df = pd.DataFrame(index=idx)

    prop_pop_all = list()
    for i in education_all_dict.values():
        prop_pop_all.append(i/32561)

    df['sample'] = prop_pop_all

    # Na amostra de mais de 50K
    prop_pop_more = list()
    for i in education_more_dict.values():
        prop_pop_more.append(i/7841)

    df['pop'] = prop_pop_more

    N = 1453
    uma_amostra = []
    for g in df.index:
        p = df.loc[g]['pop']
        s = sample_proportion100(N, p, 1)[0]
        uma_amostra.append(s/100)

    df['1random'] = uma_amostra
    plt.ylabel('Propopção')
    plt.ylabel('Grupo')
    df.plot.bar()
    st.pyplot(plt)
    plt.clf()

    total_variation(df['1random'], df['pop'])
    total_variation(df['sample'], df['pop'])

    N = 10000
    A = np.zeros(shape=(10000, len(df.index)))
    for i, g in enumerate(df.index):
        p = df.loc[g]['pop']
        A[:, i] = sample_proportion100(N, p) / 100

    all_distances = []
    for i in range(A.shape[0]):
        all_distances.append(total_variation(df['pop'], A[i])) # total_variation entra a população e 

    plt.hist(all_distances, bins=30, edgecolor='k')
    plt.ylabel('Numero de Amostras de Tamanho 10k')
    plt.xlabel('Total Variation Distance')
    plt.plot([0.255407051893586], [0], 'ro', ms=15)
    st.pyplot(plt)
    plt.clf()

    st.markdown("""
        Com o histograma da TVD vemos no ponto vermelho qual seria o valor selecionando o grupo que ganha mais de 50k, as barras mostram as diferentes amostras aleatórias.
        Como o valor é bastante. Rejeitamos a hipótese nula e indicamos que a diferença ente a educação nos grupos que ganham mais ou menos de 50k não pode ser explicada pelo acaso.
    """)
    
    # st.write("97.5% da normal ->" ,np.percentile(all_distances, 97.5))


st.markdown("""## Diferença entre gêneros
Queremos demonstrar se o fato de um gênero ganhar mais que o outro pode ou não
ser justificado pelo acaso.

Assuma uma significância de 5%.

Hipótese Nula: O acaso justifica a diferença de salário entre os grupos.

Com base nessas informações, foi realizado um teste de permutação com a diferença entre homens e mulheres
que ganham abaixo de 50K dólares.
""")


def permutation_test(df, sig, rep_num, col, attribute):
 
    N = rep_num
    filtro = df[col] == attribute
    diffs = np.zeros(N)
    sig = sig/2
    for i in range(N):
        np.random.shuffle(filtro.values)
        diffs[i] = df.loc[df['salary'] == ' <=50K'][filtro].shape[0] - df.loc[df['salary'] == ' <=50K'][~filtro].shape[0]
    LI = np.percentile(diffs, sig)
    LS = np.percentile(diffs, 100-sig)
  
    return diffs, LI, LS
  

sigma = 5
rep_num = 5000
column= "sex"
at = " Male"
difference_vector, LI,LS = permutation_test(dataset, sigma, rep_num,column,at)
ab = dataset.query('salary == " <=50K"')
male = ab[ab.sex == ' Male'].count().iloc[0]
female = ab[ab.sex == ' Female'].count().iloc[0]
sample_difference = male-female
 
 
plt.hist(difference_vector, bins=30, edgecolor='k')
plt.plot(sample_difference, [0], 'ro', ms=15)
plt.ylabel('Quantidade')
plt.xlabel('Diferença entre o número de homens e mulheres que ganham abaixo de 50K por ano')
st.pyplot(plt)

st.markdown("""
Dessa maneira, rejeitamos a hipótese nula.
Além disso, foi feito o mesmo processo com casados e não casados que ganham menos de 50K dólares por ano.
Em ambos os casos, a hipótese nula foi rejeitada.  
""")

plt.clf()
sigma = 5
rep_num = 5000
column= "marital_status"

target_values = [' Divorced', ' Married-civ-spouse']
df = dataset.query("marital_status == ' Never-married' or marital_status == ' Married-civ-spouse'")


#new_dataset = dataset.loc[(dataset['marital_status'].isin(target_values))]
at = " Married-civ-spouse"
diff1, li1,ls1 = permutation_test(df, sigma, rep_num,column,at)
ab = dataset.query('salary == " <=50K"')
nm = ab[ab.marital_status == ' Married-civ-spouse'].count().iloc[0]
mcs = ab[ab.marital_status == ' Never-Married'].count().iloc[0]
sample_difference = nm-mcs
 
plt.hist(diff1, bins=10, edgecolor='k')
plt.ylabel('Quantidade')
plt.xlabel('Diferença entre o número de Casados e Não Casados que ganham abaixo de 50K Dol')
plt.plot(sample_difference, [0], 'ro', ms=15)
st.pyplot(plt)
plt.clf()


def bootstrap(x, y, model, n=5000):
  size = len(x)
  values = np.zeros(n)
  idx = np.arange(len(y))
  for i in range(n):
    sample = np.random.choice(idx, size=size, replace=True)
    values[i] = model.score(x.values[sample], y.values[sample])
  return values


def balance_data (X, y,random_state=42):

    rus = RandomUnderSampler(random_state=random_state)
    X, y = rus.fit_resample(X, y)

    return X, y
    

st.markdown(""" # Machine Learning

Nessa parte, diversos modelos de previsão de salário foram construídos, utilizando diferentes conjuntos de características do dataset. Foram consideradas apenas duas técnicas
de classificação de ML, KNN e Regressão Logística. Portanto, é possível que outras técnicas sejam melhores ou mais adequadas para aumentar
a eficiência dos modelos de previsão.\n
Com isso, foram obtidas respostas para duas das perguntas feitas na proposta do trabalho.

* Os atributos "imutáveis" (sexo, raça, país de origem e idade) do trabalhador são menos ou mais determinantes que os outros (como escolaridade) para predizer se ele ganha acima de 50 mil dólares anuais?

Para responder essa pergunta, construímos dois modelos. O primeiro realizava a previsão apenas com os parâmetros supracitados. O outro modelo realizava a previsão com os parâmetros não "imutáveis". Após analisar os resultados,
foi constatado que o modelo que previa com base nos atributos "mutáveis" teve quase a mesma taxa de acerto, e intervalo de confiança próximo ao modelo construído com todas as features. Dessa forma, concluímos que as características mutáveis,
e.g, horas trabalhadas, nível de escolaridade, status marital, dentre outros, são mais importantes para determinar o salário de um indivíduo do que seu país de origem, sexo e raça. 
\n
*Modelo com atributos imutáveis*:
""")

##ML dos imutáveis

df = dataset.copy()

x = df.drop(columns=['workclass', 'fnlwgt', 'education', 'education_num', 'marital_status', 'occupation', 'relationship', 'capital_gain', 'capital_loss', 'hours_per_week'], axis=1)
y = x['salary'].copy()
x.drop(columns='salary', axis=1, inplace=True)
x, y = balance_data(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


numerical_categories = list()
for coluna in x.columns:
  if (x.dtypes[coluna] == 'object'):
    continue
  else:
    numerical_categories.append(coluna)

x_trn = x_train.copy()
x_trn[numerical_categories] -= x_train[numerical_categories].mean()
x_trn[numerical_categories] /= x_train[numerical_categories].std(ddof=1)
##x_test_Znormalizado.
x_ten = x_test.copy()
x_ten[numerical_categories] -= x_train[numerical_categories].mean()
x_ten[numerical_categories] /= x_train[numerical_categories].std(ddof=1)


x_trn = pd.get_dummies(x_trn)
x_ten = pd.get_dummies(x_ten)
#Removing extra columns
a = set()
b = set()

for item in x_trn.columns:
  a.add(item)
for item  in x_ten.columns:
  b.add(item)

c = a-b
d = b-a

for item in c:
    x_trn.drop(item, axis=1,inplace=True)
for item in d:
    x_ten.drop(item, axis=1, inplace=True)

lr = LogisticRegressionCV(max_iter=10000)
lr.fit(x_trn, y_train)

precisao_final = lr.score(x_ten, y_test)


st.write("Precisão Final: ", precisao_final)
boots = bootstrap(x_ten, y_test, lr)
li = np.percentile(boots, 5)
ri = np.percentile(boots, 95)
brincante = f'Intervalo de Confiança com 5% de significância: [{li}, {ri}]'
st.write(brincante)
plt.hist(boots, bins=30, edgecolor='k')

st.pyplot(plt)
plt.clf()



st.markdown("""\n*Modelo com atributos mutáveis*""")


df = dataset.copy()
x = df.drop(columns=['age', 'sex', 'native_country', 'race'], axis=1)
y = x['salary'].copy()
x.drop(columns='salary', axis=1, inplace=True)
x, y = balance_data(x, y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.25)


numerical_categories = list()
for coluna in x.columns:
  if (x.dtypes[coluna] == 'object'):
    continue
  else:
    numerical_categories.append(coluna)

x_trn = x_train.copy()
x_trn[numerical_categories] -= x_train[numerical_categories].mean()
x_trn[numerical_categories] /= x_train[numerical_categories].std(ddof=1)
##x_test_Znormalizado.
x_ten = x_test.copy()
x_ten[numerical_categories] -= x_train[numerical_categories].mean()
x_ten[numerical_categories] /= x_train[numerical_categories].std(ddof=1)


x_trn = pd.get_dummies(x_trn)
x_ten = pd.get_dummies(x_ten)
#Removing extra columns
a = set()
b = set()

for item in x_trn.columns:
  a.add(item)
for item  in x_ten.columns:
  b.add(item)

c = a-b
d = b-a

for item in c:
    x_trn.drop(item, axis=1,inplace=True)
for item in d:
    x_ten.drop(item, axis=1, inplace=True)

lr = LogisticRegressionCV(max_iter=10000)
lr.fit(x_trn, y_train)

precisao_final = lr.score(x_ten, y_test)


st.write("Precisão Final: ", precisao_final)
boots = bootstrap(x_ten, y_test, lr)
li = np.percentile(boots, 5)
ri = np.percentile(boots, 95)
brincante = f'Intervalo de Confiança com 5% de significância: [{li}, {ri}]'
st.write(brincante)
plt.hist(boots, bins=30, edgecolor='k')

st.pyplot(plt)
plt.clf()


st.markdown("""

* Com qual precisão conseguimos prever o salário de um trabalhador, baseado em seus atributos, a partir do dataset selecionado? E baseado apenas nos grupos de atributos das perguntas anteriores?

Para responder, primeiros fizemos uma comparação entre as duas técnicas através do split do conjunto de treino em treino e validação.
Após essa etapa, selecionamos o modelo de maior desempenho (Logistic Regression) e calculamos a precisão com base na predição do modelo no conjunto de teste.
Tendo isso em vista, a precisão do modelo foi de:
""")

# Regressão com todos os atributos
df = dataset.copy()
numerical_categories = list()
for coluna in df.columns:
  if (df.dtypes[coluna] == 'object'):
    continue
  else:
    numerical_categories.append(coluna)

y = df['salary']
x = df.copy()
x_train = x.drop('salary', axis=1)
x_train, y = balance_data(x_train, y)

x_train, x_test, y_train, y_test = train_test_split(x_train, y, test_size = 0.25)

x_trn = x_train.copy()
x_trn[numerical_categories] -= x_train[numerical_categories].mean()
x_trn[numerical_categories] /= x_train[numerical_categories].std(ddof=1)
##x_test_Znormalizado.
x_ten = x_test.copy()
x_ten[numerical_categories] -= x_train[numerical_categories].mean()
x_ten[numerical_categories] /= x_train[numerical_categories].std(ddof=1)
##Fazendo one-hot-encoding.
x_trn = pd.get_dummies(x_trn)
x_ten = pd.get_dummies(x_ten)
#Removing extra columns
a = set()
b = set()

for item in x_trn.columns:
  a.add(item)
for item  in x_ten.columns:
  b.add(item)

c = a-b
d = b-a

for item in c:
    x_trn.drop(item, axis=1,inplace=True)
for item in d:
    x_ten.drop(item, axis=1, inplace=True)

lr = LogisticRegressionCV(max_iter=10000)
lr.fit(x_trn, y_train)

precisao_final = lr.score(x_ten, y_test)


st.write("Precisão Final: ", precisao_final)
boots = bootstrap(x_ten, y_test, lr)
li = np.percentile(boots, 5)
ri = np.percentile(boots, 95)
brincante = f'Intervalo de Confiança com 5% de significância: [{li}, {ri}]'
st.write(brincante)
plt.hist(boots, bins=30, edgecolor='k')

st.pyplot(plt)
plt.clf()