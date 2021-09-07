import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.cluster import KMeans
from multiprocessing import Process
import matplotlib.pyplot as plt
import seaborn as sns
import SessionState
import time
import psutil

st.sidebar.title("Controls")
stop = st.sidebar.button("Stop")

state = SessionState.get(pid=None)


def job():
    for _ in range(100):
        print("In progress")
        time.sleep(1)

if stop:
    p = psutil.Process(state.pid)
    p.terminate()
    st.write("Stopped process with pid:", state.pid)
    state.pid = None


if st.button("Carregar dados"):
    def exec_full(filepath):
        global_namespace = {
            "__file__": filepath,
            "__name__": "__main__",
        }
        with open(filepath, 'rb') as file:
            exec(compile(file.read(), filepath, 'exec'), global_namespace)

    # Execute the file.
    exec_full("dados.py")

    #import dados.py
    #execfile(r'C:\Users\user\streamlit\dados.py') 
    #exec(open("dados.py").read())

    

    #balas = pd.read_csv(r"C:\Users\user\streamlit\Dados_tratados.csv",
    #                    sep = ",",
    #                    decimal=',',
    #                    encoding='latin-1',
    #                    engine='python',)
    #st.table(balas)
    #st.info('Sucesso')

#####################
st.write("""
# Previsões de desvios de custos
""")

st.write(" Previsões para uma obra")

st.markdown("""
<style>
body {
    color: #466e;
    etc. 
}
</style>
    """, unsafe_allow_html=True)


st.sidebar.header('Parâmetros')
# sidebar(inicio,fim,defaut_value)

df1 = pd.read_csv(r"C:\Users\Zé\Python\joao\streamlit\data\Dados_tratados.csv",
                      sep = ",",
                      decimal=',',
                      encoding='latin-1',
                      engine='python',
                     )

df_tratados = pd.read_csv(r"C:\Users\Zé\Python\joao\streamlit\data\df_tratado.csv",
                      sep = ",",
                      decimal=',',
                      encoding='latin-1',
                      engine='python',
                     )

from sklearn.model_selection import train_test_split

X = df1.drop(['Desvios_custo_total_categorico'] , axis=1)
y = df1["Desvios_custo_total_categorico"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.0001, random_state=42)


def user_input_features():



    Custos_previstos_orc = st.sidebar.number_input('Custos previstos orçamento')
##################################
    Custos_previstos_obra = st.sidebar.number_input('Custos previstos obra')
    Custos_prev_subemp = st.sidebar.number_input('Custos previstos subempreiteiros')
    Facturação_prevista_orc = st.sidebar.number_input('Faturação prevista orçamento')
    Facturação_prevista_obra = st.sidebar.number_input('Faturação prevista obra')
    Custos_prevmat_obra = st.sidebar.number_input('Custos previstos materiais obra')
    Custos_prevserv_obra = st.sidebar.number_input('Custos previstos serviços obra')
    dias = st.sidebar.slider('Duração da obra em dias (se não souber por a 0)', 0, 0, 700)
  
######################################
    valores_Tipoobra = df_tratados["tipoobra"].unique()
    Tipoobra_lista = list(valores_Tipoobra)

    Tipoobra = st.sidebar.selectbox("Tipo de obra", Tipoobra_lista)
######################################
    valores_Directorobra = df_tratados["directorobra"].unique()
    Directorobra_lista = list(valores_Directorobra)

    Directorobra = st.sidebar.selectbox("Diretor de obra", Directorobra_lista)
#######################################
    valores_Area = df_tratados["area"].unique()
    Area_lista = list(valores_Area)

    Area = st.sidebar.selectbox("Área", Area_lista)
#######################################
    valores_Vendedor_nome = df_tratados["vendedor_nome"].unique()

    Vendedor_nome_lista = list(valores_Vendedor_nome)

    Vendedor_nome = st.sidebar.selectbox("Comercial", Vendedor_nome_lista)
#######################################
    data = {'custos_previstos_orc': Custos_previstos_orc,
            'custos_previstos_obra': Custos_previstos_obra,
            'custos_prev_subemp': Custos_prev_subemp,
            'facturação_prevista_orc': Facturação_prevista_orc,
            'facturação_prevista_obra': Facturação_prevista_obra,
            'custos_prevmat_obra': Custos_prevmat_obra,
            'custos_prevserv_obra': Custos_prevserv_obra,
            'dias': dias,
            'tipoobra': Tipoobra,
            'directorobra': Directorobra,
            'area': Area,
            'vendedor_nome': Vendedor_nome,
            }

    df = pd.DataFrame(data, index=[0])

    df_dados = df #Df para fazer o grafico, precisa dos nomes e tal

    df['tipoobra'] = df['tipoobra'].astype('category')
    df['directorobra'] = df['directorobra'].astype('category')
    df['area'] = df['area'].astype('category')
    df['vendedor_nome'] = df['vendedor_nome'].astype('category')
   
    df = pd.get_dummies(df, columns=['tipoobra'])
    df = pd.get_dummies(df, columns=['directorobra'])
    df = pd.get_dummies(df, columns=['area'])
    df = pd.get_dummies(df, columns=['vendedor_nome'])

    df_final= pd.DataFrame(df, index=[0],columns=X.columns)
    df_final.fillna(0, inplace=True)

    z = []
    z.append(df_final)
    z.append(df_dados)
    return z

return_func = user_input_features()
df = return_func[0]
df_dados = return_func[1]


if st.button("Prever (Random Forest)"):
    #ML
    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    prediction = rf.predict(df)
    prediction_proba_yes = rf.predict_proba(df)[:, 1]
    prediction_proba_no = rf.predict_proba(df)[:, 0]

    prediction_proba_yes = prediction_proba_yes*100
    prediction_proba_yes= np.around(prediction_proba_yes,2)

    prediction_proba_no = prediction_proba_no*100
    prediction_proba_no= np.around(prediction_proba_no,2)

    if(prediction == -1):
        st.info('É provável que a obra não tenha desvios positivos.')

    if(prediction == 1):
        st.info('É provável que a obra tenha desvios positivos.')

    st.info('A previsão de não ter desvios positivos é de {}%'.format(prediction_proba_no))
    st.info('A previsão de ter desvios positivos é de {}%'.format(prediction_proba_yes))

    Directorobra_df = df_tratados.groupby("directorobra")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    Directorobra_df = Directorobra_df.mul(100).rename('Percent').reset_index()
    Directorobra_df = Directorobra_df.loc[(Directorobra_df["Desvios_custo_total_categorico"]==1)]

    Directorobra_df = Directorobra_df.rename(columns={"directorobra": "Dados"})

    Vendedor_df = df_tratados.groupby("vendedor_nome")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    Vendedor_df = Vendedor_df.mul(100).rename('Percent').reset_index()
    Vendedor_df = Vendedor_df.loc[(Vendedor_df["Desvios_custo_total_categorico"]==1)]

    Vendedor_df = Vendedor_df.rename(columns={"vendedor_nome": "Dados"})

    Tipoobra_df = df_tratados.groupby("tipoobra")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    Tipoobra_df = Tipoobra_df.mul(100).rename('Percent').reset_index()
    Tipoobra_df = Tipoobra_df.loc[(Tipoobra_df["Desvios_custo_total_categorico"]==1)]

    Tipoobra_df = Tipoobra_df.rename(columns={"tipoobra": "Dados"})

    area_df = df_tratados.groupby("area")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    area_df = area_df.mul(100).rename('Percent').reset_index()
    area_df = area_df.loc[(area_df["Desvios_custo_total_categorico"]==1)]

    area_df = area_df.rename(columns={"area": "Dados"})

    df_grafico = Directorobra_df.append(Vendedor_df, ignore_index=True)
    df_grafico = df_grafico.append(Vendedor_df, ignore_index=True)
    df_grafico = df_grafico.append(Tipoobra_df, ignore_index=True)
    df_grafico = df_grafico.append(area_df, ignore_index=True)

    df_dados["directorobra"] = df_dados["directorobra"].astype(str)
    df_dados["vendedor_nome"] = df_dados["vendedor_nome"].astype(str)
    df_dados["tipoobra"] = df_dados["tipoobra"].astype(str)
    df_dados["area"] = df_dados["area"].astype(str)

    diretor = df_dados["directorobra"][0]
    vendedor = df_dados["vendedor_nome"][0]
    tipoobra = df_dados["tipoobra"][0]
    area = df_dados["area"][0]

    df_grafico = df_grafico.loc[(df_grafico['Dados'] == diretor) |
               (df_grafico['Dados'] == vendedor) |
               (df_grafico['Dados'] == tipoobra) |
               (df_grafico['Dados'] == area)]

    #st.write(df_dados.astype('object'))
    #st.write(df_grafico.astype('object'))
    fig, ax = plt.subplots()
    plt.style.use("default")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    #fig = plt.figure(figsize=(15, 8))
    ax.set_xlim(0, 100)
    ax = sns.barplot(
        data=df_grafico,
        y="Dados",
        x="Percent",
        ci=None,)   

    ax.set(xlabel='Desvios Positivos(%)',ylabel="")
    for p in ax.patches:
        height = p.get_height() # height of each horizontal bar is the same
        width = p.get_width() # width (average number of passengers)
        # adding text to each bar
        ax.text(x = width+3, # x-coordinate position of data label, padded 3 to right of bar
        y = p.get_y()+(height/2), # # y-coordinate position of data label, padded to be in the middle of the bar
        s = '{:.0f}'.format(width), # data label, formatted to ignore decimals
        va = 'center') # sets vertical alignment (va) to center
    st.pyplot(fig)

if st.button("Prever (Linear Discriminant)"):

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    lda = LinearDiscriminantAnalysis()
    lda.fit(X_train,y_train)

    prediction = lda.predict(df)
    prediction_proba_yes = lda.predict_proba(df)[:, 1]
    prediction_proba_no = lda.predict_proba(df)[:, 0]

    prediction_proba_yes = prediction_proba_yes*100
    prediction_proba_yes= np.around(prediction_proba_yes,2)

    prediction_proba_no = prediction_proba_no*100
    prediction_proba_no= np.around(prediction_proba_no,2)

    if(prediction == -1):
        st.info('É provável que a obra não tenha desvios positivos.')

    if(prediction == 1):
        st.info('É provável que a obra tenha desvios positivos.')

    st.info('A previsão de não ter desvios positivos é de {}%'.format(prediction_proba_no))
    st.info('A previsão de ter desvios positivos é de {}%'.format(prediction_proba_yes))

    Directorobra_df = df_tratados.groupby("directorobra")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    Directorobra_df = Directorobra_df.mul(100).rename('Percent').reset_index()
    Directorobra_df = Directorobra_df.loc[(Directorobra_df["Desvios_custo_total_categorico"]==1)]

    Directorobra_df = Directorobra_df.rename(columns={"directorobra": "Dados"})

    Vendedor_df = df_tratados.groupby("vendedor_nome")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    Vendedor_df = Vendedor_df.mul(100).rename('Percent').reset_index()
    Vendedor_df = Vendedor_df.loc[(Vendedor_df["Desvios_custo_total_categorico"]==1)]

    Vendedor_df = Vendedor_df.rename(columns={"vendedor_nome": "Dados"})

    Tipoobra_df = df_tratados.groupby("tipoobra")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    Tipoobra_df = Tipoobra_df.mul(100).rename('Percent').reset_index()
    Tipoobra_df = Tipoobra_df.loc[(Tipoobra_df["Desvios_custo_total_categorico"]==1)]

    Tipoobra_df = Tipoobra_df.rename(columns={"tipoobra": "Dados"})

    area_df = df_tratados.groupby("area")["Desvios_custo_total_categorico"].value_counts(normalize=True)
    area_df = area_df.mul(100).rename('Percent').reset_index()
    area_df = area_df.loc[(area_df["Desvios_custo_total_categorico"]==1)]

    area_df = area_df.rename(columns={"area": "Dados"})

    df_grafico = Directorobra_df.append(Vendedor_df, ignore_index=True)
    df_grafico = df_grafico.append(Vendedor_df, ignore_index=True)
    df_grafico = df_grafico.append(Tipoobra_df, ignore_index=True)
    df_grafico = df_grafico.append(area_df, ignore_index=True)

    df_dados["directorobra"] = df_dados["directorobra"].astype(str)
    df_dados["vendedor_nome"] = df_dados["vendedor_nome"].astype(str)
    df_dados["tipoobra"] = df_dados["tipoobra"].astype(str)
    df_dados["area"] = df_dados["area"].astype(str)

    diretor = df_dados["directorobra"][0]
    vendedor = df_dados["vendedor_nome"][0]
    tipoobra = df_dados["tipoobra"][0]
    area = df_dados["area"][0]

    df_grafico = df_grafico.loc[(df_grafico['Dados'] == diretor) |
               (df_grafico['Dados'] == vendedor) |
               (df_grafico['Dados'] == tipoobra) |
               (df_grafico['Dados'] == area)]

    #st.write(df_dados.astype('object'))
    #st.write(df_grafico.astype('object'))
    fig, ax = plt.subplots()
    plt.style.use("default")
    plt.rcParams['figure.facecolor'] = 'white'
    plt.rcParams['axes.facecolor'] = 'white'
    #fig = plt.figure(figsize=(15, 8))
    ax.set_xlim(0, 100)
    ax = sns.barplot(
        data=df_grafico,
        y="Dados",
        x="Percent",
        ci=None,)   

    ax.set(xlabel='Desvios Positivos(%)',ylabel="")
    for p in ax.patches:
        height = p.get_height() # height of each horizontal bar is the same
        width = p.get_width() # width (average number of passengers)
        # adding text to each bar
        ax.text(x = width+3, # x-coordinate position of data label, padded 3 to right of bar
        y = p.get_y()+(height/2), # # y-coordinate position of data label, padded to be in the middle of the bar
        s = '{:.0f}'.format(width), # data label, formatted to ignore decimals
        va = 'center') # sets vertical alignment (va) to center
    st.pyplot(fig)


st.write("Previsões para várias obras")

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    excel_df = pd.read_csv(uploaded_file,
    sep = ";",
    decimal=',',
    encoding='latin-1',
    engine='python')
    

if st.button("Prever  (Random Forest)"):

    from sklearn.ensemble import RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)
    rf.fit(X_train, y_train)

    excel_df["tipoobra"] = excel_df['tipoobra'].astype('category')
    excel_df['directorobra'] = excel_df['directorobra'].astype('category')
    excel_df['area'] = excel_df['area'].astype('category')
    excel_df['vendedor_nome'] = excel_df['vendedor_nome'].astype('category')

    excel_df = pd.get_dummies(excel_df, columns=['tipoobra'])
    excel_df = pd.get_dummies(excel_df, columns=['directorobra'])
    excel_df = pd.get_dummies(excel_df, columns=['area'])
    excel_df = pd.get_dummies(excel_df, columns=['vendedor_nome'])
    excel_df= pd.DataFrame(excel_df,columns=X.columns)
    excel_df.fillna(0, inplace=True)
    prediction_proba_excel_yes = rf.predict_proba(excel_df)[:, 1]
    prediction_proba_excel_yes = prediction_proba_excel_yes*100
    teste=pd.DataFrame(prediction_proba_excel_yes)
   
    teste.columns = ['Previsão desvios positivos(%)']
    st.table(teste)
