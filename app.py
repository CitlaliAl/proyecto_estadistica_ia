import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm

st.set_page_config(page_title="App Estadística con IA", layout="centered")

st.title("📊 Análisis Estadístico con IA")

menu = st.sidebar.selectbox("Menú", [
    "Carga de datos",
    "Visualización",
    "Prueba de hipótesis",
    "Asistente IA"
])

if "data" not in st.session_state:
    st.session_state.data = None

# -------------------------
# CARGA DE DATOS
# -------------------------
if menu == "Carga de datos":

    opcion = st.radio("Selecciona:", ["CSV", "Datos sintéticos"])

    if opcion == "CSV":
        archivo = st.file_uploader("Sube CSV")
        if archivo:
            df = pd.read_csv(archivo)
            st.session_state.data = df
            st.success("Datos cargados")
            st.write(df.head())

    else:
        if st.button("Generar datos"):
            datos = np.random.normal(50, 10, 100)
            df = pd.DataFrame(datos, columns=["valores"])
            st.session_state.data = df
            st.success("Datos generados")
            st.write(df.head())

# -------------------------
# VISUALIZACIÓN
# -------------------------
elif menu == "Visualización":

    st.header("Visualización")

    if st.session_state.data is not None:
        df = st.session_state.data
        col = st.selectbox("Variable", df.columns)

        datos = pd.to_numeric(df[col], errors='coerce')
        datos = datos.dropna()
        if datos.empty:
            st.error("La columna seleccionada no contiene datos numéricos válidos")
            st.stop()

        # Histograma + KDE
        fig, ax = plt.subplots()
        sns.histplot(datos, kde=True, ax=ax)
        st.pyplot(fig)

        # Boxplot
        fig2, ax2 = plt.subplots()
        sns.boxplot(x=datos, ax=ax2)
        st.pyplot(fig2)

        # Interpretación automática
        st.subheader("Interpretación")

        if datos.mean() > datos.median():
            st.write("Sesgo a la derecha")
        elif datos.mean() < datos.median():
            st.write("Sesgo a la izquierda")
        else:
            st.write("Distribución simétrica")

        st.write("Outliers visibles en el boxplot")

    else:
        st.warning("Carga datos primero")

# -------------------------
# PRUEBA DE HIPÓTESIS
# -------------------------
elif menu == "Prueba de hipótesis":

    st.header("Prueba Z")

    m = st.number_input("Media muestral", 50.0)
    mu = st.number_input("Media H0", 50.0)
    sigma = st.number_input("Sigma", 10.0)
    n = st.number_input("n (>=30)", 30)
    alpha = st.selectbox("Nivel de significancia", [0.01, 0.05, 0.1])
    tipo = st.selectbox("Tipo de prueba", ["Bilateral", "Izquierda", "Derecha"])

    if st.button("Calcular"):

        z = (m - mu) / (sigma / np.sqrt(n))
        st.write(f"Z: {z:.4f}")

        # p-value
        if tipo == "Bilateral":
            p = 2 * (1 - norm.cdf(abs(z)))
        elif tipo == "Derecha":
            p = 1 - norm.cdf(z)
        else:
            p = norm.cdf(z)

        st.write(f"p-value: {p:.4f}")

        # decisión
        if p < alpha:
            decision = "Se rechaza H0"
            st.error(decision)
        else:
            decision = "No se rechaza H0"
            st.success(decision)

        # gráfica
        x = np.linspace(-4, 4, 100)
        y = norm.pdf(x)

        fig, ax = plt.subplots()
        ax.plot(x, y)
        ax.axvline(z, linestyle="--")
        st.pyplot(fig)

        # guardar resultados
        st.session_state.resultados = {
            "m": m,
            "mu": mu,
            "sigma": sigma,
            "n": n,
            "alpha": alpha,
            "tipo": tipo,
            "z": z,
            "p": p,
            "decision": decision
        }

# -------------------------
# ASISTENTE IA
# -------------------------
elif menu == "Asistente IA":

    st.header("Asistente con IA")

    if "resultados" in st.session_state:

        if st.button("Consultar IA"):

            r = st.session_state.resultados

            # Simulación tipo IA
            if r["p"] < r["alpha"]:
                respuesta = f"""
La IA concluye que se RECHAZA la hipótesis nula.

Esto se debe a que el p-value ({r["p"]:.4f}) es menor que el nivel de significancia ({r["alpha"]}).

Por lo tanto, existe evidencia estadística suficiente para afirmar que la media poblacional es diferente a la hipotética.
"""
            else:
                respuesta = f"""
La IA concluye que NO se rechaza la hipótesis nula.

Esto se debe a que el p-value ({r["p"]:.4f}) es mayor que el nivel de significancia ({r["alpha"]}).

No hay suficiente evidencia estadística para afirmar que la media es diferente.
"""

            st.subheader("Respuesta de la IA")
            st.write(respuesta)

            st.subheader("Comparación")
            st.write("Decisión del sistema:", r["decision"])

    else:
        st.warning("Primero realiza una prueba de hipótesis")