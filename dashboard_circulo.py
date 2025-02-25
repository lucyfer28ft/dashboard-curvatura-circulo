import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, ks_2samp, kurtosis, skew
import plotly.express as px
from scipy.spatial import distance



#Calcular radio de curvatura a partir de tres puntos
def calcular_radio_tres_puntos(p1, p2, p3):
    """
    Calcula el radio de curvatura de un círculo que pasa por tres puntos dados.

    Parámetros:
    p1, p2, p3: Tuplas (x, y) que representan los puntos en el espacio.

    Retorna:
    R: Radio de curvatura del círculo circunscrito a los tres puntos.
    """
    # Calcular las distancias entre los puntos
    A = distance.euclidean(p1, p2)
    B = distance.euclidean(p2, p3)
    C = distance.euclidean(p3, p1)

    # Semiperímetro del triángulo formado por los tres puntos
    s = (A + B + C) / 2

    # Área del triángulo usando la fórmula de Herón
    area = np.sqrt(s * (s - A) * (s - B) * (s - C))

    if area == 0:
        return np.inf  # Si los puntos son colineales, el radio es infinito (curvatura cero).

    # Radio del círculo circunscrito al triángulo
    R = (A * B * C) / (4 * area)

    return R

# Función para calcular el centro del círculo que pasa por tres puntos
def calcular_centro_circulo(p1, p2, p3):
    x0, y0 = p1
    x1, y1 = p2
    x2, y2 = p3

    # Calcular el determinante
    D = 2 * (x0 * (y1 - y2) + x1 * (y2 - y0) + x2 * (y0 - y1))

    # Verificar si los puntos son colineales
    if D == 0:
        return None  # No hay círculo único

    # Calcular coordenadas del centro del círculo
    C_x = ((x0**2 + y0**2) * (y1 - y2) + (x1**2 + y1**2) * (y2 - y0) + (x2**2 + y2**2) * (y0 - y1)) / D
    C_y = ((x0**2 + y0**2) * (x2 - x1) + (x1**2 + y1**2) * (x0 - x2) + (x2**2 + y2**2) * (x1 - x0)) / D

    return C_x, C_y


# Streamlit App
st.title("📊 Análisis de la Curvatura de las Costillas")
st.markdown(
    "Este Dashboard permite calcular los radios de curvatura de las costillas en los puntos de interés así como analizar y visualizar la distribución de dichos radios de forma general o en función de las distintas medidas de placa. 🔍")

# Cargar archivo Excel
archivo = st.file_uploader("📂 Sube tu archivo Excel", type=["xls", "xlsx"])
if archivo:
    df = pd.read_excel(archivo, header=1)

    # Definir nombres de las columnas relevantes
    asimetria = "f (Assymetry Index)"
    placa = "a (elevator plate)"
    nombre_carpeta = "FILE NAME"
    eje_menor_1 = "INDICES(A)"
    eje_menor_2 = "INDICE(B)"
    torax_columna = "INDICE€"
    esternon_columna = "INDICE(D)"
    transv = "INDICE©"


    columnas_requeridas = [ nombre_carpeta, eje_menor_1, eje_menor_2,transv, esternon_columna, torax_columna, placa, "d(Potencial Lifting Distance)MIN", asimetria]

    if not all(col in df.columns for col in columnas_requeridas):
        st.error(f"🚨 El archivo debe contener las columnas: {columnas_requeridas}")
    else:


        # Filtrar registros válidos
        df = df.dropna(subset=columnas_requeridas)
        df = df[columnas_requeridas]

        st.write(
            "## Filtrado de Registros")

        st.write("")

        df[asimetria] = df[asimetria].abs()




        # Slider para ajustar el umbral de asimetría
        asimetria_umbral = st.slider("Selecciona el umbral máximo de asimetría:", min_value=0.000, max_value=0.350,
                                     value=0.120, step=0.001)
        st.write("")

        # Filtrar los datos según la asimetría seleccionada
        df = df[df[asimetria] <= asimetria_umbral]

        #Número de registros filtrados

        num_filas = df.shape[0] - 2
        st.write(f"Número de registros de pacientes: **{num_filas}**")





        # Slider para ajustar el umbral de asimetría
        sobrecorreccion = st.slider("Selecciona el porcentaje de sobrecorreción:", min_value=0.000, max_value=1.0,
                                     value=0.1, step=0.01)
        st.write("")


        df["Indice Corrección (cm)"] = df[torax_columna] - df[esternon_columna]
        df["Eje Menor"] = (df[eje_menor_1] + df[eje_menor_2])/2
        df["Sobrecorrección (cm)"] = df["Eje Menor"]* sobrecorreccion
        df["Eje Menor corregido"] = df["Eje Menor"] + df["Sobrecorrección (cm)"]





        st.write("### Registros seleccionados que cumplen con los valores requeridos para el análisis y no exceden la asimetría definida.")
        st.write("")
        st.write( " Añadidas las columnas en las que se puede consultar los valores de los radios de curvatura y otros datos de interés ")


        st.write("")




        #Calcular radio para todos los pacientes

        # Valores dados (en cm)
        c = df[transv]  # Diámetro transversal
        d = df["Eje Menor"]  # Diámetro anteroposterior
        d_prime = df["Eje Menor corregido"]  # Aplicar sobrecorrección

        # Cálculo de radios de curvatura

        # Definir coordenadas x
        df["x0"] = - df[placa] / 20
        df["x1"] = 0
        df["x2"] = df[placa] / 20

        #Definir coordenadas y

        df["y0"] = df["Eje Menor"]
        df["y1"] = df["Eje Menor corregido"]
        df["y2"] = df["Eje Menor"]


        df["Radio de curvatura"] = df.apply(
            lambda row: calcular_radio_tres_puntos(
                (row["x0"], row["y0"]),
                (row["x1"], row["y1"]),
                (row["x2"], row["y2"])
            ), axis=1
        )



        #Mostrar el dataframe con las nuevas columnas con los datos de interés

        df = df.drop(["x0", "x1", "x2", "y0", "y1", "y2" ], axis=1)
        st.dataframe(df)



        #ESTADÍSTICAS CLAVE


        # 📊 Estadísticas clave

        media = df["Radio de curvatura"].mean()
        desviacion = df["Radio de curvatura"].std()
        curvatura_min = df["Radio de curvatura"].min()
        curvatura_max = df["Radio de curvatura"].max()
        asimetria = skew(df["Radio de curvatura"].dropna())
        curtosis_val = kurtosis(df["Radio de curvatura"].dropna())

        # 📌 Cálculo de percentiles
        df["Radio de curvatura"] = df["Radio de curvatura"].astype(str).str.strip()
        df["Radio de curvatura"] = pd.to_numeric(df["Radio de curvatura"], errors="coerce")

        if df["Radio de curvatura"].count() > 0:
            percentiles = np.percentile(df["Radio de curvatura"].dropna(), [5, 50, 95])
        else:
            percentiles = [np.nan, np.nan, np.nan]


        # 📌 Pruebas de normalidad
        stat_shapiro, p_shapiro = shapiro(df["Radio de curvatura"].dropna())
        stat_ks, p_ks = ks_2samp(df["Radio de curvatura"].dropna(), np.random.normal(media, desviacion, len(df)))

        # 📌 Superposición del histograma real con la curva normal
        fig_hist = px.histogram(df, x="Radio de curvatura", nbins=20, title="Superposición del histograma del radio medio con la curva normal", opacity=0.6,
                                marginal="box")
        x = np.linspace(media - 3 * desviacion, media + 3 * desviacion, 100)
        y = norm.pdf(x, media, desviacion) * len(df) * (
                curvatura_max - curvatura_min) / 20
        fig_hist.add_scatter(x=x, y=y, mode='lines', name="Curva Normal")

        # Mostrar estadísticas
        st.markdown("### 📌 Estadísticas Clave")
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 Media de Radio de Curvatura", f"{media:.2f} cm")
        col2.metric("📉 Desviación Estándar", f"{desviacion:.2f} cm")
        col3.metric("📈 Asimetría", f"{asimetria:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("🔼 Máximo Radio de Curvatura", f"{curvatura_max:.2f} cm")
        col5.metric("🔽 Mínimo Radio de Curvatura", f"{curvatura_min:.2f} cm")
        col6.metric("🔄 Curtosis", f"{curtosis_val:.2f}")
        st.write("·**Desviación Estándar**: mide cuánto varían los datos respecto a la media. Es decir, indica si los valores están muy dispersos o concentrados cerca del promedio. (**(cercana a 0)** → Datos muy agrupados alrededor de la media, poca variabilidad en los radios de curvatura., , **Alta** →  Datos muy dispersos respecto a la media, mayor variabilidad en los radios de curvatura")
        st.write("·**Asimetría**: mide cuán simétrica es la distribución de los datos respecto a la media. ( ·**0** → Distribución simétrica, como la normal,    ·**Negativo (< 0)** → Sesgo a la izquierda (cola más larga a la izquierda, la distribución tiene más valores menores a la media),   ·**Positivo (> 0)** → Sesgo a la derecha (cola más larga a la derecha, la distribución tiene más valores mayores a la media)).")
        st.write("·**Curtosis**: mide si los datos tienen colas más o menos pesadas en comparación con una distribución normal. (**0 o cercano a 0** → Mesocúrtica (Distribución normal, colas estándar)., **Negativo (< 0)** → Platicúrtica (Colas ligeras, distribución más plana, datos más dispersos, sin valores extremos), **Positivo (> 0)** → Leptocúrtica (Colas pesadas, picos más pronunciados, es decir, muchos valores extremos, lo que sugiere casos atípicos (outliers))). ")


        # Interpretar desviación
        def interpretar_desviacion(media, desviacion):
            rango_68 = (media - desviacion, media + desviacion)
            rango_95 = (media - 2 * desviacion, media + 2 * desviacion)
            rango_997 = (media - 3 * desviacion, media + 3 * desviacion)

            st.markdown("### 📌 Interpretación de la Desviación Estándar")
            st.write(
                f"🔹 **El 68% de los pacientes** tienen un radio medio entre **{rango_68[0]:.2f} cm** y **{rango_68[1]:.2f} cm**.")
            st.write(
                f"🔹 **El 95% de los pacientes** tienen un radio medio entre **{rango_95[0]:.2f} cm** y **{rango_95[1]:.2f} cm**.")
            st.write(
                f"🔹 **El 99.7% de los pacientes** tienen un radio medio entre **{rango_997[0]:.2f} cm** y **{rango_997[1]:.2f} cm**.")

            # Evaluación del impacto en el diseño del implante
            if desviacion < 3:
                st.success(
                    "✅ La desviación estándar es baja, lo que indica que los radios de curvatura son muy similares entre los pacientes. Esto permite diseñar un implante con medidas estándar.")
            elif 3 <= desviacion <= 7:
                st.warning(
                    "⚠️ La desviación estándar es moderada, lo que sugiere cierta variabilidad en los radios de curvatura. Puede ser útil considerar algunas opciones de curvatura.")
            else:
                st.error(
                    "🚨 La desviación estándar es alta, lo que indica una gran variabilidad en los radios de curvatura. Podría ser necesario diseñar implantes personalizados para diferentes grupos de pacientes.")


        interpretar_desviacion(media, desviacion)

        # Mostrar percentiles
        st.markdown("### 📌 Percentiles del Radio Medio de Curvatura")
        col7, col8, col9 = st.columns(3)
        col7.metric("🔹 Percentil 5%", f"{percentiles[0]:.2f} cm")
        col8.metric("🔸 Mediana (50%)", f"{percentiles[1]:.2f} cm")
        col9.metric("🔹 Percentil 95%", f"{percentiles[2]:.2f} cm")
        st.write(
            "·Los **percentiles** permiten evaluar la distribución de los radios de curvatura de las costillas en el tramo de interés en la población analizada. El percentil 5% indica el valor por debajo del cual se encuentran el 5% de los datos, la mediana (50%) representa el valor central de la distribución, y el percentil 95% muestra el valor por debajo del cual se encuentra el 95% de la población.")


        # Gráfico interactivo del Radio de Curvatura
        fig = px.histogram(df, x="Radio de curvatura", nbins=20, marginal="rug",
                           title="Histograma de la Distribución del Radio Medio de Curvatura", color_discrete_sequence=["red"])
        st.plotly_chart(fig)

        # Resultados de las pruebas de normalidad
        st.markdown("### 🧪 Pruebas de Normalidad")
        st.write(f"**Shapiro-Wilk test**: Estadístico = {stat_shapiro:.4f}, p-valor = {p_shapiro:.4f}")
        st.write(f"**Kolmogorov-Smirnov test**: Estadístico = {stat_ks:.4f}, p-valor = {p_ks:.4f}")

        if p_shapiro < 0.05:
            st.warning("🚨 Los datos NO siguen una distribución normal según el test de Shapiro-Wilk (p < 0.05)")
        else:
            st.success("✅ Los datos parecen seguir una distribución normal según el test de Shapiro-Wilk (p >= 0.05)")

        if p_ks < 0.05:
            st.warning("🚨 Los datos NO siguen una distribución normal según el test de Kolmogorov-Smirnov (p < 0.05)")
        else:
            st.success(
                "✅ Los datos parecen seguir una distribución normal según el test de Kolmogorov-Smirnov (p >= 0.05)")


        #Mostrar la superposición del histograma real con la curva normal

        st.plotly_chart(fig_hist)

        # Análisis estadístico según los tamaños de la placa

        st.title("Análisis estadístico según los tamaños de la placa")



        # Calcular la media de los radios de curvatura por cada tamaño de placa
        media_por_placa = df.groupby(placa)["Radio de curvatura"].mean()

        st.subheader("Media de Radio de Curvatura por Medida de Placa")
        st.dataframe(media_por_placa)  # O usa st.table(media_por_placa) para un formato más compacto

        # Crear figura
        fig, ax = plt.subplots(figsize=(8, 6))

        # Graficar distribución de los radios de curvatura
        ax.scatter(df[placa], df["Radio de curvatura"], alpha=0.5,
                   label="Radios individuales")

        # Graficar medias de radio por medida de placa
        ax.plot(media_por_placa.index, media_por_placa.values, marker='o', linestyle='-', color='red',
                label="Media por placa")

        # Configuración del gráfico
        ax.set_xlabel("Medida de la Placa (cm)")
        ax.set_ylabel("Radio de Curvatura (cm)")
        ax.set_title("Distribución de Radios de Curvatura según Medida de la Placa")
        ax.legend()
        ax.grid(True)

        # Mostrar gráfico en Streamlit
        st.pyplot(fig)



        # Selección de paciente
        st.title("Visualización de la Elipse corregida")
        selected_paciente = st.selectbox("Selecciona un paciente:", df[nombre_carpeta])



        # Obtener valores del paciente seleccionado
        dfp = df[df[nombre_carpeta] == selected_paciente].iloc[0]



        #Mostrar datos de interés del paciente seleccionado
        st.markdown("### 📌 Datos de interés del paciente")
        col4,col5 = st.columns(2)
        col4.metric("**Radio de curvatura**", f"{dfp["Radio de curvatura"]} cm")
        col5.metric("Longitud Placa", f"{dfp[placa] / 10} cm")


        col2, col3 = st.columns(2)
        col2.metric("Eje Menor", f"{dfp["Eje Menor"]} cm")
        col3.metric("Sobrecorrección", f"{dfp["Sobrecorrección (cm)"]} cm")





        #Graficar radio para cada paciente


        if st.button("Calcular"):

            # Valores dados (en cm)
            c = dfp[transv]  # Diámetro transversal
            d = dfp["Eje Menor corregido"]  # Diámetro anteroposterior con sobrecorrección

            # Convertir el radio a flotante
            r_curvatura = float(dfp["Radio de curvatura"])

            # Definir puntos de referencia
            p1 = x0, y0 = -dfp[placa] / 20, dfp["Eje Menor"]
            p2 = x1, y1 = 0, dfp["Eje Menor corregido"]
            p3 = x2, y2 = dfp[placa] / 20, dfp["Eje Menor"]

            # Calcular centro del círculo
            centro_circulo = calcular_centro_circulo(p1, p2, p3)
            C_x = centro_circulo[0]
            C_y = centro_circulo[1]



            # Definir función para dibujar el círculo con centro en (C_x, C_y)
            def f(x, r, C_y):
                return C_y + np.sqrt(np.maximum(0, r ** 2 - (x - C_x) ** 2))


            x = np.linspace(x0, x2, 300)

            fig, ax = plt.subplots(figsize=(8, 6))
            ax.plot(x, f(x, r_curvatura, C_y), label="Círculo correspondiente al radio de curvatura calculado",
                    linestyle="dashed")

            # Puntos de referencia
            ax.scatter([x0, x2], [y0, y2], color="blue", label="Puntos extremos de la placa")
            ax.scatter([x1], [y1], color="red", label="Esternón")

            # Configuración del gráfico
            ax.set_xlabel("Ancho (cm)")
            ax.set_ylabel("Altura (cm)")
            ax.axhline(0, color='black', linewidth=0.5)
            ax.axvline(0, color='black', linewidth=0.5)
            ax.legend()
            ax.set_title("Círculo correspondiente al radio de curvatura calculado")
            ax.grid(True)

            st.pyplot(fig)








