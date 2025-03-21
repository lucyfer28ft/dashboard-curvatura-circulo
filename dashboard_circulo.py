import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import norm, shapiro, ks_2samp, kurtosis, skew
import plotly.express as px
from scipy.spatial import distance
import plotly.graph_objects as go

st.set_page_config(layout="wide")


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

        st.markdown("<br>", unsafe_allow_html=True)


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

        st.markdown("<br>", unsafe_allow_html=True)



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

        st.markdown("<br>", unsafe_allow_html=True)


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

        st.markdown("<br>", unsafe_allow_html=True)

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

        st.markdown("<br>", unsafe_allow_html=True)

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

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # Análisis estadístico según los tamaños de la placa

        st.title("Análisis estadístico según los tamaños de la placa")

        # Calcular estadísticas clave: media, mínimo y máximo
        media_por_placa = df.groupby(placa)["Radio de curvatura"].agg(["mean", "min", "max"]).reset_index()

        # Renombrar columnas
        media_por_placa.rename(columns={
            placa: "Placa Elevadora",
            "mean": "Media Radio de Curvatura",
            "min": "Mínimo Radio de Curvatura",
            "max": "Máximo Radio de Curvatura"
        }, inplace=True)

        # Título en Streamlit
        st.subheader("Resumen de Radio de Curvatura por Placa Elevadora")

        # Mostrar la tabla en Streamlit
        st.dataframe(media_por_placa)

        # Crear gráfico interactivo con Plotly
        fig = px.scatter(df, x=placa, y="Radio de curvatura", opacity=0.6,
                         labels={"Medida de la Placa": "Placa Elevadora",
                                 "Radio de curvatura": "Radio de Curvatura (cm)"},
                         title="Distribución de Radios de Curvatura según Placa Elevadora")

        # Agregar la media al gráfico
        fig.add_scatter(x=media_por_placa["Placa Elevadora"],
                        y=media_por_placa["Media Radio de Curvatura"],
                        mode="lines+markers", line=dict(color="red"), name="Media por placa")

        # Agregar el mínimo al gráfico
        fig.add_scatter(x=media_por_placa["Placa Elevadora"],
                        y=media_por_placa["Mínimo Radio de Curvatura"],
                        mode="lines+markers", line=dict(color="blue"), name="Mínimo por placa")

        # Agregar el máximo al gráfico
        fig.add_scatter(x=media_por_placa["Placa Elevadora"],
                        y=media_por_placa["Máximo Radio de Curvatura"],
                        mode="lines+markers", line=dict(color="green"), name="Máximo por placa")

        # Mostrar gráfico en Streamlit
        st.plotly_chart(fig)


        #Elegir dos radios estandar y decidir para que medidas de placa unos y para que medidas de placa otros

        # Ordenar por medida de placa
        media_por_placa = media_por_placa.sort_values(by="Placa Elevadora")

        # Cálculo del salto de radio promedio entre medidas consecutivas
        media_por_placa["Delta Media"] = media_por_placa["Media Radio de Curvatura"].diff()

        # Mostrar tabla para explorar
        st.subheader("📐 Clasificación de Placas Elevadoras según Radio de Curvatura Promedio")
        st.write("""📌 La decisión se basa en cómo cambia el radio de curvatura promedio conforme aumenta la medida de la placa elevadora.
Al ordenar las placas por tamaño y analizar la media de los radios de curvatura, buscamos el punto donde se produce el mayor salto entre dos placas consecutivas.
Ese salto indica un cambio estructural significativo en la anatomía o en la forma en que se comportan las costillas, lo que justifica usar dos radios estándar diferentes:

Uno para placas más pequeñas (donde el tórax es más cerrado o curvo).
Otro para placas más grandes (donde el tórax tiende a ser más ancho y plano).""")

        st.write("### 📊 Análisis de Transiciones")
        st.dataframe(media_por_placa)

        st.write("""📉 La "media delta" es simplemente la diferencia entre las medias de radio de curvatura de una placa y la siguiente en tamaño.
        Sirve para detectar cuándo hay un cambio brusco entre dos medidas de placa.
        Ese salto nos ayuda a identificar el punto en que deberíamos cambiar de un radio estándar pequeño a uno grande.""")

        # Buscar el mayor salto de radio medio entre placas consecutivas
        mayor_salto = media_por_placa["Delta Media"].abs().idxmax()
        corte = media_por_placa.loc[mayor_salto, "Placa Elevadora"]

        # Dividir entre placas chicas y grandes según ese punto de corte
        placas_chicas = media_por_placa[media_por_placa["Placa Elevadora"] <= corte]
        placas_grandes = media_por_placa[media_por_placa["Placa Elevadora"] > corte]

        # Calcular radios estándar
        radio_pequeno_estandar = round(placas_chicas["Media Radio de Curvatura"].mean(), 2)
        radio_grande_estandar = round(placas_grandes["Media Radio de Curvatura"].mean(), 2)

        # Mostrar decisión
        st.markdown(f"""
        ### ✅ Recomendación de Radios Estándar

        - 🔸 Para placas elevadoras de **{placas_chicas['Placa Elevadora'].min()} mm a {corte} mm** → usar **radio estándar pequeño ≈ {radio_pequeno_estandar} cm**  
        - 🔹 Para placas elevadoras de **mayores a {corte} mm** → usar **radio estándar grande ≈ {radio_grande_estandar} cm**
        """)

        st.write("""La recomendación del radio estándar se basa en calcular el promedio de los radios de curvatura reales dentro de cada grupo (placas pequeñas y grandes).
        Así, el radio estándar pequeño representa bien a los pacientes con placas menores, y el radio estándar grande, a los que usan placas mayores.
        Esto permite usar valores representativos y funcionales, adaptados a la anatomía real de cada grupo.""")



        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)



        #Visualización del radio de curvatura de un paciente seleccionado

        # Título general
        st.title("Visualización del Radio de Curvatura en un Paciente")

        # Selección de paciente
        selected_paciente = st.selectbox("Selecciona un paciente:", df[nombre_carpeta])

        # Obtener valores del paciente seleccionado
        dfp = df[df[nombre_carpeta] == selected_paciente].iloc[0]

        # Mostrar datos clave
        st.markdown("### 📌 Datos de interés del paciente")
        col4, col5 = st.columns(2)
        col4.metric("**Radio de curvatura**", f"{dfp['Radio de curvatura']} cm")
        col5.metric("Longitud Placa", f"{dfp[placa] / 10} cm")

        col2, col3 = st.columns(2)
        col2.metric("Eje Menor", f"{dfp['Eje Menor']} cm")
        col3.metric("Sobrecorrección", f"{dfp['Sobrecorrección (cm)']} cm")

        # Botón para calcular
        if st.button("Calcular"):
            # Parámetros
            c = dfp[transv]
            d = dfp["Eje Menor corregido"]
            r_curvatura = float(dfp["Radio de curvatura"])

            # Puntos extremos
            x0, y0 = -dfp[placa] / 20, dfp["Eje Menor"]
            x1, y1 = 0, dfp["Eje Menor corregido"]
            x2, y2 = dfp[placa] / 20, dfp["Eje Menor"]


            # Calcular centro del círculo
            def calcular_centro_circulo(p1, p2, p3):
                A = np.array([
                    [2 * (p2[0] - p1[0]), 2 * (p2[1] - p1[1])],
                    [2 * (p3[0] - p2[0]), 2 * (p3[1] - p2[1])]
                ])
                b = np.array([
                    p2[0] ** 2 - p1[0] ** 2 + p2[1] ** 2 - p1[1] ** 2,
                    p3[0] ** 2 - p2[0] ** 2 + p3[1] ** 2 - p2[1] ** 2
                ])
                centro = np.linalg.solve(A, b)
                return centro


            centro_circulo = calcular_centro_circulo((x0, y0), (x1, y1), (x2, y2))
            C_x, C_y = centro_circulo


            # Función del círculo (parte superior)
            def f(x, r, C_y):
                return C_y + np.sqrt(np.maximum(0, r ** 2 - (x - C_x) ** 2))


            x_vals = np.linspace(x0, x2, 300)
            y_vals = f(x_vals, r_curvatura, C_y)

            # Crear gráfico interactivo
            fig = go.Figure()

            # Círculo (línea discontinua)
            fig.add_trace(go.Scatter(
                x=x_vals, y=y_vals,
                mode='lines',
                line=dict(dash='dash', color='black'),
                name="Círculo del radio de curvatura"
            ))

            # Puntos extremos (placa)
            fig.add_trace(go.Scatter(
                x=[x0, x2], y=[y0, y2],
                mode='markers',
                marker=dict(color='blue', size=10),
                name="Extremos de la placa"
            ))

            # Esternón
            fig.add_trace(go.Scatter(
                x=[x1], y=[y1],
                mode='markers',
                marker=dict(color='red', size=12),
                name="Esternón"
            ))

            # Ejes guía (rejilla más marcada) y a escala
            fig.update_layout(
                title="Círculo correspondiente al radio de curvatura calculado",
                xaxis=dict(title="Ancho (cm)", showgrid=True, dtick=1, zeroline=True),
                yaxis=dict(title="Altura (cm)", showgrid=True, dtick=1, zeroline=True, scaleanchor="x"),
                height=400,
                plot_bgcolor='rgba(0,0,0,0)'
            )

            st.plotly_chart(fig, use_container_width=True)








