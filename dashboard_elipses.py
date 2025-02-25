import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from scipy.stats import norm, shapiro, ks_2samp, kurtosis, skew
import plotly.express as px




# Streamlit App
st.title("📊 Análisis Curvatura de las Costillas")

# Cargar archivo Excel
archivo = st.file_uploader("📂 Sube tu archivo Excel", type=["xls", "xlsx"])
if archivo:
    df = pd.read_excel(archivo, header=1)

    # Definir nombres de las columnas relevantes
    eje_mayor_1 = "INDICES(A)"
    eje_mayor_2 = "INDICE(B)"
    suma_ejes_menores = "INDICE©"
    asimetria = "f (Assymetry Index)"
    placa = "a (elevator plate)"
    nombre_carpeta = "FILE NAME"
    columnas_requeridas = [eje_mayor_1, eje_mayor_2, suma_ejes_menores, asimetria, placa, nombre_carpeta]

    if not all(col in df.columns for col in columnas_requeridas):
        st.error(f"🚨 El archivo debe contener las columnas: {columnas_requeridas}")
    else:


        # Filtrar registros válidos
        df = df.dropna(subset=columnas_requeridas)

        st.write(
            "## Filtrado de Registros")

        st.write("")

        df[asimetria] = df[asimetria].abs()




        # Slider para ajustar el umbral de asimetría
        asimetria_umbral = st.slider("Selecciona el umbral máximo de asimetría:", min_value=0.000, max_value=0.080,
                                     value=0.020, step=0.001)
        st.write("")

        # Filtrar los datos según la asimetría seleccionada
        df = df[df[asimetria] <= asimetria_umbral]

        #Número de registros filtrados

        num_filas = df.shape[0] - 2
        st.write(f"Hay **{num_filas}** registros")




        # Función que calcula la media del radio de curvatura para cada fila del DataFrame
        def calcular_media_radio_curvatura(row):
            x0 = row[placa] / 2
            x1 = x0 + 20

            # Generar 100 puntos entre x0 y x1
            x_vals = np.linspace(x0, x1, 100)
            y_vals = elipse_y(x_vals, row["Eje Menor"], row["Eje Mayor"])

            # Calcular radios de curvatura
            radios = radio_curvatura(x_vals, y_vals, row["Eje Menor"], row["Eje Mayor"])

            # Devolver la media de los radios
            return np.mean(radios)

        # Calcular el eje mayor individual a través de la media entre ambos ejes mayores
        df["Eje Mayor"] = (df[eje_mayor_1] + df[eje_mayor_2])/2*10

        # Calcular el eje menor individual
        df["Eje Menor"] = df["Eje Mayor"] * np.sqrt(3) / 3



        def elipse_y(x, a, b):
            return b * np.sqrt(1 - ((x - a) ** 2 / a ** 2))


        # Calcular el radio de curvatura en esos puntos
        def radio_curvatura(x, y, a, b):
            return ((b ** 2 * (x - a) ** 2) + (a ** 2 * y ** 2)) ** (3 / 2) / (a ** 2 * b ** 2)



        df["Radio medio de curvatura"] = (df.apply(calcular_media_radio_curvatura, axis = 1))


        st.write("### Registros seleccionados que cumplen con los valores requeridos para el análisis y no exceden la asimetría definida.")
        st.write("")
        st.write( " Añadidas las columnas en las que se puede consultar el valor de su eje mayor, eje menor, y radio medio de curvatura en el tramo de interés. ")


        st.write("")

        st.dataframe(df)



        #ESTADÍSTICAS CLAVE


        # 📊 Estadísticas clave
        media = df["Radio medio de curvatura"].mean()
        desviacion = df["Radio medio de curvatura"].std()
        curvatura_min = df["Radio medio de curvatura"].min()
        curvatura_max = df["Radio medio de curvatura"].max()
        asimetria = skew(df["Radio medio de curvatura"].dropna())
        curtosis_val = kurtosis(df["Radio medio de curvatura"].dropna())

        # 📌 Cálculo de percentiles
        df["Radio medio de curvatura"] = df["Radio medio de curvatura"].astype(str).str.strip()
        df["Radio medio de curvatura"] = pd.to_numeric(df["Radio medio de curvatura"], errors="coerce")

        if df["Radio medio de curvatura"].count() > 0:
            percentiles = np.percentile(df["Radio medio de curvatura"].dropna(), [5, 50, 95])
        else:
            percentiles = [np.nan, np.nan, np.nan]


        # 📌 Pruebas de normalidad
        stat_shapiro, p_shapiro = shapiro(df["Radio medio de curvatura"].dropna())
        stat_ks, p_ks = ks_2samp(df["Radio medio de curvatura"].dropna(), np.random.normal(media, desviacion, len(df)))

        # 📌 Superposición del histograma real con la curva normal
        fig_hist = px.histogram(df, x="Radio medio de curvatura", nbins=20, title="Superposición del histograma del radio medio con la curva normal", opacity=0.6,
                                marginal="box")
        x = np.linspace(media - 3 * desviacion, media + 3 * desviacion, 100)
        y = norm.pdf(x, media, desviacion) * len(df) * (
                curvatura_max - curvatura_min) / 20
        fig_hist.add_scatter(x=x, y=y, mode='lines', name="Curva Normal")

        # Mostrar estadísticas
        st.markdown("### 📌 Estadísticas Clave")
        col1, col2, col3 = st.columns(3)
        col1.metric("📏 Media de Radio de Curvatura", f"{media:.2f} mm")
        col2.metric("📉 Desviación Estándar", f"{desviacion:.2f} mm")
        col3.metric("📈 Asimetría", f"{asimetria:.2f}")

        col4, col5, col6 = st.columns(3)
        col4.metric("🔼 Máximo Radio de Curvatura", f"{curvatura_max:.2f} mm")
        col5.metric("🔽 Mínimo Radio de Curvatura", f"{curvatura_min:.2f} mm")
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
                f"🔹 **El 68% de los pacientes** tienen un radio medio entre **{rango_68[0]:.2f} mm** y **{rango_68[1]:.2f} mm**.")
            st.write(
                f"🔹 **El 95% de los pacientes** tienen un radio medio entre **{rango_95[0]:.2f} mm** y **{rango_95[1]:.2f} mm**.")
            st.write(
                f"🔹 **El 99.7% de los pacientes** tienen un radio medio entre **{rango_997[0]:.2f} mm** y **{rango_997[1]:.2f} mm**.")

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
        col7.metric("🔹 Percentil 5%", f"{percentiles[0]:.2f} mm")
        col8.metric("🔸 Mediana (50%)", f"{percentiles[1]:.2f} mm")
        col9.metric("🔹 Percentil 95%", f"{percentiles[2]:.2f} mm")
        st.write(
            "·Los **percentiles** permiten evaluar la distribución de los radios de curvatura de las costillas en el tramo de interés en la población analizada. El percentil 5% indica el valor por debajo del cual se encuentran el 5% de los datos, la mediana (50%) representa el valor central de la distribución, y el percentil 95% muestra el valor por debajo del cual se encuentra el 95% de la población.")


        # Gráfico interactivo del Radio de Curvatura
        fig = px.histogram(df, x="Radio medio de curvatura", nbins=20, marginal="rug",
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




        # Gráfico de Radio de Curvatura según el tamaño de la placa
        fig = px.scatter(df, x=placa, y="Radio medio de curvatura", title="Curvatura de las Costillas según Tamaño de Placa", labels={placa: "Tamaño de la Placa (dm)", "Radio_Curvatura_Extremos": "Radio de Curvatura en los Extremos (cm)"})
        st.plotly_chart(fig)








        #VISUALIZAR RADIO INDIVIDUAL DE PACIENTE

        # Selección de paciente
        st.title("Visualización del Radio de Curvatura en la Elipse")
        selected_paciente = st.selectbox("Selecciona un paciente:", df[nombre_carpeta])



        # Obtener valores del paciente seleccionado
        paciente_data = df[df[nombre_carpeta] == selected_paciente].iloc[0]



        # Calcular el eje mayor individual a través de la media entre ambos ejes mayores
        b = (paciente_data[eje_mayor_1] + paciente_data[eje_mayor_2])/2 * 10

        # Calcular el eje menor individual
        a = b * np.sqrt(3) / 3


        # Valor inicial de x (extremo de la placa) desde el cual queremos empezar a calcular el radio de curvatura
        x0 = paciente_data[placa] / 2
        x1 = x0 + 20


        #Mostrar datos de interés del paciente seleccionado
        st.markdown("### 📌 Datos de interés del paciente")
        col1, col2, col3 = st.columns(3)
        col1.metric("Eje Mayor", f"{b} mm")
        col2.metric("Eje Menor", f"{a} mm")
        col3.metric("Longitud Placa", f"{x0*2} mm")




        def elipse_y(x, a, b):
            return b * np.sqrt(1 - ((x - a) ** 2 / a ** 2))


        # Generar los puntos de la proyección en la elipse

        x_vals = np.linspace(x0, x1, 100)  # 100 puntos entre x0 y x0+20
        y_vals = elipse_y(x_vals, a, b)


        # Calcular el radio de curvatura en esos puntos
        def radio_curvatura(x, y, a, b):
            return ((b ** 2 * (x - a) ** 2) + (a ** 2 * y ** 2)) ** (3 / 2) / (a ** 2 * b ** 2)

        radios_curvatura = radio_curvatura(x_vals, y_vals, b, a)


        st.write("### Radios de curvatura")
        col4, col5= st.columns(2)
        col4.metric("Extremo izquierdo", f"{radios_curvatura[0]} mm")
        col5.metric("Extremo derecho", f"{radios_curvatura[99]} mm")



        # Crear la primera gráfica: Proyección en la elipse


        # Graficar la elipse y la proyección
        theta = np.linspace(0, 2 * np.pi, 300)
        x_elipse = a/2 * np.cos(theta) + a/2  # Mover el centro al (a, b)
        y_elipse = b/2 * np.sin(theta) - b/2  # Ajustar el centro para que toque (0,0)

        fig2, ax2 = plt.subplots(figsize=(6, 8))
        ax2.plot(x_elipse, y_elipse, 'b', linewidth=2, label="Óvalo")
        ax2.plot(x_vals, y_vals, 'r--', linewidth=2, label="Proyección")
        sc = ax2.scatter(x_vals, y_vals, c=radios_curvatura, cmap='viridis', label="Curvatura")

        # Ejes y anotaciones
        ax2.axhline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.axvline(0, color='gray', linestyle='--', alpha=0.5)
        ax2.scatter([x0, x0 + 20], [0, 0], color='red', zorder=3, label="Inicio y Fin de la Proyección")

        ax2.set_xlim(-1, 2 * a + 21)
        ax2.set_ylim(-b - 1, 1)
        ax2.set_aspect('equal', adjustable='datalim')

        fig2.colorbar(sc, label="Radio de Curvatura")
        ax2.set_title("Proyección sobre el Óvalo y Radios de Curvatura")
        ax2.legend()

        # Mostrar la primera gráfica en Streamlit
        st.pyplot(fig2)




        # Segunda gráfica

        fig1, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(x_vals, radios_curvatura, label="Radio de Curvatura", color='b')
        ax1.axhline(0, color='gray', linestyle='--', alpha=0.5)  # Línea base
        ax1.set_xlabel("x (posición en el eje horizontal)")
        ax1.set_ylabel("Radio de Curvatura (R)")
        ax1.set_title(f"Curvatura para {selected_paciente}")
        ax1.legend()
        ax1.grid()

        # Mostrar la segunda gráfica en Streamlit
        st.pyplot(fig1)




