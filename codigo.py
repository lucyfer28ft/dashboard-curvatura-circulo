
df["Radio_Curvatura"] = df.apply(
    lambda row: radio_curvatura(row["Extremo_Plaque"], row["Eje_Menor"], row["Eje_Mayor"]), axis=1)

st.write("### Datos Calculados")
st.dataframe(df)

# Seleccionar un caso para graficar
selected_index = st.selectbox("Selecciona un índice de la tabla para visualizar", df.index)

# Obtener valores seleccionados
a = df.loc[selected_index, "Eje_Menor"]
b = df.loc[selected_index, "Eje_Mayor"]
x0 = df.loc[selected_index, "Extremo_Plaque"]

# Generar valores de x para el gráfico
x_vals = np.linspace(x0, x0 + 20, 100)
radii = np.array([radio_curvatura(x, a, b) for x in x_vals])

# Graficar la variación del radio de curvatura
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(x_vals, radii, label="Radio de Curvatura", color='b')
ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
ax.set_xlabel("x (posición en el eje horizontal)")
ax.set_ylabel("Radio de Curvatura (R)")
ax.set_title("Variación del Radio de Curvatura en la Proyección de la Línea sobre la Elipse")
ax.legend()
ax.grid()

st.pyplot(fig)

# Explicación detallada
st.write("### Explicación del Cálculo")
st.markdown(
    """
    - **Eje Mayor 1 y 2:** Representan los ejes mayores de los óvalos de la caja torácica.
    - **Suma de los Ejes Menores:** Distancia total entre los óvalos.
    - **Índice de Asimetría:** Mide la diferencia en simetría.
    - **Radio de Curvatura:** Calculado en función del extremo de la placa y los valores de la elipse.
    """
)
















#Resultados numéricos curvatura

# Mostrar resultados numéricos
st.write("### Resultados numéricos de curvatura:")
st.write(f"Radio de curvatura: {r_curvatura:.2f} cm")









#Codigo para eliminar ciertas columnas

,"STATE", "SUITABLE", "KIT", "NOTES", "IMAGES3", "SURGERY DATE", "MONTH", "COUNTRY2 CODE", "CITY2", "HOSPITAL","SURGEON", "COMBINED TECHNIQUE", "ANALGESICS INTRAOPERATIVE ", "COMPLICATIONS INTRAOPERATORY ","DRAINAGE PLACEMENT", "COMPRESSIVE BANDAGE PLACEMENT", "RESULT","IMAGES2", "NOTES2", "DATE FOLLOW-UP 1", "DAYS AFETER SURGERY 1", "DIAGNOSIS 1
