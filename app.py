import streamlit as st
import pandas as pd
import joblib

# ---------------------------------------------------------
# Configuración
# ---------------------------------------------------------
st.set_page_config(
    page_title="Predicción de Demanda Energética",
    page_icon="🏢",
    layout="centered"
)

# ---------------------------------------------------------
# Cargar modelos y scaler
# ---------------------------------------------------------
scaler = joblib.load("minmax_scaler.pkl")
modelo_heating = joblib.load("svm_heating.pkl")
modelo_cooling = joblib.load("red_neuronal_cooling.pkl")

# ---------------------------------------------------------
# Título
# ---------------------------------------------------------
st.title("🏢 Predicción de Demanda Energética")

st.write(
    "Esta aplicación permite estimar la demanda energética "
    "de calefacción y refrigeración de edificios."
)

st.info(
    "Puede ingresar los datos manualmente o cargar un archivo "
    "CSV o Excel con las características de los edificios."
)

# ---------------------------------------------------------
# Variables requeridas
# ---------------------------------------------------------
variables_modelo = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Overall_Height",
    "Glazing_Area"
]

# ---------------------------------------------------------
# Opción 1: Cargar archivo
# ---------------------------------------------------------
st.subheader("📂 Cargar archivo")

archivo = st.file_uploader(
    "Seleccione un archivo CSV o Excel",
    type=["csv", "xlsx", "xls"]
)

if archivo is not None:

    try:
        # Leer CSV
        if archivo.name.endswith(".csv"):
            datos = pd.read_csv(archivo)

        # Leer Excel
        else:
            datos = pd.read_excel(archivo)

        st.success("Archivo cargado correctamente.")

        st.subheader("📋 Datos cargados")
        st.dataframe(datos)

        # -------------------------------------------------
        # Verificar variables
        # -------------------------------------------------
        faltantes = [
            variable
            for variable in variables_modelo
            if variable not in datos.columns
        ]

        if faltantes:

            st.error(
                "El archivo no contiene las siguientes variables requeridas:"
            )

            for variable in faltantes:
                st.write(f"- {variable}")

        else:

            # ---------------------------------------------
            # Seleccionar variables del modelo
            # ---------------------------------------------
            X = datos[variables_modelo].copy()

            # Verificar valores faltantes
            if X.isnull().any().any():

                st.error(
                    "El archivo contiene valores vacíos en las variables "
                    "utilizadas por el modelo."
                )

            else:

                # -----------------------------------------
                # Botón de predicción
                # -----------------------------------------
                if st.button("🔮 Realizar predicción"):

                    # Normalización
                    X_normalizado = scaler.transform(X)

                    # Predicciones
                    heating_pred = modelo_heating.predict(
                        X_normalizado
                    )

                    cooling_pred = modelo_cooling.predict(
                        X_normalizado
                    )

                    # -------------------------------------
                    # Agregar resultados
                    # -------------------------------------
                    resultados = datos.copy()

                    resultados["Heating_Load_Predicho"] = heating_pred
                    resultados["Cooling_Load_Predicho"] = cooling_pred

                    # -------------------------------------
                    # Mostrar resultados
                    # -------------------------------------
                    st.subheader("📊 Resultados de la predicción")

                    st.dataframe(resultados)

                    # -------------------------------------
                    # Descargar resultados
                    # -------------------------------------
                    csv_resultados = resultados.to_csv(
                        index=False,
                        encoding="utf-8-sig"
                    )

                    st.download_button(
                        label="⬇️ Descargar resultados CSV",
                        data=csv_resultados,
                        file_name="predicciones_resultado.csv",
                        mime="text/csv"
                    )

    except Exception as e:

        st.error(
            f"No fue posible procesar el archivo: {e}"
        )


# ---------------------------------------------------------
# Opción 2: Entrada manual
# ---------------------------------------------------------
st.divider()

st.subheader("✏️ Evaluar un edificio manualmente")

relative_compactness = st.number_input(
    "Relative Compactness",
    min_value=0.0,
    max_value=1.0,
    value=0.80,
    step=0.01
)

surface_area = st.number_input(
    "Surface Area",
    min_value=0.0,
    value=650.0,
    step=1.0
)

wall_area = st.number_input(
    "Wall Area",
    min_value=0.0,
    value=335.0,
    step=1.0
)

overall_height = st.number_input(
    "Overall Height",
    min_value=0.0,
    value=5.3,
    step=0.1
)

glazing_area = st.number_input(
    "Glazing Area",
    min_value=0.0,
    max_value=1.0,
    value=0.20,
    step=0.01
)

if st.button("🔮 Predecir edificio"):

    datos_manual = pd.DataFrame({
        "Relative_Compactness": [relative_compactness],
        "Surface_Area": [surface_area],
        "Wall_Area": [wall_area],
        "Overall_Height": [overall_height],
        "Glazing_Area": [glazing_area]
    })

    datos_normalizado = scaler.transform(datos_manual)

    heating_pred = modelo_heating.predict(
        datos_normalizado
    )[0]

    cooling_pred = modelo_cooling.predict(
        datos_normalizado
    )[0]

    st.subheader("📊 Resultado")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "🔥 Heating Load",
            f"{heating_pred:.2f}"
        )

    with col2:
        st.metric(
            "❄️ Cooling Load",
            f"{cooling_pred:.2f}"
        )