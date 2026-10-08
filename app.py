import streamlit as st
import pandas as pd
import joblib


# ============================================================
# CONFIGURACIÓN DE LA PÁGINA
# ============================================================

st.set_page_config(
    page_title="Predicción de Demanda Energética",
    page_icon="🏢",
    layout="centered"
)


# ============================================================
# CARGAR SCALER Y MODELOS
# ============================================================

scaler = joblib.load("minmax_scaler.pkl")
modelo_heating = joblib.load("svm_heating.pkl")
modelo_cooling = joblib.load("red_neuronal_cooling.pkl")


# ============================================================
# TÍTULO
# ============================================================

st.title("🏢 Predicción de Demanda Energética")

st.write(
    "Ingrese las características físicas del edificio "
    "para estimar su demanda de calefacción y refrigeración."
)

st.info(
    "Ingrese los valores originales. "
    "La normalización se realiza automáticamente."
)


# ============================================================
# DATOS DEL EDIFICIO
# ============================================================

st.subheader("Características del edificio")


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


# ============================================================
# BOTÓN DE PREDICCIÓN
# ============================================================

if st.button("🔮 Realizar predicción"):

    # Crear DataFrame con las variables
    datos = pd.DataFrame({
        "Relative_Compactness": [relative_compactness],
        "Surface_Area": [surface_area],
        "Wall_Area": [wall_area],
        "Overall_Height": [overall_height],
        "Glazing_Area": [glazing_area]
    })


    # Normalizar automáticamente
    datos_normalizados = scaler.transform(datos)


    # Realizar predicciones
    heating_pred = modelo_heating.predict(datos_normalizados)[0]

    cooling_pred = modelo_cooling.predict(datos_normalizados)[0]


    # ========================================================
    # MOSTRAR RESULTADOS
    # ========================================================

    st.subheader("Resultados de la predicción")


    col1, col2 = st.columns(2)


    with col1:
        st.metric(
            label="🔥 Heating Load",
            value=f"{heating_pred:.2f}"
        )


    with col2:
        st.metric(
            label="❄️ Cooling Load",
            value=f"{cooling_pred:.2f}"
        )


    st.success("Predicción realizada correctamente.")