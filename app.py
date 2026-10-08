
import streamlit as st
import pandas as pd
import numpy as np
import joblib

from sklearn.metrics import mean_absolute_error, mean_squared_error


# =========================================================
# CONFIGURACIÓN
# =========================================================

st.set_page_config(
    page_title="Predicción de Demanda Energética",
    page_icon="🏢",
    layout="wide"
)


# =========================================================
# VARIABLES DEL MODELO
# =========================================================

variables_modelo = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Overall_Height",
    "Glazing_Area"
]


# =========================================================
# CARGAR MODELOS
# =========================================================

@st.cache_resource
def cargar_modelos():
    scaler = joblib.load("minmax_scaler.pkl")
    modelo_heating = joblib.load("svm_heating.pkl")
    modelo_cooling = joblib.load("red_neuronal_cooling.pkl")

    # Comprobar el número de variables esperado
    if hasattr(scaler, "n_features_in_"):
        if scaler.n_features_in_ != len(variables_modelo):
            raise ValueError(
                "El escalador no fue configurado con las "
                "cinco variables esperadas."
            )

    return scaler, modelo_heating, modelo_cooling


try:
    scaler, modelo_heating, modelo_cooling = cargar_modelos()

except (FileNotFoundError, OSError, ValueError) as e:
    st.error(f"No se pudieron cargar los modelos: {e}")
    st.stop()


# =========================================================
# FUNCIONES AUXILIARES
# =========================================================

def preparar_datos(datos):
    """Selecciona las variables y convierte sus valores a números."""

    X = datos[variables_modelo].copy()

    for columna in variables_modelo:
        X[columna] = pd.to_numeric(
            X[columna],
            errors="coerce"
        )

    # Los infinitos también se consideran valores inválidos
    X = X.replace([np.inf, -np.inf], np.nan)

    return X


def calcular_metricas(valores_reales, predicciones):
    """Calcula MAE y RMSE excluyendo valores reales inválidos."""

    reales = pd.to_numeric(
        valores_reales,
        errors="coerce"
    ).to_numpy(dtype=float)

    predicciones = np.asarray(
        predicciones,
        dtype=float
    )

    mascara = np.isfinite(reales) & np.isfinite(predicciones)

    if not mascara.any():
        return None

    reales_validos = reales[mascara]
    predicciones_validas = predicciones[mascara]

    mae = mean_absolute_error(
        reales_validos,
        predicciones_validas
    )

    rmse = np.sqrt(
        mean_squared_error(
            reales_validos,
            predicciones_validas
        )
    )

    return mae, rmse, int(mascara.sum())


# =========================================================
# TÍTULO
# =========================================================

st.title("🏢 Predicción de Demanda Energética")

st.write(
    "Esta aplicación permite cargar información de edificios "
    "desde un archivo CSV o Excel y realizar predicciones "
    "de demanda de calefacción y refrigeración."
)

st.info(
    "El archivo puede contener muchas columnas. "
    "El modelo utilizará únicamente las cinco variables "
    "necesarias para realizar las predicciones."
)


# =========================================================
# CARGAR ARCHIVO
# =========================================================

st.subheader("📂 Cargar archivo")

archivo = st.file_uploader(
    "Seleccione un archivo CSV o Excel",
    type=["csv", "xlsx", "xls"]
)


if archivo is not None:

    # -----------------------------------------------------
    # LEER ARCHIVO
    # -----------------------------------------------------

    try:
        if archivo.name.lower().endswith(".csv"):
            datos = pd.read_csv(archivo)
        else:
            datos = pd.read_excel(archivo)

        st.success(
            f"Archivo '{archivo.name}' cargado correctamente."
        )

    except Exception as e:
        st.error(
            f"No se pudo leer el archivo. "
            f"Compruebe su formato. Detalle: {e}"
        )
        st.stop()

    # -----------------------------------------------------
    # INFORMACIÓN DEL ARCHIVO
    # -----------------------------------------------------

    st.subheader("📋 Datos cargados")

    col1, col2, col3 = st.columns(3)

    col1.metric("Registros", datos.shape[0])
    col2.metric("Columnas", datos.shape[1])
    col3.metric(
        "Valores faltantes",
        int(datos.isnull().sum().sum())
    )

    st.dataframe(
        datos,
        use_container_width=True
    )

    # -----------------------------------------------------
    # VALIDAR COLUMNAS
    # -----------------------------------------------------

    st.subheader("🔎 Verificación de variables")

    faltantes = [
        variable
        for variable in variables_modelo
        if variable not in datos.columns
    ]

    if faltantes:

        st.error(
            "El archivo no contiene todas las variables "
            "necesarias para realizar la predicción."
        )

        st.write("Variables faltantes:")

        for variable in faltantes:
            st.write(f"❌ {variable}")

    else:

        st.success(
            "Todas las variables necesarias están presentes."
        )

        st.write("Variables utilizadas por los modelos:")

        st.code("\n".join(variables_modelo))

        # -------------------------------------------------
        # PREPARAR DATOS
        # -------------------------------------------------

        X = preparar_datos(datos)

        filas_validas = ~X.isnull().any(axis=1)

        cantidad_invalidas = int((~filas_validas).sum())

        if cantidad_invalidas > 0:

            st.warning(
                f"Se encontraron {cantidad_invalidas} filas "
                "con variables incompletas o no numéricas. "
                "Estas filas no se utilizarán para predecir."
            )

            faltantes_por_columna = X.isnull().sum()
            faltantes_por_columna = faltantes_por_columna[
                faltantes_por_columna > 0
            ]

            st.write("Valores inválidos por variable:")
            st.dataframe(faltantes_por_columna)

        # -------------------------------------------------
        # BOTÓN DE PREDICCIÓN
        # -------------------------------------------------

        if st.button(
            "🔮 Realizar predicción",
            type="primary",
            key="predecir_archivo"
        ):

            X_validos = X.loc[filas_validas].copy()

            if X_validos.empty:

                st.error(
                    "No existen registros completos para "
                    "realizar la predicción."
                )

            else:

                try:
                    # -------------------------------------
                    # NORMALIZAR
                    # -------------------------------------

                    X_normalizado = scaler.transform(X_validos)

                    # -------------------------------------
                    # PREDICCIONES
                    # -------------------------------------

                    heating_pred = modelo_heating.predict(
                        X_normalizado
                    )

                    cooling_pred = modelo_cooling.predict(
                        X_normalizado
                    )

                    # -------------------------------------
                    # CREAR RESULTADOS
                    # -------------------------------------

                    resultados = datos.loc[
                        filas_validas
                    ].copy()

                    resultados["Heating_Load_Predicho"] = (
                        heating_pred
                    )

                    resultados["Cooling_Load_Predicho"] = (
                        cooling_pred
                    )

                    resultados["Total_Load_Predicho"] = (
                        resultados["Heating_Load_Predicho"]
                        + resultados["Cooling_Load_Predicho"]
                    )

                    # -------------------------------------
                    # COMPROBAR CARGAS REALES
                    # -------------------------------------

                    tiene_heating_real = (
                        "Heating_Load" in resultados.columns
                    )

                    tiene_cooling_real = (
                        "Cooling_Load" in resultados.columns
                    )

                    if tiene_heating_real:

                        resultados["Heating_Load"] = pd.to_numeric(
                            resultados["Heating_Load"],
                            errors="coerce"
                        )

                        resultados["Error_Heating"] = (
                            resultados["Heating_Load"]
                            - resultados["Heating_Load_Predicho"]
                        )

                    if tiene_cooling_real:

                        resultados["Cooling_Load"] = pd.to_numeric(
                            resultados["Cooling_Load"],
                            errors="coerce"
                        )

                        resultados["Error_Cooling"] = (
                            resultados["Cooling_Load"]
                            - resultados["Cooling_Load_Predicho"]
                        )

                    if tiene_heating_real and tiene_cooling_real:

                        resultados["Total_Load"] = (
                            resultados["Heating_Load"]
                            + resultados["Cooling_Load"]
                        )

                    # -------------------------------------
                    # MOSTRAR RESULTADOS
                    # -------------------------------------

                    st.subheader("📊 Resultados de la predicción")

                    st.dataframe(
                        resultados,
                        use_container_width=True
                    )

                    # -------------------------------------
                    # MÉTRICAS DE EVALUACIÓN
                    # -------------------------------------

                    if tiene_heating_real or tiene_cooling_real:

                        st.subheader(
                            "📈 Evaluación de las predicciones"
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            st.markdown("### 🔥 Heating Load")

                            if tiene_heating_real:

                                metricas = calcular_metricas(
                                    resultados["Heating_Load"],
                                    heating_pred
                                )

                                if metricas is not None:

                                    mae, rmse, n = metricas

                                    st.metric("MAE", f"{mae:.2f}")
                                    st.metric("RMSE", f"{rmse:.2f}")

                                    st.caption(
                                        f"Registros evaluados: {n}"
                                    )

                                else:
                                    st.warning(
                                        "No hay valores reales válidos "
                                        "para evaluar calefacción."
                                    )

                            else:
                                st.info(
                                    "El archivo no contiene "
                                    "Heating_Load real."
                                )

                        with col2:

                            st.markdown("### ❄️ Cooling Load")

                            if tiene_cooling_real:

                                metricas = calcular_metricas(
                                    resultados["Cooling_Load"],
                                    cooling_pred
                                )

                                if metricas is not None:

                                    mae, rmse, n = metricas

                                    st.metric("MAE", f"{mae:.2f}")
                                    st.metric("RMSE", f"{rmse:.2f}")

                                    st.caption(
                                        f"Registros evaluados: {n}"
                                    )

                                else:
                                    st.warning(
                                        "No hay valores reales válidos "
                                        "para evaluar refrigeración."
                                    )

                            else:
                                st.info(
                                    "El archivo no contiene "
                                    "Cooling_Load real."
                                )

                    else:

                        st.info(
                            "El archivo no contiene las columnas "
                            "de demanda real. Se mostrarán las "
                            "predicciones sin métricas de evaluación."
                        )

                    # -------------------------------------
                    # DESCARGAR RESULTADOS
                    # -------------------------------------

                    st.subheader("📥 Descargar resultados")

                    csv_resultados = resultados.to_csv(
                        index=False
                    ).encode("utf-8-sig")

                    st.download_button(
                        label="⬇️ Descargar CSV",
                        data=csv_resultados,
                        file_name="predicciones_resultado.csv",
                        mime="text/csv"
                    )

                except (ValueError, TypeError, AttributeError) as e:

                    st.error(
                        "Ocurrió un error al preparar los datos "
                        f"o ejecutar los modelos: {e}"
                    )


# =========================================================
# SEPARADOR
# =========================================================

st.divider()


# =========================================================
# PREDICCIÓN MANUAL
# =========================================================

st.subheader("✏️ Evaluar un edificio manualmente")

col1, col2 = st.columns(2)

with col1:

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

with col2:

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


# =========================================================
# EJECUTAR PREDICCIÓN MANUAL
# =========================================================

if st.button(
    "🔮 Predecir edificio",
    type="primary",
    key="predecir_manual"
):

    datos_manual = pd.DataFrame({
        "Relative_Compactness": [relative_compactness],
        "Surface_Area": [surface_area],
        "Wall_Area": [wall_area],
        "Overall_Height": [overall_height],
        "Glazing_Area": [glazing_area]
    })

    try:

        # ---------------------------------------------
        # NORMALIZAR
        # ---------------------------------------------

        datos_normalizado = scaler.transform(datos_manual)

        # ---------------------------------------------
        # PREDICCIONES
        # ---------------------------------------------

        heating_pred = float(
            modelo_heating.predict(datos_normalizado)[0]
        )

        cooling_pred = float(
            modelo_cooling.predict(datos_normalizado)[0]
        )

        total_pred = heating_pred + cooling_pred

        # ---------------------------------------------
        # MOSTRAR RESULTADOS
        # ---------------------------------------------

        st.subheader("📊 Resultado de la predicción")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "🔥 Heating Load",
            f"{heating_pred:.2f}"
        )

        col2.metric(
            "❄️ Cooling Load",
            f"{cooling_pred:.2f}"
        )

        col3.metric(
            "⚡ Total Load",
            f"{total_pred:.2f}"
        )

        st.success(
            "Predicción realizada correctamente."
        )

    except (ValueError, TypeError, AttributeError) as e:

        st.error(
            f"No se pudo realizar la predicción manual: {e}"
        )