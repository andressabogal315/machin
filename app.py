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
# CARGAR MODELOS
# =========================================================

@st.cache_resource
def cargar_modelos():

    scaler = joblib.load("minmax_scaler.pkl")

    modelo_heating = joblib.load(
        "svm_heating.pkl"
    )

    modelo_cooling = joblib.load(
        "red_neuronal_cooling.pkl"
    )

    return scaler, modelo_heating, modelo_cooling


scaler, modelo_heating, modelo_cooling = cargar_modelos()


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
    "El modelo utilizará únicamente las variables necesarias "
    "para realizar la predicción."
)


# =========================================================
# VARIABLES QUE UTILIZA EL MODELO
# =========================================================

variables_modelo = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Overall_Height",
    "Glazing_Area"
]


# =========================================================
# CARGAR ARCHIVO
# =========================================================

st.subheader("📂 Cargar archivo")

archivo = st.file_uploader(
    "Seleccione un archivo CSV o Excel",
    type=["csv", "xlsx", "xls"]
)


if archivo is not None:

    try:

        # -------------------------------------------------
        # LEER CSV
        # -------------------------------------------------

        if archivo.name.lower().endswith(".csv"):

            datos = pd.read_csv(archivo)

        # -------------------------------------------------
        # LEER EXCEL
        # -------------------------------------------------

        else:

            datos = pd.read_excel(archivo)


        # -------------------------------------------------
        # MENSAJE DE ÉXITO
        # -------------------------------------------------

        st.success(
            f"Archivo '{archivo.name}' cargado correctamente."
        )


        # =================================================
        # INFORMACIÓN DEL ARCHIVO
        # =================================================

        st.subheader("📋 Datos cargados")

        col1, col2, col3 = st.columns(3)

        with col1:

            st.metric(
                "Registros",
                datos.shape[0]
            )

        with col2:

            st.metric(
                "Columnas",
                datos.shape[1]
            )

        with col3:

            st.metric(
                "Valores faltantes",
                int(datos.isnull().sum().sum())
            )


        # -------------------------------------------------
        # MOSTRAR DATASET
        # -------------------------------------------------

        st.dataframe(
            datos,
            use_container_width=True
        )


        # =================================================
        # VERIFICAR COLUMNAS DEL MODELO
        # =================================================

        st.subheader(
            "🔎 Verificación de variables"
        )

        faltantes = [
            variable
            for variable in variables_modelo
            if variable not in datos.columns
        ]


        # -------------------------------------------------
        # SI FALTAN COLUMNAS
        # -------------------------------------------------

        if faltantes:

            st.error(
                "❌ El archivo no contiene todas las "
                "variables necesarias para realizar "
                "la predicción."
            )

            st.write(
                "Variables faltantes:"
            )

            for variable in faltantes:

                st.write(
                    f"❌ {variable}"
                )


        # -------------------------------------------------
        # SI ESTÁN TODAS LAS COLUMNAS
        # -------------------------------------------------

        else:

            st.success(
                "✅ Todas las variables necesarias "
                "para el modelo están presentes."
            )


            # =================================================
            # MOSTRAR VARIABLES UTILIZADAS
            # =================================================

            st.write(
                "Variables utilizadas por los modelos:"
            )

            st.code(
                "\n".join(variables_modelo)
            )


            # =================================================
            # PREPARAR DATOS
            # =================================================

            X = datos[
                variables_modelo
            ].copy()


            # -------------------------------------------------
            # CONVERTIR A NUMÉRICO
            # -------------------------------------------------

            for columna in variables_modelo:

                X[columna] = pd.to_numeric(
                    X[columna],
                    errors="coerce"
                )


            # =================================================
            # VALORES FALTANTES
            # =================================================

            faltantes_por_columna = X.isnull().sum()

            hay_faltantes = (
                faltantes_por_columna.sum() > 0
            )


            if hay_faltantes:

                st.warning(
                    "⚠️ Algunas filas contienen "
                    "valores faltantes en las variables "
                    "utilizadas por el modelo."
                )

                st.write(
                    faltantes_por_columna[
                        faltantes_por_columna > 0
                    ]
                )

                st.info(
                    "Las filas incompletas no serán utilizadas "
                    "para realizar la predicción."
                )


            # =================================================
            # BOTÓN DE PREDICCIÓN
            # =================================================

            if st.button(
                "🔮 Realizar predicción",
                type="primary"
            ):

                # -------------------------------------------------
                # FILAS COMPLETAS
                # -------------------------------------------------

                filas_validas = ~X.isnull().any(axis=1)

                X_validos = X[
                    filas_validas
                ].copy()


                # -------------------------------------------------
                # COMPROBAR QUE HAYA DATOS
                # -------------------------------------------------

                if len(X_validos) == 0:

                    st.error(
                        "❌ No existen registros completos "
                        "para realizar la predicción."
                    )

                else:

                    # =================================================
                    # NORMALIZAR
                    # =================================================

                    X_normalizado = scaler.transform(
                        X_validos
                    )


                    # =================================================
                    # PREDICCIÓN HEATING
                    # =================================================

                    heating_pred = modelo_heating.predict(
                        X_normalizado
                    )


                    # =================================================
                    # PREDICCIÓN COOLING
                    # =================================================

                    cooling_pred = modelo_cooling.predict(
                        X_normalizado
                    )


                    # =================================================
                    # CREAR RESULTADOS
                    # =================================================

                    resultados = datos[
                        filas_validas
                    ].copy()


                    resultados[
                        "Heating_Load_Predicho"
                    ] = heating_pred


                    resultados[
                        "Cooling_Load_Predicho"
                    ] = cooling_pred


                    # =================================================
                    # TOTAL LOAD REAL
                    # =================================================

                    if (
                        "Heating_Load" in resultados.columns
                        and
                        "Cooling_Load" in resultados.columns
                    ):

                        resultados["Total_Load"] = (
                            pd.to_numeric(
                                resultados["Heating_Load"],
                                errors="coerce"
                            )
                            +
                            pd.to_numeric(
                                resultados["Cooling_Load"],
                                errors="coerce"
                            )
                        )


                    # =================================================
                    # TOTAL LOAD PREDICHO
                    # =================================================

                    resultados[
                        "Total_Load_Predicho"
                    ] = (
                        resultados[
                            "Heating_Load_Predicho"
                        ]
                        +
                        resultados[
                            "Cooling_Load_Predicho"
                        ]
                    )


                    # =================================================
                    # COMPARAR CON VALORES REALES
                    # =================================================

                    if (
                        "Heating_Load" in resultados.columns
                        and
                        "Cooling_Load" in resultados.columns
                    ):

                        resultados[
                            "Error_Heating"
                        ] = (
                            pd.to_numeric(
                                resultados["Heating_Load"],
                                errors="coerce"
                            )
                            -
                            resultados[
                                "Heating_Load_Predicho"
                            ]
                        )


                        resultados[
                            "Error_Cooling"
                        ] = (
                            pd.to_numeric(
                                resultados["Cooling_Load"],
                                errors="coerce"
                            )
                            -
                            resultados[
                                "Cooling_Load_Predicho"
                            ]
                        )


                    # =================================================
                    # RESULTADOS
                    # =================================================

                    st.subheader(
                        "📊 Resultados de la predicción"
                    )

                    st.dataframe(
                        resultados,
                        use_container_width=True
                    )


                    # =================================================
                    # MÉTRICAS
                    # =================================================

                    if (
                        "Heating_Load" in resultados.columns
                        and
                        "Cooling_Load" in resultados.columns
                    ):

                        heating_real = pd.to_numeric(
                            resultados["Heating_Load"],
                            errors="coerce"
                        )

                        cooling_real = pd.to_numeric(
                            resultados["Cooling_Load"],
                            errors="coerce"
                        )

                        # -----------------------------------------
                        # MAE
                        # -----------------------------------------

                        heating_mae = mean_absolute_error(
                            heating_real,
                            heating_pred
                        )

                        cooling_mae = mean_absolute_error(
                            cooling_real,
                            cooling_pred
                        )


                        # -----------------------------------------
                        # RMSE
                        # -----------------------------------------

                        heating_rmse = np.sqrt(
                            mean_squared_error(
                                heating_real,
                                heating_pred
                            )
                        )

                        cooling_rmse = np.sqrt(
                            mean_squared_error(
                                cooling_real,
                                cooling_pred
                            )
                        )


                        # =================================================
                        # MOSTRAR MÉTRICAS
                        # =================================================

                        st.subheader(
                            "📈 Evaluación de las predicciones"
                        )


                        col1, col2 = st.columns(2)


                        with col1:

                            st.markdown(
                                "### 🔥 Heating Load"
                            )

                            st.metric(
                                "MAE",
                                f"{heating_mae:.2f}"
                            )

                            st.metric(
                                "RMSE",
                                f"{heating_rmse:.2f}"
                            )


                        with col2:

                            st.markdown(
                                "### ❄️ Cooling Load"
                            )

                            st.metric(
                                "MAE",
                                f"{cooling_mae:.2f}"
                            )

                            st.metric(
                                "RMSE",
                                f"{cooling_rmse:.2f}"
                            )


                    # =================================================
                    # DESCARGAR RESULTADOS
                    # =================================================

                    st.subheader(
                        "📥 Descargar resultados"
                    )


                    csv_resultados = resultados.to_csv(
                        index=False,
                        encoding="utf-8-sig"
                    )


                    st.download_button(
                        label="⬇️ Descargar CSV",
                        data=csv_resultados,
                        file_name="predicciones_resultado.csv",
                        mime="text/csv"
                    )


# =========================================================
# SEPARADOR
# =========================================================

st.divider()


# =========================================================
# PREDICCIÓN MANUAL
# =========================================================

st.subheader(
    "✏️ Evaluar un edificio manualmente"
)


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
# PREDICCIÓN MANUAL
# =========================================================

if st.button(
    "🔮 Predecir edificio"
):

    datos_manual = pd.DataFrame({

        "Relative_Compactness": [
            relative_compactness
        ],

        "Surface_Area": [
            surface_area
        ],

        "Wall_Area": [
            wall_area
        ],

        "Overall_Height": [
            overall_height
        ],

        "Glazing_Area": [
            glazing_area
        ]

    })


    # -----------------------------------------------------
    # NORMALIZAR
    # -----------------------------------------------------

    datos_normalizado = scaler.transform(
        datos_manual
    )


    # -----------------------------------------------------
    # PREDICCIONES
    # -----------------------------------------------------

    heating_pred = modelo_heating.predict(
        datos_normalizado
    )[0]


    cooling_pred = modelo_cooling.predict(
        datos_normalizado
    )[0]


    total_pred = (
        heating_pred
        +
        cooling_pred
    )


    # =====================================================
    # MOSTRAR RESULTADOS
    # =====================================================

    st.subheader(
        "📊 Resultado"
    )


    col1, col2, col3 = st.columns(3)


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


    with col3:

        st.metric(
            "⚡ Total Load",
            f"{total_pred:.2f}"
        )
