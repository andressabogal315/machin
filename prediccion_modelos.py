
import joblib
import pandas as pd

# Cargar scaler
scaler = joblib.load("minmax_scaler.pkl")

# Cargar modelos finales
modelo_heating = joblib.load("svm_heating.pkl")
modelo_cooling = joblib.load("red_neuronal_cooling.pkl")

# Variables de entrada
variables_X = [
    "Relative_Compactness",
    "Surface_Area",
    "Wall_Area",
    "Overall_Height",
    "Glazing_Area"
]

# Leer datos de prueba
datos = pd.read_csv("datos_prueba_modelo.csv")

# Seleccionar variables
X = datos[variables_X]

# Escalar datos
X_scaled = scaler.transform(X)

# Realizar predicciones
datos["Heating_Load_Predicho"] = modelo_heating.predict(X_scaled)
datos["Cooling_Load_Predicho"] = modelo_cooling.predict(X_scaled)

# Guardar resultados
datos.to_csv(
    "predicciones_resultado.csv",
    index=False,
    encoding="utf-8-sig"
)

print("Predicciones generadas correctamente.")
print(datos)
