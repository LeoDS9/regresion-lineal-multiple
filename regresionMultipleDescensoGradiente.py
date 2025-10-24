import numpy as np
import pandas as pd

# ==========================
# 1. Leer los datos desde CSV
# ==========================
data = pd.read_csv("datos.csv")

# Eliminar columna ID
data = data.drop(columns=["ID"])

# ==========================
# 2. Preparar X e y
# ==========================
X = data[["TV", "Radio", "Newspaper"]].values
y = data["Sales"].values.reshape(-1, 1)

# Agregar columna de unos para el intercepto
X = np.c_[np.ones(X.shape[0]), X]

# Normalizar las variables (excepto la columna de 1s)
X_mean = np.mean(X[:, 1:], axis=0)
X_std = np.std(X[:, 1:], axis=0)
X[:, 1:] = (X[:, 1:] - X_mean) / X_std

# ==========================
# 3. Descenso del gradiente
# ==========================
alpha = 0.01      # Tasa de aprendizaje
num_iters = 3000 # Iteraciones
m = len(y)
theta = np.zeros((X.shape[1], 1))

for i in range(num_iters):
    gradients = (1/m) * X.T.dot(X.dot(theta) - y)
    theta -= alpha * gradients

    #chequear si el cambio es menor a la tolerancia
    if np.linalg.norm(alpha * gradients) < 1e-6:
        print(f"Convergencia alcanzada en la iteración {i}.")
        break
   


# ==========================
# 4. Resultados
# ==========================
print("\nCoeficientes estimados (theta):")
print(theta)

print("\nEcuación estimada:")
print(f"y = {theta[0][0]:.4f} + {theta[1][0]:.4f}*TV_norm + {theta[2][0]:.4f}*Radio_norm + {theta[3][0]:.4f}*Newspaper_norm")

# ==========================
# 5. Evaluar el modelo
# ==========================
y_pred = X.dot(theta)
mse = np.mean((y - y_pred)**2)
print(f"\nError cuadrático medio (MSE): {mse:.4f}")
