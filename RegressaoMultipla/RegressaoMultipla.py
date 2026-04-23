import matplotlib.pyplot as plt
import csv  
from pathlib import Path

FEATURES = [
    "RAM_GB",
    "Storage_GB",
    "Battery_mAh",
    "Screen_Size_Inch",
    "Camera_MP",
    "Release_Year",
]
TARGET = "Price_USD"
DATASET_FILE = Path(__file__).resolve().parent / "mobile_price_dataset.csv"

def ler_dataset():
    if not DATASET_FILE.exists():
        raise FileNotFoundError(f"Arquivo de dataset não encontrado: {DATASET_FILE}")

    with DATASET_FILE.open(mode="r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        X = []
        y = []
        for row in reader:
            x_row = [1.0] + [float(row[feature]) for feature in FEATURES]
            X.append(x_row)
            y.append(float(row[TARGET]))
    print("Dataset lido com sucesso!")
    return X, y

def definir_conjunto_treino_e_teste(X, y, proporcao_treino=0.7):
    n = len(X)
    n_treino = int(n * proporcao_treino)
    X_treino = X[:n_treino]
    y_treino = y[:n_treino]
    X_teste = X[n_treino:]
    y_teste = y[n_treino:]

    print(f"Conjunto de treino: {len(X_treino)} pontos")
    print(f"Conjunto de teste: {len(X_teste)} pontos")
    return X_treino, y_treino, X_teste, y_teste

def transpor(matriz):
    return [list(col) for col in zip(*matriz)]

def multiplicar_matrizes(a, b):
    result = [[0.0] * len(b[0]) for _ in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                result[i][j] += a[i][k] * b[k][j]
    return result

def multiplicar_matriz_vetor(matriz, vetor):
    return [sum(row[j] * vetor[j] for j in range(len(vetor))) for row in matriz]

def inverter_matriz(matriz):
    n = len(matriz)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)] for i, row in enumerate(matriz)]

    for i in range(n):
        pivot_row = max(range(i, n), key=lambda r: abs(aug[r][i]))
        if abs(aug[pivot_row][i]) < 1e-12:
            raise ValueError("Matriz singular ou quase singular")
        aug[i], aug[pivot_row] = aug[pivot_row], aug[i]

        pivot = aug[i][i]
        aug[i] = [value / pivot for value in aug[i]]

        for j in range(n):
            if j != i:
                factor = aug[j][i]
                aug[j] = [aug[j][k] - factor * aug[i][k] for k in range(2 * n)]

    return [[aug[i][j + n] for j in range(n)] for i in range(n)]

def calcular_coeficientes_multiplas(X, y):
    Xt = transpor(X)
    XtX = multiplicar_matrizes(Xt, X)
    XtY = multiplicar_matriz_vetor(Xt, y)
    XtX_inv = inverter_matriz(XtX)
    coeficientes = multiplicar_matriz_vetor(XtX_inv, XtY)
    return coeficientes

def prever(X, coeficientes):
    return [sum(coeficiente * xij for coeficiente, xij in zip(coeficientes, x_i)) for x_i in X]

def calcular_metricas(y_teste, y_pred):
    n = len(y_teste)
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_teste, y_pred)) / n
    media = sum(y_teste) / n
    ss_tot = sum((yt - media) ** 2 for yt in y_teste)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_teste, y_pred))
    r2 = 1 - ss_res / ss_tot if ss_tot != 0 else 0.0
    return mse, r2

def mostrar_coeficientes(coeficientes):
    print("Coeficientes da regressão linear múltipla:")
    print(f"Intercepto: {coeficientes[0]:.4f}")
    for nome, valor in zip(FEATURES, coeficientes[1:]):
        print(f"  {nome}: {valor:.4f}")

def scatter_plot(X_treino, y_treino, X_teste, y_teste, coeficientes, mse, r2):
    plt.figure(figsize=(10, 6))

    # Scatter plot para treino: Price vs Storage (índice 2 em X, pois [1.0, RAM, Storage, ...])
    plt.scatter([x[2] for x in X_treino], y_treino, color='blue', alpha=0.5, label='Treino (Reais)')
    plt.scatter([x[2] for x in X_teste], y_teste, color='orange', marker='x', label='Teste (Reais)')

    storage_idx = FEATURES.index("Storage_GB") + 1  # +1 porque X[0] é 1.0
    medias = [sum(x[i] for x in X_treino) / len(X_treino) for i in range(1, len(FEATURES)+1)]
    storage_min = min(x[storage_idx] for x in X_treino + X_teste)
    storage_max = max(x[storage_idx] for x in X_treino + X_teste)
    line_x = [storage_min, storage_max]
    line_y = []
    for s in line_x:
        x_pred = [1.0] + medias[:]
        x_pred[storage_idx] = s
        y_pred = sum(c * val for c, val in zip(coeficientes, x_pred))
        line_y.append(y_pred)
    plt.plot(line_x, line_y, color='red', linewidth=2, label='Linha de Regressão (múltipla)')

    plt.title('Regressão Linear Múltipla: Price vs Storage')
    plt.xlabel('Storage_GB')
    plt.ylabel('Price_USD')

    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    texto_metricas = f'Métricas do Modelo:\nMSE: {mse:.2f}\nR²: {r2:.4f}'
    plt.text(1.05, 0.5, texto_metricas, transform=plt.gca().transAxes,
             fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def _main():
    X, y = ler_dataset()
    if not X:
        return

    X_treino, y_treino, X_teste, y_teste = definir_conjunto_treino_e_teste(X, y, proporcao_treino=0.7)

    coeficientes = calcular_coeficientes_multiplas(X_treino, y_treino)
    mostrar_coeficientes(coeficientes)

    y_pred = prever(X_teste, coeficientes)
    mse, r2 = calcular_metricas(y_teste, y_pred)

    print(f"\nResultados:")
    print(f"MSE: {mse:.4f}")
    print(f"R²: {r2:.4f}")

    scatter_plot(X_treino, y_treino, X_teste, y_teste, coeficientes, mse, r2)

if __name__ == "__main__":
    _main()