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
    X_treino, y_treino = X[:n_treino], y[:n_treino]
    X_teste, y_teste = X[n_treino:], y[n_treino:]

    print(f"Dataset: {n} pontos | Treino: {len(X_treino)} | Teste: {len(X_teste)}")
    return X_treino, y_treino, X_teste, y_teste

def normalizar_zscore(X_treino, X_teste):
    n_features = len(X_treino[0]) - 1
    means = [sum(row[i+1] for row in X_treino) / len(X_treino) for i in range(n_features)]
    stds = [(sum((row[i+1] - m)**2 for row in X_treino) / len(X_treino))**0.5 for i, m in enumerate(means)]

    def aplicar(X):
        return [[1.0] + [(row[i+1] - means[i]) / (stds[i] or 1.0) for i in range(n_features)] for row in X]

    return aplicar(X_treino), aplicar(X_teste)

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

def gradiente_descendente(X, y, alpha, epochs):
    n_samples = len(X)
    n_features = len(X[0])
    coefs = [0.0] * n_features
    historico_perda = []

    for epoch in range(epochs):
        y_pred = prever(X, coefs)
        erros = [yp - yt for yp, yt in zip(y_pred, y)]
        
        # Cálculo dos gradientes
        gradientes = [0.0] * n_features
        for i in range(n_samples):
            for j in range(n_features):
                gradientes[j] += erros[i] * X[i][j]
        
        # Atualização dos coeficientes
        for j in range(n_features):
            coefs[j] -= (alpha / n_samples) * gradientes[j]
        
        # Salva a perda (MSE) a cada época
        mse = sum(e**2 for e in erros) / n_samples
        historico_perda.append(mse)
        
        # Critério de parada se divergir
        if mse > 1e20: break 
            
    return coefs, historico_perda

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

def plot_comparacao(original, normalizado):
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    
    def plot_subset(ax_loss, ax_pred, dados, titulo):
        Xt, yt, Xts, yts, coefs, mse, r2, loss_hist = dados
        
        # 1. Curva de Perda
        ax_loss.plot(range(len(loss_hist)), loss_hist, color='red', linewidth=1.5)
        ax_loss.set_title(f'Curva de Perda: {titulo}', fontsize=14, pad=15)
        ax_loss.set_xlabel('Épocas', fontsize=12)
        ax_loss.set_ylabel('MSE (Perda)', fontsize=12)
        ax_loss.grid(True, linestyle='--', alpha=0.5)

        # 2. Real vs Predito
        y_pred_tr = prever(Xt, coefs)
        y_pred_te = prever(Xts, coefs)
        ax_pred.scatter(yt, y_pred_tr, color='blue', alpha=0.5, label='Treino', s=40)
        ax_pred.scatter(yts, y_pred_te, color='orange', marker='x', label='Teste', s=60)
        
        lims = [min(yt + y_pred_tr), max(yt + y_pred_tr)]
        ax_pred.plot(lims, lims, 'r--', alpha=0.7, label='Ideal')
        ax_pred.set_title(f'Real vs Predito: {titulo}\nMSE: {mse:.2f} | R²: {r2:.4f}', fontsize=14, pad=15)
        ax_pred.set_xlabel('Preço Real (USD)', fontsize=12)
        ax_pred.set_ylabel('Preço Predito (USD)', fontsize=12)
        ax_pred.legend(fontsize=10)
        ax_pred.grid(True, linestyle='--', alpha=0.5)

    plot_subset(axes[0, 0], axes[1, 0], original, "Original")
    plot_subset(axes[0, 1], axes[1, 1], normalizado, "Normalizado")
    
    plt.subplots_adjust(hspace=0.4, wspace=0.3, top=0.92, bottom=0.08, left=0.08, right=0.92)
    plt.show()

def treinar_e_avaliar(Xt, yt, Xts, yts, alpha, epochs):
    coefs, loss_hist = gradiente_descendente(Xt, yt, alpha, epochs)
    y_pred = prever(Xts, coefs)
    mse, r2 = calcular_metricas(yts, y_pred)
    return coefs, mse, r2, loss_hist

def _main():
    X, y = ler_dataset()
    if not X: return

    X_tr, y_tr, X_te, y_te = definir_conjunto_treino_e_teste(X, y)

    # 1. Modelo Original (Necessita de alpha baixíssimo para não explodir)
    print("\n--- Modelo Original ---")
    alpha_orig = 1e-9 
    epochs = 2000
    res_orig = treinar_e_avaliar(X_tr, y_tr, X_te, y_te, alpha_orig, epochs)
    mostrar_coeficientes(res_orig[0])

    # 2. Modelo Normalizado (Suporta alpha maior e converge rápido)
    print("\n--- Modelo Normalizado ---")
    X_tr_n, X_te_n = normalizar_zscore(X_tr, X_te)
    alpha_norm = 0.1
    res_norm = treinar_e_avaliar(X_tr_n, y_tr, X_te_n, y_te, alpha_norm, epochs)
    mostrar_coeficientes(res_norm[0])

    plot_comparacao(
        (X_tr, y_tr, X_te, y_te) + res_orig,
        (X_tr_n, y_tr, X_te_n, y_te) + res_norm
    )

if __name__ == "__main__":
    _main()