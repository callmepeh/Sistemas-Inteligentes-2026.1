import matplotlib.pyplot as plt
import os
import csv  

def ler_dataset():
    diretorio_do_script = os.path.dirname(os.path.abspath(__file__))
    caminho_do_csv = os.path.join(diretorio_do_script, "mobile_price_dataset.csv")

    with open(caminho_do_csv, mode= 'r') as file:
        reader = csv.DictReader(file)
        dataset = []
        for row in reader:
            preco = float(row["Price_USD"])
            armazenamento = float(row["Storage_GB"])
            dataset.append((preco, armazenamento))
    print("Dataset lido com sucesso!")
    return dataset


def definir_xe_y(dataset):
    x = []
    y = []
    for ponto in dataset:
        x.append(ponto[0])
        y.append(ponto[1])
    mostrar_xe_y(x, y)
    return x, y     
    
def mostrar_xe_y(x, y):
    print("Valores de x (Price_USD):", x)
    print("Valores de y (Storage_GB):", y)

def definirconjuntodetreinoeconjuntodeteste(x, y, proporcao_treino=0.7):
    n = len(x)
    n_treino = int(n * proporcao_treino)
    
    x_treino = x[:n_treino]
    y_treino = y[:n_treino]
    
    x_teste = x[n_treino:]
    y_teste = y[n_treino:]

    print(f"Conjunto de treino: {len(x_treino)} pontos")
    print(f"Conjunto de teste: {len(x_teste)} pontos")
    
    return x_treino, y_treino, x_teste, y_teste

def calcularAeB(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(xi ** 2 for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    
    return m, b

def calcular2metricasdedesempenho(y_teste, y_pred):
    n = len(y_teste)
    mse = sum((yt - yp) ** 2 for yt, yp in zip(y_teste, y_pred)) / n
    rmse = mse ** 0.5
    return mse, rmse

def calcular_regressao_simples(x, y):
    m, b = calcularAeB(x, y)
    return m, b

# def scatter_plot(x, y, m, b):
#     import matplotlib.pyplot as plt
#     plt.scatter(x, y, color='blue', label='Dados Reais')
#     x_line = [min(x), max(x)]
#     y_line = [m * xi + b for xi in x_line]
#     plt.plot(x_line, y_line, color='red', label='Linha de Regressão')
#     plt.xlabel('Price_USD')
#     plt.ylabel('Storage_GB')
#     plt.title('Regressão Simples: Price vs Storage')
#     plt.legend()
#     plt.show()

def scatter_plot(x_tr, y_tr, x_te, y_te, m, b, mse, r2):
    plt.figure(figsize=(10, 6))

    plt.scatter(x_tr, y_tr, color='blue', alpha=0.5, label='Treino (Reais)')
    plt.scatter(x_te, y_te, color='orange', marker='x', label='Teste (Reais)')
    
    x_total = x_tr + x_te
    line_x = [min(x_total), max(x_total)]
    line_y = [m * xi + b for xi in line_x]
    plt.plot(line_x, line_y, color='red', linewidth=2, label=f'Reta: y = {m:.2f}x + {b:.2f}')
    
    plt.title('Regressão Linear: Preço vs Armazenamento')
    plt.xlabel('Variável X (Independente): Preço em USD')
    plt.ylabel('Variável Y (Dependente): Armazenamento em GB')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    texto_metricas = f'Métricas do Modelo:\nMSE: {mse:.2f}\nR²: {r2:.4f}'
    plt.text(1.05, 0.5, texto_metricas, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout() # Ajusta o layout para não cortar a legenda externa
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()

def calcular_r2(y_real, y_pred):
    y_media = sum(y_real) / len(y_real)
    sq_res = sum((yr - yp) ** 2 for yr, yp in zip(y_real, y_pred))
    sq_total = sum((yr - y_media) ** 2 for yr in y_real)
    return 1 - (sq_res / sq_total)

# dataset = ler_dataset()
# x, y = definir_xe_y(dataset)
# x_treino, y_treino, x_teste, y_teste = definirconjuntodetreinoeconjuntodeteste(x, y, proporcao_treino=0.7)
# definirconjuntodetreinoeconjuntodeteste(x, y, proporcao_treino=0.7)
# calcularAeB(x_treino, y_treino)
# calcular2metricasdedesempenho(y_teste, y_treino)
# calcular_regressao_simples(x_treino, y_treino)
# m, b = calcular_coeficientes(x_treino, y_treino)
# scatter_plot(x, y, m, b)


def _main():
    dataset = ler_dataset()
    if not dataset:
        return

    x, y = definir_xe_y(dataset)

    x_treino, y_treino, x_teste, y_teste = definirconjuntodetreinoeconjuntodeteste(x, y, proporcao_treino=0.7)

    m, b = calcular_regressao_simples(x_treino, y_treino)
    print(f"\nModelo Treinado:")
    print(f"Coeficiente angular (m): {m:.4f}")
    print(f"Coeficiente linear (b): {b:.4f}")

    # y = mx + b
    y_predito = [m * xi + b for xi in x_teste]

    mse, rmse = calcular2metricasdedesempenho(y_teste, y_predito)
    r2 = calcular_r2(y_teste, y_predito)

    print(f"\nResultados:")
    print(f"MSE: {mse:.2f}")
    print(f"R²: {r2:.4f}")

    scatter_plot(x_treino, y_treino, x_teste, y_teste, m, b, mse, r2)

if __name__ == "__main__":
    _main()