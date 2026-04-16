
import csv  

def ler_dataset():
   # dataset = "mobile_price_dataset.csv"
    with open("mobile_price_dataset.csv", mode= 'r') as file:
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
    m, b = calcular_coeficientes(x, y)
    return m, b


def calcular_coeficientes(x, y):
    n = len(x)
    sum_x = sum(x)
    sum_y = sum(y)
    sum_x_squared = sum(xi ** 2 for xi in x)
    sum_xy = sum(xi * yi for xi, yi in zip(x, y))
    
    m = (n * sum_xy - sum_x * sum_y) / (n * sum_x_squared - sum_x ** 2)
    b = (sum_y - m * sum_x) / n
    
    return m, b

def scatter_plot(x, y, m, b):
    import matplotlib.pyplot as plt
    plt.scatter(x, y, color='blue', label='Dados Reais')
    x_line = [min(x), max(x)]
    y_line = [m * xi + b for xi in x_line]
    plt.plot(x_line, y_line, color='red', label='Linha de Regressão')
    plt.xlabel('Price_USD')
    plt.ylabel('Storage_GB')
    plt.title('Regressão Simples: Price vs Storage')
    plt.legend()
    plt.show()

dataset = ler_dataset()
x, y = definir_xe_y(dataset)
x_treino, y_treino, x_teste, y_teste = definirconjuntodetreinoeconjuntodeteste(x, y, proporcao_treino=0.7)
definirconjuntodetreinoeconjuntodeteste(x, y, proporcao_treino=0.7)
calcularAeB(x_treino, y_treino)
calcular2metricasdedesempenho(y_teste, y_treino)
calcular_regressao_simples(x_treino, y_treino)
m, b = calcular_coeficientes(x_treino, y_treino)
scatter_plot(x, y, m, b)


def _main():
    mostrar_xe_y(x, y)
    m, b = calcular_regressao_simples(x, y)
    print(f"Coeficiente angular (m): {m}")
    print(f"Coeficiente linear (b): {b}")
    #dataset = ler_dataset()
    #x, y = definir_xe_y(dataset)
    #mostrar_xe_y(x, y)
    #m, b = calcular_regressao_simples(x, y)
    #print(f"Coeficiente angular (m): {m}")
    #print(f"Coeficiente linear (b): {b}")