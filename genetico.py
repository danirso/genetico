import numpy as np
import random
import matplotlib.pyplot as plt

# Gerar pontos aleatórios
def gerar_pontos_uniforme(n):
    return np.random.rand(n, 2)

def gerar_pontos_circulo(n, raio=1):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.array([(raio * np.cos(t), raio * np.sin(t)) for t in theta])

# Calcular a distância euclidiana entre dois pontos
def distancia_euclidiana(ponto1, ponto2):
    return np.linalg.norm(ponto1 - ponto2)
''
# Avaliação da aptidão: soma das distâncias entre os pontos em uma rota
def calcular_aptidao(caminho, pontos):
    distancia_total = 0
    for i in range(len(caminho) - 1):
        distancia_total += distancia_euclidiana(pontos[caminho[i]], pontos[caminho[i + 1]])
    # Retorno ao ponto inicial
    distancia_total += distancia_euclidiana(pontos[caminho[-1]], pontos[caminho[0]])
    return distancia_total

# Inicialização da população: gerar permutações aleatórias dos pontos
def inicializar_populacao(tamanho_populacao, n_pontos):
    return [np.random.permutation(n_pontos) for _ in range(tamanho_populacao)]

# Cruzamento: Order Crossover (OX)
def cruzamento(pai1, pai2):
    tamanho = len(pai1)
    inicio, fim = sorted(random.sample(range(tamanho), 2))
    
    filho = [-1] * tamanho
    filho[inicio:fim] = pai1[inicio:fim]
    
    pos = fim
    for gene in pai2:
        if gene not in filho:
            if pos >= tamanho:
                pos = 0
            filho[pos] = gene
            pos += 1
    return filho

# Mutação: troca dois genes de posição
def mutacao(caminho, taxa_mutacao):
    if random.random() < taxa_mutacao:
        i, j = random.sample(range(len(caminho)), 2)
        caminho[i], caminho[j] = caminho[j], caminho[i]
    return caminho

# Seleção por torneio
def selecao_torneio(populacao, aptidoes, k=3):
    selecionados = random.sample(list(zip(populacao, aptidoes)), k)
    selecionados.sort(key=lambda x: x[1])
    return selecionados[0][0]

# Algoritmo Genético
def algoritmo_genetico(pontos, tamanho_populacao=100, geracoes=500, taxa_mutacao=0.1):
    n_pontos = len(pontos)
    populacao = inicializar_populacao(tamanho_populacao, n_pontos)
    melhor_caminho = None
    melhor_aptidao = float('inf')

    historico_aptidao = []

    for geracao in range(geracoes):
        aptidoes = [calcular_aptidao(individuo, pontos) for individuo in populacao]
        
        # Atualizar o melhor caminho
        menor_aptidao = min(aptidoes)
        if menor_aptidao < melhor_aptidao:
            melhor_aptidao = menor_aptidao
            melhor_caminho = populacao[aptidoes.index(menor_aptidao)]
        
        # Mostrar desempenho a cada época
        historico_aptidao.append(melhor_aptidao)
        print(f"Geração {geracao+1}, Melhor aptidão: {melhor_aptidao}")

        # Nova geração
        nova_populacao = []
        for _ in range(tamanho_populacao // 2):
            pai1 = selecao_torneio(populacao, aptidoes)
            pai2 = selecao_torneio(populacao, aptidoes)
            filho1 = cruzamento(pai1, pai2)
            filho2 = cruzamento(pai2, pai1)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))

        populacao = nova_populacao

    return melhor_caminho, historico_aptidao

# Visualizar o resultado
def plotar_caminho(pontos, caminho, titulo="Caminho"):
    plt.figure()
    for i in range(len(caminho)):
        p1 = pontos[caminho[i]]
        p2 = pontos[caminho[(i+1) % len(caminho)]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo-')
    plt.scatter(pontos[:,0], pontos[:,1], c='red')
    plt.title(titulo)
    plt.show()

# Exemplo de uso
if _name_ == "_main_":
    # Gerar pontos uniformes
    pontos_uniformes = gerar_pontos_uniforme(10)
    melhor_caminho, historico = algoritmo_genetico(pontos_uniformes)

    # Plotar o melhor caminho encontrado
    plotar_caminho(pontos_uniformes, melhor_caminho, "Caminho - Pontos Uniformes")

    # Gerar pontos no círculo
    pontos_circulo = gerar_pontos_circulo(10)
    melhor_caminho, historico = algoritmo_genetico(pontos_circulo)

    # Plotar o melhor caminho encontrado
    plotar_caminho(pontos_circulo, melhor_caminho, "Caminho - Pontos no Círculo")