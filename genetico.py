import numpy as np
import random
import matplotlib.pyplot as plt

# Classe para representar um indivíduo
class Individuo:
    def __init__(self, caminho):
        self.caminho = caminho  # o gene
        self.aptidao = None     

    def calcular_aptidao(self, pontos):
        self.aptidao = calcular_aptidao(self.caminho, pontos)
        return self.aptidao

# Gerar pontos aleatórios
def gerar_pontos_uniforme(n):
    return np.random.rand(n, 2)

def gerar_pontos_circulo(n, raio=1):
    theta = np.linspace(0, 2 * np.pi, n, endpoint=False)
    return np.array([(raio * np.cos(t), raio * np.sin(t)) for t in theta])

# Calcular a distância euclidiana entre dois pontos
def distancia_euclidiana(ponto1, ponto2):
    return np.linalg.norm(ponto1 - ponto2)

#soma das distâncias entre os pontos em uma rota
def calcular_aptidao(caminho, pontos):
    distancia_total = 0
    for i in range(len(caminho) - 1):
        distancia_total += distancia_euclidiana(pontos[caminho[i]], pontos[caminho[i + 1]])
    # Retorno ao ponto inicial
    distancia_total += distancia_euclidiana(pontos[caminho[-1]], pontos[caminho[0]])
    return distancia_total


def inicializar_populacao(tamanho_populacao, n_pontos):
    return [Individuo(np.random.permutation(n_pontos)) for _ in range(tamanho_populacao)]

# Cruzamento Order Crossover (OX)
def cruzamento(pai1, pai2):
    tamanho = len(pai1.caminho)
    inicio, fim = sorted(random.sample(range(tamanho), 2))

    filho = [-1] * tamanho
    filho[inicio:fim] = pai1.caminho[inicio:fim]

    pos = fim
    for gene in pai2.caminho:
        if gene not in filho:
            if pos >= tamanho:
                pos = 0
            filho[pos] = gene
            pos += 1
    return Individuo(filho)

# Mutação
def mutacao(individuo, taxa_mutacao):
    if random.random() < taxa_mutacao:
        i, j = random.sample(range(len(individuo.caminho)), 2)
        individuo.caminho[i], individuo.caminho[j] = individuo.caminho[j], individuo.caminho[i]
    return individuo

# Seleção por torneio
def selecao_torneio(populacao, k=3):
    selecionados = random.sample(populacao, k)
    selecionados.sort(key=lambda x: x.aptidao)
    return selecionados[0]

# Algoritmo Genético
def algoritmo_genetico(pontos, tamanho_populacao=1000, geracoes=1000, taxa_mutacao=0.2, paciencia=200, intervalo_plot=200):
                        #para 20 ponto: pop=1000       ger=1000        mut=0.2   
                        #para 30 pontos: pop=4000      ger=1000        mut=0.2 quase resolve, aumentar a pop ajuda e muito        
    n_pontos = len(pontos)
    populacao = inicializar_populacao(tamanho_populacao, n_pontos)
    
    # Calcular a aptidão inicial
    for individuo in populacao:
        individuo.calcular_aptidao(pontos)

    melhor_caminho = None
    melhor_aptidao = float('inf')
    sem_melhora = 0

    historico_aptidao = []

    for geracao in range(geracoes):
        # Atualizar o melhor caminho
        aptidoes = [individuo.calcular_aptidao(pontos) for individuo in populacao]
        menor_aptidao = min(aptidoes)

        if menor_aptidao < melhor_aptidao:
            melhor_aptidao = menor_aptidao
            melhor_caminho = populacao[aptidoes.index(menor_aptidao)]
            sem_melhora = 0 
        else:
            sem_melhora += 1

       
        historico_aptidao.append(melhor_aptidao)
        print(f"Geração {geracao + 1}, Melhor aptidão: {melhor_aptidao}")

        
        if (geracao + 1) % intervalo_plot == 0:
            plotar_caminho(pontos, melhor_caminho.caminho, f"Caminho - Geração {geracao + 1}")

    
        if sem_melhora >= paciencia:
            print("Critério de parada atingido.")
            break

        nova_populacao = []
        for _ in range(tamanho_populacao // 2):
            pai1 = selecao_torneio(populacao)
            pai2 = selecao_torneio(populacao)
            filho1 = cruzamento(pai1, pai2)
            filho2 = cruzamento(pai2, pai1)
            nova_populacao.append(mutacao(filho1, taxa_mutacao))
            nova_populacao.append(mutacao(filho2, taxa_mutacao))

        populacao = nova_populacao

    return melhor_caminho.caminho, historico_aptidao


def plotar_caminho(pontos, caminho, titulo="Caminho"):
    plt.figure()
    for i in range(len(caminho)):
        p1 = pontos[caminho[i]]
        p2 = pontos[caminho[(i + 1) % len(caminho)]]
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'bo-')
    plt.scatter(pontos[:, 0], pontos[:, 1], c='red')
    plt.title(titulo)
    plt.show()


if __name__ == "__main__":
   
    pontos_uniformes = gerar_pontos_uniforme(20)
    melhor_caminho, historico = algoritmo_genetico(pontos_uniformes)

   
    plotar_caminho(pontos_uniformes, melhor_caminho, "Caminho - Pontos Uniformes")
    
   
    input("Pressione Enter para continuar e gerar os pontos em círculo...")

    
    pontos_circulo = gerar_pontos_circulo(20)
    melhor_caminho, historico = algoritmo_genetico(pontos_circulo)


    plotar_caminho(pontos_circulo, melhor_caminho, "Caminho - Pontos no Círculo")
