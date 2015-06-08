# -*- coding: utf-8 -*-
import math
import csv
import sys
import random
import time

#funcao de ativacao
def tanh(y):
	return math.tanh(y)

def sigmoid(x):
 	return 1.0 / (1.0 + math.exp(-x))

def dsigmoid(x):
	return x*(1.0 - x)

#derivada da tanh dado que y é tanh(y)
def dtanh(y):
	return 1.0 - y**2

def parser_csv(escolhaDataset): #treino
	
	entrada = []
	valor_esperado = []
	arquivo = ""
	if escolhaDataset == '1':
		arquivo = "train_diagnosis.csv" # https://archive.ics.uci.edu/ml/datasets/Acute+Inflammations
	elif escolhaDataset == '2':
		arquivo = "abalone.train" # https://archive.ics.uci.edu/ml/datasets/Abalone
	elif escolhaDataset == '3':
		arquivo = "train_data_KAHRAMAN.csv" # https://archive.ics.uci.edu/ml/datasets/User+Knowledge+Modeling
	elif escolhaDataset == '4':
		arquivo = "breast-cancer-wisconsin.train" # https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Original)
	elif escolhaDataset == '5':
		arquivo = ""
	f = open(arquivo, 'rt')
	try:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			if '?' not in row:
				if escolhaDataset == '1':
					entrada.append(row[1:-2]) #excluindo a 1a e ultima coluna
					valor_esperado.append(row[5])
				elif escolhaDataset == '2':
					entrada.append(row[:-1]) #todas as colunas
					valor_esperado.append(row[8])
				elif escolhaDataset == '3':
					entrada.append(row[:-1]) #todas as colunas
					valor_esperado.append(row[5])
				elif escolhaDataset == '4':
					entrada.append(row[1:-1]) #excluindo a 1a coluna
					valor_esperado.append(row[9])
				elif escolhaDataset == '5':
					entrada.append(row[1:-1]) #excluindo a 1a coluna
					valor_esperado.append(row[6])
				
	finally:
		f.close()
	return entrada , valor_esperado

def parser_csv2(escolhaDataset): #testes
	
	#f = open(sys.argv[2], 'rt')
	entrada = []
	valor_esperado = []
	arquivo = ""
	if escolhaDataset == '1':
		arquivo = "test_diagnosis.csv"
	elif escolhaDataset == '2':
		arquivo = "abalone.data"
	elif escolhaDataset == '3':
		arquivo = "test_data_KAHRAMAN.csv" 
	elif escolhaDataset == '4':
		arquivo = "breast-cancer-wisconsin.data"
	elif escolhaDataset == '5':
		arquivo = ""
	f = open(arquivo, 'rt')
	try:
		reader = csv.reader(f, delimiter=',')
		for row in reader:
			if '?' not in row:
				if escolhaDataset == '1':
					entrada.append(row[1:-2]) #excluindo a 1a e ultima coluna
					valor_esperado.append(row[5])
				elif escolhaDataset == '2':
					entrada.append(row[:-1]) #todas as colunas
					valor_esperado.append(row[8])
				elif escolhaDataset == '3':
					entrada.append(row[:-1]) #todas as colunas
					valor_esperado.append(row[5])
				elif escolhaDataset == '4':
					entrada.append(row[1:-1]) #excluindo a 1a coluna
					valor_esperado.append(row[9])
				elif escolhaDataset == '5':
					entrada.append(row[1:-1]) #excluindo a 1a coluna
					valor_esperado.append(row[6])
	finally:
		f.close()
	return entrada , valor_esperado	

def cria_matriz(i,j):
  m = []
  for k in range(i):
	m.append([0.0]*j)
  return m
  
def main():
	if len(sys.argv) == 2:
		escolhaDataset = sys.argv[1]
		entrada , valor_esperado = parser_csv(escolhaDataset)
		entrada2, valor_esperado2 = parser_csv2(escolhaDataset)
		if escolhaDataset == '1':
			neural_test = NeuralNetwork(5,5,1)
		elif escolhaDataset == '2':
			neural_test = NeuralNetwork(8,8,1)
		elif escolhaDataset == '3':
			neural_test = NeuralNetwork(5,4,1)
		elif escolhaDataset == '4':
			neural_test = NeuralNetwork(9,10,1)		
		
		print "Matriz de pesos da camada de entrada:"
		print neural_test.peso_entrada
		print "Matriz de pesos da camada escondida:"
		print neural_test.peso_escondido
		
		grupo_de_teste = []
		valor_esperado_teste = []
		for i in range(1,len(entrada2)):
		#for i in 0,20:
			grupo_de_teste.append(entrada2[i])
			valor_esperado_teste.append(float(valor_esperado2[i]))
		
		grupo_de_treinamento = []
		valor_esperado_treinamento = []
		for i in range(1,len(entrada)):
			grupo_de_treinamento.append(entrada[i])
			valor_esperado_treinamento.append(float(valor_esperado[i]))
		
		if escolhaDataset == '1':
			neural_test.treino(grupo_de_treinamento,valor_esperado_treinamento,100,0.01,0.005)
		elif escolhaDataset == '2':
			neural_test.treino(grupo_de_treinamento,valor_esperado_treinamento,100,0.01,0.005)
		elif escolhaDataset == '3':
			neural_test.treino(grupo_de_treinamento,valor_esperado_treinamento,20,0.1,0.1)
		elif escolhaDataset == '4':
			neural_test.treino(grupo_de_treinamento,valor_esperado_treinamento,100,0.01,0.05)
		
		
		neural_test.teste(grupo_de_teste,valor_esperado_teste,escolhaDataset)
	
		print "Matriz de pesos da camada de entrada final:"
		print neural_test.peso_entrada
		print "Matriz de pesos da camada escondida final:"
		print neural_test.peso_escondido
	else:
		print "Modo de uso: $ python main.py [DATASET]" 
		print ""
		print "Escolha o dataset:"
		print "1- Inflamacao da Bexiga [0:Nao, 1:Sim] - 100 epochs, N = 0.01, erro = 0.005"
		print "2- Crustaceo Abalone [Idade] - 100 epochs, N = 0.01, erro = 0.005"
		print "3- Conhecimento dos Estudantes [Baixo..Alto] - 20 epochs, N = 0.1, erro = 0.1"
		print "4- Cancer de Mama [2:Benigno, 4:Maligno] - 100 epochs, N = 0.01, erro = 0.05"
	
	
def gerar_pesos_aleatorios(matriz):
	#Wi,j -> peso do neuronio i para o neuronio j
	for i in range(len(matriz)):
		for j in range(len(matriz[0])):
			matriz[i][j] = round(random.uniform(-1,1), 2)
	return matriz
		
class NeuralNetwork:
	def __init__(self,nos_entrada,nos_escondidos,nos_saida):
		
		self.nos_entrada = nos_entrada + 1 
		self.nos_escondidos = nos_escondidos + 1
		self.nos_saida = nos_saida
		
		self.peso_entrada = cria_matriz(self.nos_entrada,self.nos_escondidos)
		self.peso_escondido = cria_matriz(self.nos_escondidos,self.nos_saida)
		
		self.matriz_y_escondida = [1.0]*self.nos_escondidos
		self.matriz_y_saida = [1.0]*self.nos_saida
		self.matriz_entrada = [1.0]*self.nos_entrada

		self.peso_entrada = gerar_pesos_aleatorios(self.peso_entrada)
		self.peso_escondido = gerar_pesos_aleatorios(self.peso_escondido)
		
	def calc_y(self,entrada):
		
		#print entrada
		for i in range(self.nos_entrada-1):			
			self.matriz_entrada[i] = entrada[i]

		#camada de entrada -> camada escondida
		for i in range(self.nos_escondidos-1):
			total = 0.0
			for j in range(self.nos_entrada-1):				
				total += float(entrada[j]) * self.peso_entrada[j][i]
				# print total
			self.matriz_y_escondida[i] = tanh(total)
		
		#camada escondida -> camada de saida
		for k in range(self.nos_saida):
			total = 0.0
			for j in range(self.nos_escondidos):
				total += self.matriz_y_escondida[j] * self.peso_escondido[j][k]
			self.matriz_y_saida = total

		return self.matriz_y_saida
	  
	def retroPropagacao(self,saida_esperada, N):
		
		#calculando erros (deltas) da saida
		delta_saida = [0.0]*self.nos_saida
		for i in range(self.nos_saida):
			delta_saida[i] = saida_esperada - self.matriz_y_saida
		
		#calculando erros (deltas) da camada escondida
		delta_escondido = [0.0]*self.nos_escondidos
		for i in range(self.nos_escondidos):
			erro = 0.0
			for j in range(self.nos_saida):
				erro += delta_saida[j] * self.peso_escondido[i][j]
			delta_escondido[i] = dtanh(self.matriz_y_escondida[i]) * erro

		#atualizando pesos da camada escondida
		for i in range(self.nos_escondidos):
			for k in range(self.nos_saida):
				self.peso_escondido[i][k] = self.peso_escondido[i][k] + N * delta_saida[k] * self.matriz_y_escondida[i]        	
			
		#atualizando pesos da camda de entrada
		for i in range(self.nos_entrada):
			for j in range(self.nos_escondidos):
				self.peso_entrada[i][j] = self.peso_entrada[i][j] + N * delta_escondido[j] * float(self.matriz_entrada[i])
				
		#calcular o erro geral
		erro = 0.0
		for i in range(self.nos_saida):
			erro += 0.5*((saida_esperada - self.matriz_y_saida)**2)
		
		return erro
			
	def treino(self,entrada,valor_esperado,epocas,N,erro_max):
		for i in range(0,epocas):
			erro = 0.0
			for j in range(0,len(entrada)):
				r = self.calc_y(entrada[j])
				tmp = self.retroPropagacao(valor_esperado[j],N)
				erro += tmp
			if erro < erro_max:
				print "Erro: %f" %erro
				break
	
	def teste(self,entrada,valor_esperado,escolhaDataset):
		erroSoma = 0
		for i in range(len(entrada)):
			y = self.calc_y(entrada[i])			
			if escolhaDataset == '1':
				if abs(valor_esperado[i] - y) < 0.1:
					erroSoma += 1
			elif escolhaDataset == '2':
				if abs(valor_esperado[i] - y) < 0.5:
					erroSoma += 1
			elif escolhaDataset == '3':
				if abs(valor_esperado[i] - y) < 0.5:
					erroSoma += 1
			elif escolhaDataset == '4':
				if abs(valor_esperado[i] - y) < 0.5:
					erroSoma += 1
			print "Valor esperado: %f || Valor obtido: %f" %(valor_esperado[i],y)
		taxaAcerto = float(erroSoma)/float(len(entrada))
		print "TAXA DE ACERTO FINAL: %.2f" %taxaAcerto
	  
if __name__ == '__main__':
	main()