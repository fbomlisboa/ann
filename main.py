# -*- coding: utf-8 -*-
import math
import csv
import sys
import random

#funcao de ativacao
def tanh(y):
	return math.tanh(y)

#derivada da tanh dado que y é tanh(y)
def dtanh(y):
	return 1 - y**2

def parser_csv():
	
	f = open(sys.argv[1], 'rt')
	entrada = []
	valor_esperado = []
	try:
		reader = csv.reader(f)
		for row in reader:
			if row[2] == 'jan': row[2] = 1
			if row[2] == 'feb': row[2] = 2
			if row[2] == 'mar': row[2] = 3
			if row[2] == 'apr': row[2] = 4
			if row[2] == 'may': row[2] = 5
			if row[2] == 'jun': row[2] = 6
			if row[2] == 'jul': row[2] = 7
			if row[2] == 'aug': row[2] = 8
			if row[2] == 'sep': row[2] = 9
			if row[2] == 'oct': row[2] = 10
			if row[2] == 'nov': row[2] = 11
			if row[2] == 'dec': row[2] = 12
			if row[3] == 'mon': row[3] = 1
			if row[3] == 'tue': row[3] = 2
			if row[3] == 'wed': row[3] = 3
			if row[3] == 'thu': row[3] = 4
			if row[3] == 'fri': row[3] = 5
			if row[3] == 'sat': row[3] = 6
			if row[3] == 'sun': row[3] = 7
			entrada.append(row[:-1])
			valor_esperado.append(row[12])
	finally:
		f.close()
	return entrada , valor_esperado

def cria_matriz(i,j):
  m = []
  for k in range(i):
	m.append([0.0]*j)
  return m
  
def main():
	entrada , valor_esperado = parser_csv()
	neural_test = NeuralNetwork(12,9,1)
	
	print "Matriz de pesos da camada de entrada:"
	print neural_test.peso_entrada
	print "Matriz de pesos da camada escondida:"
	print neural_test.peso_escondido
	
	grupo_de_teste = []
	valor_esperado_teste = []
	for i in range(100,200):
		grupo_de_teste.append(entrada[i])
		valor_esperado_teste.append(float(valor_esperado[i]))
	
	grupo_de_treinamento = []
	valor_esperado_treinamento = []
	for i in range(201,251):
		grupo_de_treinamento.append(entrada[i])
		valor_esperado_treinamento.append(float(valor_esperado[i]))
	
	neural_test.treino(grupo_de_treinamento,valor_esperado_treinamento,1000,0.5,0.005)
	neural_test.teste(grupo_de_teste,valor_esperado_teste)
	
	print "Matriz de pesos da camada de entrada final:"
	print neural_test.peso_entrada
	print "Matriz de pesos da camada escondida final:"
	print neural_test.peso_escondido    
	
	
def gerar_pesos_aleatorios(matriz):
	#Wi,j -> peso do neuronio i para o neuronio j
	for i in range(len(matriz)):
		for j in range(len(matriz[0])):
			matriz[i][j] = round(random.uniform(-1,1), 2)
	return matriz
		
class NeuralNetwork:
	def __init__(self,nos_entrada,nos_escondidos,nos_saida):
		
		self.nos_entrada = nos_entrada
		self.nos_escondidos = nos_escondidos
		self.nos_saida = nos_saida
		
		self.peso_entrada = cria_matriz(self.nos_entrada,self.nos_escondidos)
		self.peso_escondido = cria_matriz(self.nos_escondidos,self.nos_saida)
		
		self.matriz_y_escondida = [1.0]*self.nos_escondidos
		self.matriz_y_saida = [1.0]*self.nos_saida
		
		self.peso_entrada = gerar_pesos_aleatorios(self.peso_entrada)
		self.peso_escondido = gerar_pesos_aleatorios(self.peso_escondido)
		
	def calc_y(self,entrada):
		
		self.matriz_entrada = entrada
		#camada de entrada -> camada escondida
		for i in range(self.nos_escondidos):
			total = 0.0
			for j in range(self.nos_entrada):
				total += float(entrada[i]) * self.peso_entrada[j][i]
			self.matriz_y_escondida[i] = tanh(total)
		
		#camada escondida -> camada de saida
		for k in range(self.nos_saida):
			total = 0.0
			for j in range(self.nos_escondidos):
				total += self.matriz_y_escondida[k] * self.peso_escondido[j][k]
			self.matriz_y_saida = total
			
		return self.matriz_y_saida
	  
	def retroPropagacao(self,saida_esperada, N):
		
		#calculando erros (deltas) da saida
		delta_saida = [0.0]*self.nos_saida
		for i in range(self.nos_saida):
			delta_saida[i] = saida_esperada - self.matriz_y_saida
		
		#atualizando pesos da camada escondida
		for i in range(self.nos_escondidos):
			for k in range(self.nos_saida):
				self.peso_escondido[i][k] = self.peso_escondido[i][k] + N * delta_saida[k] * self.matriz_y_escondida[i]        	
				
		#calculando erros (deltas) da camada escondida
		delta_escondido = [0.0]*self.nos_escondidos
		for i in range(self.nos_escondidos):
			erro = 0.0
			for j in range(self.nos_saida):
				erro += delta_saida[j] * self.peso_escondido[i][j]
			delta_escondido[i] = dtanh(self.matriz_y_escondida[i]) * erro
			
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
		erro = 0.0
		for i in range(0,epocas):
			for j in range(0,len(entrada)):
				r = self.calc_y(entrada[j])
				tmp = self.retroPropagacao(valor_esperado[j],N)
				print "Erro: %f da iteração %d da época %d" %(tmp,j,i)

				erro += tmp
			if erro < erro_max:
				print "Erro: %f" %erro
				break
	
	def teste(self,entrada,valor_esperado):
		for i in range(len(entrada)):
			y = self.calc_y(entrada[i])
			print "Valor esperado: %f || Valor obtido: %f" %(valor_esperado[i],y)
	  
if __name__ == '__main__':
	main()