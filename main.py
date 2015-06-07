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

def sigmoid(x):
  return 1 / (1 + math.exp(-x))
  
def cria_matriz(i,j):
  m = []
  for k in range(i):
	m.append([0.0]*j)
  return m
  
def main():
	entrada , valor_esperado = parser_csv()
	neural_test = NeuralNetwork(12,9,1)
	
	
	
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
		for i in range(self.nos_escondidos):
			for j in range(self.nos_entrada):


		
	  
	def retroPropagacao(self,saida_esperada, N):
		
		
			
	def treino(self,entrada,valor_esperado,epocas,N,erro_max):
		
	
	def teste(self,entrada,valor_esperado):
		
	  
if __name__ == '__main__':
	main()