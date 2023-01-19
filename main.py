import threading
import time, math
from graph import Graph
import random
from rede import Rede 
class Main(Graph):
	"""docstring for Main"""
	def __init__(self):
		super(Main, self).__init__()
		t = 21
		self.g = 20000
		self.nc = Rede(1, 128, 64, 32,  1, 0.02)
		self.entr = [x/math.pi for x in range(t)]
		self.said = [(math.cos(x)+1) for x in self.entr]
		self.erro = []
		self.btn.clicked.connect(self.run)
		self.init(0,self.g)
		self.run(i,e, c)

	def run(self):
		s = len(self.entr)
		r = [x for x in range(s)]
		inicio = time.time()
		for x in range(self.g):
			self.a = x
			a = self.nc.train([[self.entr[r[x%s]]]], [[self.said[r[x%s]]]])
			self.erro.append(a)
			if(x%s == 0):
				random.shuffle(r)

			fim = time.time()
			if(fim - inicio >= 0.5):
				inicio = fim
				self.draw()

		self.progressBar.setValue(100)
		self.out()

	def out(self):
		p = []
		for x in self.entr:
			p.append(self.nc.predict([[x]]))
		self.plot(p, self.erro)

if __name__ == '__main__':
	m = Main()
