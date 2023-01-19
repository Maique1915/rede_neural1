import numpy as np
from activationFunc import ActivationFunc

class NumbersClassifier(ActivationFunc):
	"""docstring for NumbersClassifier"""
	def __init__(self, i, h, o, lr):
		self.b_ih = np.array([[2*np.random.rand()-1] for y in range(h)])
		self.b_ho = np.array([[2*np.random.rand()-1] for y in range(o)])
		self.w_ih = np.array([[2*np.random.rand()-1 for x in range(i)] for y in range(h)])
		self.w_ho = np.array([[2*np.random.rand()-1 for x in range(h)] for y in range(o)])
		self.erro = [0]
		print(self.b_ih.shape, self.w_ih.shape)
		print(self.b_ho.shape, self.w_ho.shape)

		self.lr = lr

	def rand(self, dataset, pc):
		train = []
		test = []
		for x in dataset:
			if (np.random.rand()>=pc):
				test.append(x)
			else:
				train.append(x)
		return train,test


	def MSE(self, a, b):
		return np.array([np.mean((a-b)**2, axis=1)])

	def __feedForward(self, a):
		h = self.sigmoid(self.w_ih@a+self.b_ih)
		s = self.sigmoid(self.w_ho@h+self.b_ho)
		return h, s

	def __learn(self, b, w, x, err_x, layer_ant):
		d = err_x*self.d_sigmoid(x)*self.lr
		print(d.shape)
		v = (d@layer_ant.T)
		print(v.shape)
		return b + d, w + v

	def __backPropagation(self, i, e, sig_h, sig_o):
		err_o = e - sig_o
		self.erro = err_o
		print(self.w_ho.shape)
		self.b_ho, self.w_ho = self.__learn(self.b_ho, self.w_ho, sig_o, err_o, sig_h)
		print(self.w_ih.shape, err_o.shape)
		err_h = self.w_ho.T @ err_o
		self.b_ih, self.w_ih = self.__learn(self.b_ih, self.w_ih, sig_h, err_h, i)

	def train(self, i, e):
		i = np.array(i)
		e = np.array(e)
		sig_h, sig_o = self.__feedForward(i)
		self.__backPropagation(self.sigmoid(i), self.sigmoid(e), sig_h, sig_o)

	def predict(self, i):
		i = np.array(i)
		sig_h, sig_o = self.__feedForward(i)
		return np.sum(self.i_sigmoid(sig_o)).tolist()

	def erros(self):
		return np.mean(self.erro)



if __name__ == '__main__':
	rn = NumbersClassifier(3, 4, 1, 0.5)
	
	'''(~a * b) + (c + ~d) = xor'''
	dataset = [
		[[1], [1], [1], [1], [0]],
		[[1], [1], [1], [0], [1]],
		[[1], [1], [0], [1], [0]],
		[[1], [1], [0], [0], [0]],
		[[1], [0], [1], [1], [1]],
		[[1], [0], [1], [0], [1]],
		[[1], [0], [0], [1], [1]],
		[[1], [0], [0], [0], [1]],
		[[0], [1], [1], [1], [0]],
		[[0], [1], [1], [0], [1]],
		[[0], [1], [0], [1], [0]],
		[[0], [1], [0], [0], [0]],
		[[0], [0], [1], [1], [0]],
		[[0], [0], [1], [0], [1]],
		[[0], [0], [0], [1], [0]],
		[[0], [0], [0], [0], [0]]]
		
	dataset2 = [
		[[0], [0], [0],[0]],
		[[0], [0], [1],[0]],
		[[0], [1], [0],[1]],
		[[0], [1], [1],[0]],
		[[1], [0], [0],[1]],
		[[1], [0], [1],[0]],
		[[1], [1], [0],[1]],
		[[1], [1], [1],[0]],
	]
	
	
		
	#train, test = rn.rand(dataset2, 0.8)
	#i = np.random.randint(len(train))-1
	#rn.train(train[i][:-1], train[i][-1])

'''
	for y in range(100000):
		i = np.random.randint(len(train))-1
		rn.train(train[i][:-1], train[i][-1])
		if((y+1)%10000 == 0 and rn.predict(test[np.random.randint(len(test))-1][:-1])[0][0] > 0.9):
			print("antes:",y,"interações")
			break
		

	for x in dataset2:
		r = rn.predict(x[:-1])[0][0]
		if(r >= 0.8 and x[-1][0] == 1 or r < 0.8 and x[-1][0] == 0):
			print("acertou",x[-1][0], r)
		else:
			print("errou",x[-1][0], r)


		self.b_ih = np.array([[2*np.random.rand()-1] for y in range(h)])
		self.b_ho = np.array([[2*np.random.rand()-1] for y in range(o)])
		self.w_ih = np.array([[2*np.random.rand()-1 for x in range(i)] for y in range(h)])
		self.w_ho = np.array([[2*np.random.rand()-1 for x in range(h)] for y in range(o)])
'''