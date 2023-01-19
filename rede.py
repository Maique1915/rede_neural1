import numpy as np
from activationFunc import ActivationFunc

class Rede(ActivationFunc):
	"""docstring for Rede"""
	def __init__(self, *args):
		self.b = [np.array([[2*np.random.rand()-1] for y in range(args[i])]) for i in range(1, len(args)-1)]
		self.w = [np.array([[2*np.random.rand()-1 for x in range(args[i-1])] for y in range(args[i])]) for i in range(1, len(args)-1)]
		self.lr = args[-1]
		self.t = len(args)-1

	def rand(self, dataset, pc):
		train = []
		test = []
		for x in dataset:
			if (np.random.rand()>=pc):
				test.append(x)
			else:
				train.append(x)
		return train,test

	def __feedForward(self, a):
		s = []
		for w, b in zip(self.w, self.b):
			a = self.sigmoid(w@a+b)
			s.append(a)
		return s

	def __backPropagation(self, i, e, o):
		t = len(self.b)
		self.erro = (e - o[-1])**2
		d_err = 2*(e - o[-1])*self.lr
		for l in range(1, t):
			d = d_err*self.d_sigmoid(o[-l])
			self.b[-l] = self.b[-l] + d	
			self.w[-l] = self.w[-l] + d@o[-l-1].T
			d_err = self.w[-l].T @ d_err

		self.w[0] = self.w[0] + d_err*self.d_sigmoid(o[0])@i.T
		
	def train(self, i, e):
		i = np.array(i)
		e = self.sigmoid(np.array(e))
		sig = self.__feedForward(i)
		self.__backPropagation(i, e, sig)
		return np.sum(self.erro[-1].tolist())

	def predict(self, i):
		i = np.array(i)
		sig_o = self.__feedForward(i)
		return np.sum(self.i_sigmoid(sig_o[-1])/len(sig_o[-1])).tolist()

if __name__ == '__main__':
	rn = dc(3, 4, 1, 0.5)
	
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
		[[0], [0], [0], [0]],
		[[0], [0], [1], [0]],
		[[0], [1], [0], [1]],
		[[0], [1], [1], [0]],
		[[1], [0], [0], [1]],
		[[1], [0], [1], [0]],
		[[1], [1], [0], [1]],
		[[1], [1], [1], [0]]
	]
	
	
		
	train, test = rn.rand(dataset2, 0.8)
	i = np.random.randint(len(train))-1
	rn.train(train[i][:-1], train[i][-1])

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