#!/usr/bin/env python

import tempfile
import subprocess

"""
	wrappy-svm	
	A wrapper for Octave which provides a Linear SVM classifier
	http://github.com/le1ca/wrappy-svm/
	
	Travis Mick, Sept. 2014
	<root@lo.calho.st>
	
"""

class _svm_qp_helper:
	""" _svm_qp_helper:
	    Helper class which interfaces with Octave to solve the quadratic
	    programming portion of the SVM training.
	"""

	def __init__(self, q, y):
		""" Constructor:
	    	Takes coefficient matrix q and class matrix y.
		"""
		self.q = q
		self.y = y
		
	def solve(self):
		""" solve:
			Calculates the optimal solution to the QP and returns its vector.
		"""
		self.temp = tempfile.NamedTemporaryFile()
		self._write()
		return self._run()
		
	def _write(self):
		""" _write:
		    Writes a temporary file containing the Octave script needed to compute
		    the QP solution.
		"""
		self.temp.write("Q = [")
		for i in range(len(self.q)):
			for j in range(len(self.q[i])):
				self.temp.write(str(self.q[i][j])+" ")
			if i < len(self.q) - 1:
				self.temp.write("; ")
		self.temp.write("];\n")
		self.temp.write("a = [")
		for i in range(len(self.q)):
			self.temp.write("-1 ")
		self.temp.write("]';\n")
		self.temp.write("y = [")
		for i in range(len(self.y)):
			self.temp.write(str(self.y[i])+" ")
		self.temp.write("];\n")
		self.temp.write("b = [")
		for i in range(len(self.y)):
			self.temp.write("0 ")
		self.temp.write("]';\n")
		self.temp.write("l = qp([], Q, a, y, 0, b, []);\n")
		self.temp.write("disp(l);\n")
		self.temp.flush()
		
	def _run(self):
		""" _run:
			Invokes Octave with the previously-written temporary file and parses
			its output into the list which is returned.
		"""
		sol = []
		p = subprocess.Popen(["octave", "-q", self.temp.name], stdout=subprocess.PIPE)
		for line in iter(p.stdout.readline,''):
			sol.append(float(line))
		self.temp.close()
		return sol
		

class wrappy_svm:
	""" wrappy_svm:
	    Provides a Linear SVM classifier by interacting with Octave. Takes a list
	    of tuples x and a list of their class values y upon construction. Creates
	    the SVM model when train() is invoked. Can determine the class of a new
	    tuple with the classify() method.
	"""

	def __init__(self, x, y):
		""" Constructor:
			Takes list of tuples x and list of their class values y. All tuples
			must be in the same dimension. All y values must be either 1 or -1.
			The number of y values must be the same as the number of x values.
		"""
		if len(x) != len(y):
			raise Exception("x and y must have same dimension")
		for i in range(1,len(x)):
			if len(x[1]) != len(x[i]):
				raise Exception("all x must have same dimension")
		for i in y:
			if i != 1 and i != -1:
				raise Exception("all y values must be 1 or -1")
		self.x = x
		self.y = y
		self.t = False
		
	def dot_prod(self, a, b):
		""" dot_prod:
			Returns the dot product of two vectors a and b.
		"""
		return sum(p*q for p,q in zip(a, b))
		
	def matrix(self):
		""" matrix:
			Returns the quadratic coefficient matrix used in the determination of
			the support vectors.
		"""
		m = []
		for i in range(len(self.x)):
			r = []
			for j in range(len(self.x)):
				r.append(self.y[i] * self.y[j] * self.dot_prod(self.x[i], self.x[j]))
			m.append(r)
		return m
	
	def weights(self):
		""" weights:
			Returns a vector containing the weights of the support vectors.
		"""
		if not self.t:
			raise Exception("svm not trained")
		w = []
		for i in range(len(self.l)):
			factor = self.l[i] * self.y[i]
			w.append([])
			for j in range(len(self.x[i])):
				w[i].append(factor * self.x[i][j])
		for j in range(len(w[i])):
			for i in range(1,len(w)):
				w[0][j] += w[i][j]
		return w[0]
		
	def bias(self):
		""" bias:
			Returns the bias of the dividing hyperplane.
		"""
		if not self.t:
			raise Exception("svm not trained")
		i = -1
		for j in range(len(self.l)):
			if self.l[j] > 0:
				i = j
				break
		return self.y[i] - self.dot_prod(self.w, self.x[i])
		
	def train(self):
		""" train:
			Builds the SVM model from the provided x and y values. You must train
			the instance in order to use it to classify.
		"""
		if self.t:
			raise Exception("svm already trained")
		h = _svm_qp_helper(self.matrix(), self.y)
		self.t = True
		self.l = h.solve()
		self.w = self.weights()
		self.b = self.bias()
	
	def hyperplane(self):
		""" hyperplane:
			Returns a textual representation of the equation of the dividing
			hyperplane in terms of a vector X.
		"""
		if not self.t:
			raise Exception("svm not trained")
		return str(self.w) + " * X + " + str(self.b) + " = 0"
		
	def classify(self, v):
		""" classify:
			Returns the predicted class value of the vector v.
		"""
		if not self.t:
			raise Exception("svm not trained")
		if len(v) != len(self.x[0]):
			raise Exception("incorrect vector dimension")
		return cmp(self.dot_prod(self.w, v) + self.b, 0)
		
svm = wrappy_svm

def wrappy_svm_test():
	""" wrappy_svm_test:
		Simple function to demonstrate basic functionality of wrappy_svm.
	"""

	x = [(0.3, 0.4), (0.4, 0.6), (0.9, 0.4), (0.7, 0.8)]
	y = [1, -1, -1, -1]
	
	mysvm = wrappy_svm(x, y)
	mysvm.train()
	
	print("hyperplane:")
	print(mysvm.hyperplane())
	print("")
	
	print("class of (0.5, 0.5):")
	print(mysvm.classify([0.5, 0.5]))
	print("")
	
	print("class of (0.2, 0.2):")
	print(mysvm.classify((0.2, 0.2)))
	print("")
	
if __name__ == '__main__':
	wrappy_svm_test()
