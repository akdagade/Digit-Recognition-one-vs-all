import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import os 
seed = 	17

def display(img):
    image = img.reshape(28,28)
    plt.axis('off')
    plt.imshow(image, cmap=cm.binary)
    plt.show()


def sigmoid(g):
	return (1 / (1 + np.exp(-g)))


def train_model(x, alpha, itr, m, n, theta, X, y):
	y = (y==x).astype(int)
	j = 0 
	j_history = []
	min_j = float("inf")

	for i in range(0,itr):
		
		g = X.dot(theta)
		h = sigmoid(g)
		
		j = (-1/m) * np.sum((y * np.log(h)) + ((1-y) * np.log(1-h))) # 42000 x 1  # m x 1
		gradiant = (1/m) * np.sum((h-y) * X, axis=0) 
		gradiant = gradiant.reshape(n,1) # 784 X 1 
		theta = theta - (alpha*gradiant)
		print("Itr >> " + str(i), end = '\r')
		if min_j > j:
			min_j=j
			min_theta = theta

		if i % 1000 == 0:
			print(">> Cost : " + str(j) + " <<")
	print(">> Minimum Cost : " + str(min_j) + " <<")
	return min_theta


def predict(theta, X):
	g = X.dot(theta)
	return sigmoid(g)


def main():
		#np.set_printoptions(suppress=False)
		#np.set_printoptions(threshold=np.nan)
		##### Setup path variales
		os.chdir("/home/akshay/tfenv/Projects/digit_recognizer")
		path = os.getcwd() + '/all/'
		
		#initialize training data
		train = pd.read_csv(path + 'train.csv') #42000 x 785 
		m = train.shape[0]
		train_X = train.iloc[:,1:].values
		train_X = train_X * (1/255) #42000 x 784
		train_y = train.iloc[:,0].values.reshape(m,1) #42000 x 1

		#initialize testing data
		test = pd.read_csv(path + 'test.csv') #28000 x 784 
		test_X = test.values * (1/255) #28000 x 784
		test_y = np.zeros((28000,1)) #28000 x 1

		#initialize training parameters
		np.random.seed(seed)
		alpha = 1
		itr = 2001
		n = train_X.shape[1]
		theta = []
		for x in range(0,10):
			theta.append(np.random.randn(n,1))

		#train loop for each number
		for x in range(0,10):
			print("\n\n### Starting training for number : " + str(x) + " ###\n")
			theta[x] = train_model(x, alpha, itr, m, n, theta[x], train_X, train_y)
		
		#prediction loop
		m_test = test_X.shape[0]
		prediction = np.zeros((m_test,10))
		for x in range(0,10):
			prediction[:,x] = predict(theta[x], test_X).reshape(m_test)

		final = np.argmax(prediction,axis=1)
		np.savetxt("predictions.csv",final)

		for x in range(0,15):
			display(test_X[x])


if __name__ == '__main__':
	main()