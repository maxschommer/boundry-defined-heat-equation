import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def func(x, a, b, c, d, e):
	return a*np.sin(b*((c*x)**2+d)) + e


def main():

	fitfile = "optimizedV.npy"
	v = np.load(fitfile)
	t = np.asarray(range(len(v)))/len(v)


	popt, pcov = curve_fit(func, t, v, p0=[-8,1,2.5, .2, 4])

	print(popt)
	plt.figure()
	plt.plot(t, func(t, *popt), 'r-',
	         label='fit: a=%5.3f, b=%5.3f, c=%5.3f, d=%5.3f, e=%5.3f' % tuple(popt))
	plt.plot(t, v)
	plt.plot(t, func(t, -8,1,2.5, .2, 4))
	plt.show()
	# plt.figure()

	# t2 = np.arange(-2, 2, .01)
	# plt.plot(t2, func(t2, -15,10,3, -3.5, 0))
	plt.show()

if __name__ == '__main__':
	main()