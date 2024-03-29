import numpy as np
import matplotlib.pyplot as plt
from grad import grad, FTCS, f, goalFunc
from multiprocessing import Pool

pool = Pool(4) 

dt = 0.0005
dx = 0.001
k = 10**(-3)
L = 0.05
t_f = 1

def gradDescent(v, delta, iters, step, dt, dx, t_f, L, k, f_type):
    for i in range(iters):
        grad_calc = grad(v, delta,  dt, dx, t_f, L, k, pool, f_type)
        v = v + step*grad_calc
        # print(grad_calc)
    return v

def runGradDescent(f_type):
    outfile = "optimizedV.npy"
    # v = np.random.rand(10)*2-1
    v = np.asarray([-10, -10, -10, -10, -10, -10, -10, -10, 10, -10])*.5

    v = np.load(outfile)
    np.save(".bak/"+outfile, v)
    # errFunct(goalFunc(x), T)
    # grad(v, .001, goalFunc(x), T)
    optimizedV = gradDescent(v, .001, 300, .2, dt, dx, t_f, L, k, f_type=f_type)
    np.save(outfile, optimizedV)
    # print("V: ", optimizedV)
    print("Saved output")

    x,T,r,s = FTCS(dt,dx,t_f,L,k,optimizedV, f_type)
    plt.figure(1)
    plt.plot(f(np.arange(0, 1, .01), optimizedV,  t_f, f_type=f_type))
    plt.figure(2)
    plt.plot(x,T[-1,:])
    plt.plot(x, goalFunc(x, L), "-")
    plt.show()

def runGradDescentExp():
    f_type = "sin"
    outfile = "optimizedSinV.npy"
    v = np.random.rand(10)*2-1
 #    v = np.asarray([-1.04536718e+01,  1.38486036e+00,  2.07509965e+00, -6.90751056e-03,
 # -2.97975491e+00])

    # v = np.load(outfile)
    np.save(".bak/"+outfile, v)
    # errFunct(goalFunc(x), T)
    # grad(v, .001, goalFunc(x), T)
    optimizedV = gradDescent(v, .0001, 20, .01, dt, dx, t_f, L, k, f_type=f_type)
    np.save(outfile, optimizedV)
    print("V: ", optimizedV)
    # print("Saved output")

    x,T,r,s = FTCS(dt,dx,t_f,L,k,optimizedV, f_type)
    t = np.arange(0, 1, .01)
    plt.figure(1)
    plt.plot(t, f(t, optimizedV,  t_f, f_type=f_type))
    plt.xlabel("t")
    plt.ylabel("T")
    plt.figure(2)
    plt.plot(x,T[-1,:])
    plt.plot(x, goalFunc(x, L), "-")
    plt.xlabel("x")
    plt.ylabel("T")
    plt.show()


def plotHeatEqn(f_type):
    v = np.random.rand(10)*2 - 1
    v = np.asarray([-10, -10, -10, -10, -10, -10, -10, -10, 10, 1])*.8
    v = np.load("optimizedV.npy")
    # v = [1,0,0,0,0,0,0,0,0]
    print(v)

    x,T,r,s = FTCS(dt,dx,t_f,L,k,v, f_type)

    t = np.arange(0, t_f, dt)
    plt.plot(t, f(t, v, t_f, f_type=f_type))
    plt.xlabel("t")
    plt.ylabel("T")
    plt.show()

    plot_times = np.arange(0.01,1.0,0.01)
    for t in plot_times:
        print(int(t/dt))
        plt.plot(x,T[int(t/dt),:])
        bottom, top = plt.ylim()  # return the current ylim
        plt.ylim((-2, 2))   # set the ylim to bottom, top
        # ylim(bottom, top)     # set the ylim to bottom, top
        plt.xlabel("x")
        plt.ylabel("T")
        plt.pause(.05)
        plt.clf()
    plt.plot(x,T[-1,:])

    # plt.plot(x, T[-1, :])
    plt.plot(x, goalFunc(x,L))

    plt.xlabel("x")
    plt.ylabel("T")
    plt.pause(.05)
    plt.show()

def main():
    # runGradDescent(f_type="ints")
    # runGradDescentExp()
    plotHeatEqn(f_type="ints")


if __name__ == '__main__':
    main()