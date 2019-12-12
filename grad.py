import numpy as np

def errFunct(fG, T):
    t_steps = T.shape[0]
    diff = np.square(T[-1, :] - fG)
    err = np.sum(diff)
    return err

def indGrad(v, delta, i, fG, curr_err,dt,dx,t_f,L,k, f_type):
    v[i] = v[i] + delta
    x,T,r,s = FTCS(dt,dx,t_f,L,k,v, f_type)
    term_err = errFunct(fG, T)
    return (curr_err - term_err)/delta


def f(t, v, t_f, f_type="sin"):
    if f_type == "sin":
        res  = np.zeros(t.shape[0])
        for n, coef in enumerate(v, 1):
            res = res + coef*np.sin(np.pi*(n)/t_f*t)
        return res
    if f_type == "ints":
        res = np.zeros(t.shape[0])
        for i in range(t.shape[0]):
            res[i] = v[int(len(v)*i/t.shape[0])]
        return res
    if f_type == "expsin":
        from fit_data import func
        return func(t, *v)

def goalFunc(x, L):
    return np.sin(2*np.pi/L*x)


def FTCS(dt,dx,t_f,L,k,v, f_type):
    s = k*dt/(dx**2)
    if s > .5:
        print("WARNING: Unstable solution. Increase dx or decrease dt")
        return
    x = np.arange(0,L+dx,dx)
    t = np.arange(0,t_f+dt,dt)
    r = len(t)
    c = len(x)
    T = np.zeros([r,c])
    T[:,0] = f(t, v, t_f, f_type=f_type)
    for n in range(0,r-1): # Time
        for j in range(1,c-1): # Space
            T[n+1,j] = T[n,j] + s*(T[n,j-1] - 2*T[n,j] + T[n,j+1])
    return x,T,r,s

def grad(v, delta, dt, dx, t_f, L, k, pool, f_type):
    res = np.zeros(v.shape)
    x,T,r,s = FTCS(dt,dx,t_f,L,k,v, f_type)
    fG = goalFunc(x, L)
    curr_err = errFunct(fG, T)
    all_args = []
    for i in range(len(v)):
        all_args.append([v, delta, i, fG, curr_err, dt,dx,t_f,L,k, f_type])
    print("Error: ", curr_err)
    # for i in range(len(v)):

    #     res[i] = indGrad(v, delta, i, fG, curr_err)
    res = pool.starmap(indGrad, all_args)
    return np.asarray(res)