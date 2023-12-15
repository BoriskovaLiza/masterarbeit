# SDC definitions and helper functions

import numpy as np
import matplotlib.pyplot as plt
import time


def Euler_SDC(N, M, t, u0, ops):
    """
    Euler-SDC: F(phi) = f(t, phi); N time substeps Chebyshev nodes; M sweeps
    """
    f = ops[0]
    
    steps = np.shape(t)[0]
    timestep = np.abs(t[1] - t[0])
    
    Ti = (N+1) * (steps-1) + 1
    tau = np.zeros((Ti))
    taui, _ = np.polynomial.chebyshev.chebgauss(N)
    
    for i in range(steps-1):
        tau[(N+1)*i] = t[i]
        tau[(N+1)*i+1:(N+1)*i+(N+1)] = taui[::-1] / 2 * timestep + timestep / 2 + t[i]
    tau[-1] = 1.
    
    dtau = tau[1:] - tau[:-1]
    
    uv_sdc = np.zeros((Ti), dtype=np.cfloat)
    uv_sdc[0] = u0
    
    # init chebychev series
    Order = N+2
    nodetonode = np.zeros((N+1))
    
    for j in range(steps-1): # timesteps
        left = j*(N+1)
        right = (j+1)*(N+1)+1
    
        # propagate initial solution
        for i in range(left+1,right):
            uv_sdc[i] = uv_sdc[i-1] + dtau[i-1] * f(tau[i-1], uv_sdc[i-1])
    
        tau_slice = tau[left:right]
        dtau_slice = dtau[0:N+1]
        integr = np.zeros((Order))
        series = np.polynomial.chebyshev.Chebyshev(np.zeros((Order)), \
             domain=[tau_slice[0], tau_slice[-1]], window=[tau_slice[0], tau_slice[-1]])
    
        for k in range(M): # sweeps
            uv_sdc_slice = np.copy(uv_sdc[left:right]) # save φk
            # fit a series
            series = series.fit(tau_slice, f(tau_slice, uv_sdc_slice), deg=Order)
            # integrate
            integseries = series.integ(m=1, lbnd=tau_slice[0])
            zerotonode = integseries(tau_slice)
            nodetonode = zerotonode[1:] - zerotonode[:-1]
        
            for i in range(N+1): # time substeps
                # φk+1i+1 =φk+1i +hiF(hτi+l,φk+1)−F(hτi+l,φk)+Ii+1(φk)
                uv_sdc[left+i+1] = uv_sdc[left+i] + dtau_slice[i] * \
                                   (f(tau_slice[i], uv_sdc[left+i]) - \
                                    f(tau_slice[i], uv_sdc_slice[i])) + nodetonode[i]
    return tau, uv_sdc

def IMEXSDC(N, M, t, u0, ops):
    """
    IMEX-SDC: F(phi) = l * phi + fn(phi); N time substeps Chebyshev nodes; M sweeps
    """
    l, fn = ops[0], ops[1]
    
    steps = np.shape(t)[0]
    timestep = np.abs(t[1] - t[0])
    
    Ti = (N+1) * (steps-1) + 1
    tau = np.zeros((Ti))
    taui, _ = np.polynomial.chebyshev.chebgauss(N)
    
    for i in range(steps-1):
        tau[(N+1)*i] = t[i]
        tau[(N+1)*i+1:(N+1)*i+(N+1)] = taui[::-1] / 2 * timestep + timestep / 2 + t[i]
    tau[-1] = 1.
    
    dtau = tau[1:] - tau[:-1]
    
    u_sdc = np.zeros((Ti), dtype=np.cfloat)
    u_sdc[0] = u0
    
    # init chebychev series
    Order = N+2
    nodetonode = np.zeros((N+1))
    
    for j in range(steps-1): # timesteps
        left = j*(N+1)
        right = (j+1)*(N+1)+1
    
        # propagate initial solution
        for i in range(left+1,right):
            u_sdc[i] = 1/(1 - dtau[i-1]*l) * (u_sdc[i-1] + dtau[i-1] * fn(u_sdc[i-1]))
    
        tau_slice = tau[left:right]
        dtau_slice = dtau[0:N+1]
        integr = np.zeros((Order))
        series = np.polynomial.chebyshev.Chebyshev(np.zeros((Order)), \
                 domain=[tau_slice[0], tau_slice[-1]], window=[tau_slice[0], tau_slice[-1]])
    
        for k in range(M): # sweeps
            # Interpolate Λφk (s) + N (s, φk (s))ds.
            u_sdc_slice = np.copy(u_sdc[left:right]) # save φk
            series = series.fit(tau_slice, l * u_sdc_slice + fn(u_sdc_slice), deg=Order)
        
            integseries = series.integ(m=1, lbnd=tau_slice[0])
            zerotonode = integseries(tau_slice)
            nodetonode = zerotonode[1:] - zerotonode[:-1]
        
            for i in range(N+1): # time substeps
                # φk+1i+1 =[I−hiΛ]^-1 ...
                u_sdc[left+i+1] = 1 / (1 - dtau_slice[i] * l) * \
                                    (u_sdc[left+i] - (dtau_slice[i] * l) * u_sdc_slice[i+1] + \
                                     dtau_slice[i] * (fn(u_sdc[left+i]) - fn(u_sdc_slice[i])) + \
                                     nodetonode[i])
    return tau, u_sdc

from fd_weights_explicit import get_fd_stencil
from mpmath import gammainc

def get_phi_by_integral(z, n: int):
    # hits "recursion limit" sometimes, eg (n=2, z=-0.002)
    if n == 0: return np.exp(z)
    return np.exp(z) * z**(-n) * gammainc(n, 0, z, regularized=True)

def get_phi_by_series(z, n: int):
    # Buvoli (32)
    if n == 0: return np.exp(z)
    
    res = np.zeros_like(z)
    for k in range(100):              
        res += 1./np.math.factorial(k+n) * z**k
    
    return res

def get_phi_by_recursion(z, n: int):
    # for low n
    if n == 0: return np.exp(z)
    return (get_phi_by_recursion(z, n-1) - 1/np.math.factorial(n-1)) / z

def get_phi_by_contour_integration(z, n: int):
    # Buvoli ~(34), table 2
    if n == 0: return np.exp(z)
    P = 32
    
    res = np.zeros_like(z, dtype="complex128")
    R = np.minimum(np.abs(z) / 4, np.ones_like(z) * 1e-7)
    for k in range(P):
        theta = 2 * np.pi * k / P
        res += get_phi_by_series(z + R * np.exp(0.j * theta), n)
        
    return res / P

def get_phi_by_recursive_contour_integration(z, n: int):
    # Buvoli (34)
    # doesnt really work
    if n == 0: return np.exp(z)
    P = 32
    
    res = np.zeros_like(z, dtype="complex128")
    R = 1e-7
    r = R / 2
    for k in range(P):
        theta = 2 * np.pi * k / P
        res += ( get_phi_by_recursive_contour_integration(z + r * np.exp(0.j * theta), n-1) - \
                1 / np.math.factorial(n-1) )/( z + R * np.exp(0.j * theta) )
        
    return res / P

def get_phi_by_explicit_formula(z, n:int):
    if n == 0:
        return np.exp(z)
    elif n == 1:
        return (np.exp(z)-1)/z
    elif n == 2:
        return (np.exp(z)-1-z)/(z**2)
    elif n == 3:
        return (2*np.exp(z)-2-2*z-z**2)/(2*z**3)
    elif n == 4:
        return (6*np.exp(z)-6-6*z-3*z**2-z**3)/(6*z**4)
    elif n == 5:
        return (24*np.exp(z) -24 - 24*z - 12*z**2 - 4*z**3 - z**4)/(24*z**5)
    raise Exception("explicit formula not implemented for n")


def initPhi(z, n: int, get_phi):
    phi_res = np.zeros((n+1), dtype="complex128")
    for i in range(n+1):
        phi_res[i] = get_phi(z, i)
    return phi_res

def weights(z, qi, m):
    # generates a_ij matrix
    n = np.shape(qi)[0]
    a = np.zeros((m, n))
    for i in range(m):
        a[i] = get_fd_stencil(i, z, qi)
    return a

def generate_weights(N, tau_slice, dtau_slice, l):
    # Buvoli (29) generates w_ij
    w = np.zeros((N+1, N+2), dtype="complex128")

    get_phi = get_phi_by_series
    if (np.abs(l * dtau_slice[0]) < 1.): 
        get_phi = get_phi_by_contour_integration
    
    for i in range(N+1):
        # [φ0(hiΛ), ... , φN (hiΛ)]
        phi = initPhi(l * dtau_slice[i], N+2, get_phi)
        q = (tau_slice - tau_slice[i]) / dtau_slice[i]
        a = weights(0, q, N+1)
        # w[i][l] += a(i)φj+1(hiΛ)
        for l in range(N+2):
            for j in range(N+1):
                w[i, l] += phi[j+1] * a[j, l]   
        w[i, :] *= dtau_slice[i]
    return w

def check_lim(l, t):
    # just for l = 0 case
    if np.isclose([l], [0.0]):
        return t
    return (np.exp(l * t) - 1.) / l

def ETDSDC(N, M, t, u0, ops):
    """
    ETD-SDC: F(phi) = l * phi + fn(phi); N time substeps Chebyshev nodes; M sweeps
    """
    l, fn = ops[0], ops[1]
    
    steps = np.shape(t)[0]
    timestep = np.abs(t[1] - t[0])
    
    Ti = (N+1) * (steps-1) + 1
    tau = np.zeros((Ti))
    taui, _ = np.polynomial.chebyshev.chebgauss(N)
    
    for i in range(steps-1):
        tau[(N+1)*i] = t[i]
        tau[(N+1)*i+1:(N+1)*i+(N+1)] = taui[::-1] / 2 * timestep + timestep / 2 + t[i]
    tau[-1] = t[steps-1]
    
    dtau = tau[1:] - tau[:-1]
    
    u_sdc = np.zeros((Ti), dtype=np.cfloat)
    u_sdc[0] = u0
    for j in range(steps-1): # timesteps
        left = j*(N+1)
        right = (j+1)*(N+1)+1
    
        # propagate initial solution
        for i in range(left+1,right):
            u_sdc[i] = u_sdc[i-1] * np.exp(l * dtau[i-1]) + \
                        check_lim(l, dtau[i-1]) * fn(u_sdc[i-1])
    
        tau_slice = tau[left:right]
        dtau_slice = dtau[0:N+1]
        weights_block = generate_weights(N, tau_slice, dtau_slice, l)
    
        for k in range(M): # sweeps
            u_sdc_slice = np.copy(u_sdc[left:right]) # save φk
            
            for i in range(N+1): # time substeps
                # φk+1i+1 =φk+1i ehiΛ + Λ^−1 [ehiΛ−1] (N(hτi,φk+1i) − N(hτi,φki))+Wi;i+1(φk)
                u_sdc[left+i+1] = u_sdc[left+i] * np.exp(l * dtau_slice[i]) + \
                                    check_lim(l, dtau_slice[i]) * \
                                    (fn(u_sdc[left+i]) - fn(u_sdc_slice[i])) + \
                                    np.sum( weights_block[i] * fn( u_sdc_slice ))
    
    return tau, u_sdc

def plot_functions(plot_name, function_names, functions, shared_x):
    """
    Helper function: plots Re(func(x)) and Im(func(x)) on two subplots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(plot_name, fontsize=10)
    fig.set_size_inches((10,4))

    for name, func in zip(function_names, functions):
        ax1.plot(shared_x, func(shared_x).real, label=name, linestyle="--")
        ax2.plot(shared_x, func(shared_x).imag, label=name, linestyle="--")
    
    ax1.legend()
    ax1.set_xlabel("x")
    ax1.set_ylabel("Re(f(x))")

    ax2.legend()
    ax2.set_xlabel("x")
    ax2.set_ylabel("Im(f(x))")
    plt.show()
    
    return

def plot_functions_imag(plot_name, function_names, functions, shared_x):
    """
    Helper function: plots Re(func(x)) and Im(func(x)) on two subplots
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2)
    fig.suptitle(plot_name, fontsize=10)
    fig.set_size_inches((10,8))

    for name, func in zip(function_names, functions):
        ax1.plot(shared_x.real, func(shared_x).real, label=name, linestyle="--")
        ax2.plot(shared_x.real, func(shared_x).imag, label=name, linestyle="--")
        ax3.plot(shared_x.imag, func(shared_x).real, label=name, linestyle="--")
        ax4.plot(shared_x.imag, func(shared_x).imag, label=name, linestyle="--")
    
    ax1.legend()
    ax1.set_xlabel("Re(x)")
    ax1.set_ylabel("Re(f(x))")

    ax2.legend()
    ax2.set_xlabel("Re(x)")
    ax2.set_ylabel("Im(f(x))")
    
    ax3.legend()
    ax3.set_xlabel("Im(x)")
    ax3.set_ylabel("Re(f(x))")
    
    ax4.legend()
    ax4.set_xlabel("Im(x)")
    ax4.set_ylabel("Im(f(x))")
    
    plt.show()
    
    return

def plot_solutions(plot_name, modes, method_names, method_solutions, shared_t):
    """
    Helper function: plots Re(u) and Im(u) on two subplots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    modes_string = ""
    for mode in modes:
        modes_string += ' %.2f %+.2fi;' % (mode.real, mode.imag)
    fig.suptitle('%s: Dahlquist w/ modes%s' % (plot_name, modes_string), fontsize=10)
    fig.set_size_inches((10,4))

    for name, solution in zip(method_names, method_solutions):
        ax1.plot(shared_t, solution.real, label=name, linestyle="--")
        ax2.plot(shared_t, solution.imag, label=name, linestyle="--")
    
    ax1.legend()
    ax1.set_xlabel("t")
    ax1.set_ylabel("Re(u(t))")

    ax2.legend()
    ax2.set_xlabel("t")
    ax2.set_ylabel("Im(u(t))")
    plt.show()
    
    return

def benchmark(solver, solver_args, analytical_solution, folds):
    """
    Helper function: dt, dt/2, dt/4 timing + analytical() error L_inf
    """
    N, M, t, u0, op_list = solver_args
    bench_t = t.copy()
    print(M, "sweeps")
    print("timestep   |   max(E)   | cpu time [ms]")
    print("----------------------------")
    errors = np.zeros((folds))
    timesteps = np.zeros((folds))
    
    for fold in range(folds):
        timesteps[fold] = np.abs(bench_t[1] - bench_t[0])
        t_0 = time.time_ns()
        tau_solver, u_solver = solver(N, M, bench_t, u0, op_list)
        t_1 = time.time_ns()
        errors[fold] = np.max(np.abs(analytical_solution(tau_solver) - u_solver))
        print("%.4e | %.4e | %.2f" % (np.abs(bench_t[1] - bench_t[0]), \
                                    np.max(np.abs(analytical_solution(tau_solver) - u_solver)), (t_1 - t_0) * 1e-6))
        
        bench_t /= 2
        bench_t = np.append(bench_t, (bench_t + bench_t[-1])[1:])
    # plot_solutions("Benchmarking", [], ["analytical", "last-bench"], [analytical(tau), u], tau)
    return errors, timesteps