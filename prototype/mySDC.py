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

# def get_phi_by_integral(z, n: int):
#     # hits "recursion limit" sometimes, eg (n=2, z=-0.002)
#     if n == 0: return np.exp(z)
#     return np.exp(z) * z**(-n) * gammainc(n, 0, z, regularized=True)

def get_phi_by_series(z, n: int):
    # Buvoli (32)
    if n == 0: return np.exp(z)
    
    res = np.zeros_like(z, dtype="complex128")
    P = 100
    
    if np.abs(z) > 0.8:
        res = get_phi_by_explicit_formula(z, n) 
    else:
        for k in range(P):
            res += 1./np.math.factorial(k+n) * z**k
    return res

def get_phi_by_contour_integration(z, n: int):
    # Buvoli ~(34), table 2
    if n == 0: return np.exp(z)
    
    res = np.zeros_like(z, dtype="complex128")
    P = 256
    R = 1.
    
    if np.abs(z) > 0.8:
        res = get_phi_by_explicit_formula(z, n) 
    else:
        for k in range(P):
            theta = 2 * np.pi * k / P * 1.j
            res += get_phi_by_explicit_formula(R * np.exp(theta) + z, n)
        res /= P
    return res

def get_phi_by_explicit_formula(z, n:int):
    match n:
        case 0:
            return np.exp(z)
        case 1:
            return (np.exp(z) - 1)/z
        case 2:
            return (np.exp(z) - 1 - z)/(z**2)
        case 3:
            return (2*np.exp(z) - 2 - 2*z - z**2)/(2*z**3)
        case 4:
            return (6*np.exp(z) - 6 - 6*z - 3*z**2 - z**3)/(6*z**4)
        case 5:
            return (24*np.exp(z) - 24 - 24*z - 12*z**2 - 4*z**3 - z**4)/(24*z**5)
        case 6:
            return (120*np.exp(z) - 120 - 120*z - 60*z**2 - 20*z**3 - 5*z**4 - z**5)/(120*z**6)
        case 7:
            return (720*np.exp(z) - 720 - 720*z - 360*z**2 - 120*z**3 - 30*z**4 - 6*z**5 - z**6)/(720*z**7)
        case 8:
            return (5040*np.exp(z) - 5040 - 5040*z - 2520*z**2 - 840*z**3 - 210*z**4 - 42*z**5 - 7*z**6 - z**7)/(5040*z**8)
        case 9:
            return (40320*np.exp(z) - 40320 - 40320*z - 20160*z**2 - 6720*z**3 - 1680*z**4 - 336*z**5 - 56*z**6 - 8*z**7 - z**8)/(40320*z**9)
        case 10:
            return (362880*np.exp(z) - 362880 - 362880.0*z**1 - 181440.0*z**2 - 60480.0*z**3 - 15120.0*z**4 - 3024.0*z**5 - 504.0*z**6 - 72.0*z**7 - 9.0*z**8 - 1.0*z**9)/(362880*z**10)
        case 11:
            return (3628800*np.exp(z) - 3628800 - 3628800.0*z**1 - 1814400.0*z**2 - 604800.0*z**3 - 151200.0*z**4 - 30240.0*z**5 - 5040.0*z**6 - 720.0*z**7 - 90.0*z**8 - 10.0*z**9 - 1.0*z**10)/(3628800*z**11)
        case 12:
            return (39916800*np.exp(z) - 39916800 - 39916800.0*z**1 - 19958400.0*z**2 - 6652800.0*z**3 - 1663200.0*z**4 - 332640.0*z**5 - 55440.0*z**6 - 7920.0*z**7 - 990.0*z**8 - 110.0*z**9 - 11.0*z**10 - 1.0*z**11)/(39916800*z**12)
        case 13:
            return (479001600*np.exp(z) - 479001600 - 479001600.0*z**1 - 239500800.0*z**2 - 79833600.0*z**3 - 19958400.0*z**4 - 3991680.0*z**5 - 665280.0*z**6 - 95040.0*z**7 - 11880.0*z**8 - 1320.0*z**9 - 132.0*z**10 - 12.0*z**11 - 1.0*z**12)/(479001600*z**13)
        case 14:
            return (6227020800*np.exp(z) - 6227020800 - 6227020800.0*z**1 - 3113510400.0*z**2 - 1037836800.0*z**3 - 259459200.0*z**4 - 51891840.0*z**5 - 8648640.0*z**6 - 1235520.0*z**7 - 154440.0*z**8 - 17160.0*z**9 - 1716.0*z**10 - 156.0*z**11 - 13.0*z**12 - 1.0*z**13)/(6227020800*z**14)
        case 15:
            (87178291200*np.exp(z) - 87178291200 - 87178291200.0*z**1 - 43589145600.0*z**2 - 14529715200.0*z**3 - 3632428800.0*z**4 - 726485760.0*z**5 - 121080960.0*z**6 - 17297280.0*z**7 - 2162160.0*z**8 - 240240.0*z**9 - 24024.0*z**10 - 2184.0*z**11 - 182.0*z**12 - 14.0*z**13 - 1.0*z**14)/(87178291200*z**15)
    raise Exception("explicit formula not implemented for n")


def initPhi(z, n: int):
    phi_res = np.zeros((n), dtype="complex128")
    for i in range(n):
        # switch to contour integration if needed
        phi_res[i] = get_phi_by_contour_integration(z, i)
    return phi_res

def weights(z, qi, m):
    # generates a_ij matrix
    n = np.shape(qi)[0]
    a = np.zeros((n, m+1)) # n, m+1
    for i in range(n):
        a[:, i] = get_fd_stencil(i, z, qi)
    return a

def generate_weights(N, tau_slice, dtau_slice, l):
    # Buvoli (29) generates w_ij
    w = np.zeros((N+1, N+2), dtype="complex128") 
    phi0 = np.array(())
    phi1 = np.array(())
    
    for i in range(1, N+2):
        # [φ0(hiΛ), ... , φN (hiΛ)]
        phi = initPhi(l * dtau_slice[i-1], N+3)
        
        phi0 = np.append(phi0, phi[0])
        phi1 = np.append(phi1, dtau_slice[i-1] * phi[1])
        q = (tau_slice - tau_slice[i-1]) / dtau_slice[i-1] 
        a = weights(0, q, N+1)
        
        # w[i][l] += a(i)φj+1(hiΛ)
        w[i-1][:] += dtau_slice[i-1] * np.dot(a, phi[1:])
            
    return w, phi0, phi1

#def check_lim(l, t):
#    # just for l = 0 case
#    if np.isclose([l], [0.0]):
#        return t
#    return (np.exp(l * t) - 1.) / l 

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
    # pre-generate weights, assuming timestep=Const
    weights_block, phi0, phi1 = generate_weights(N, tau[0:N+2], dtau[0:N+1], l)
    for j in range(steps-1): # timesteps
        left = j*(N+1)
        right = (j+1)*(N+1)+1
            
        tau_slice = tau[left:right]
        dtau_slice = dtau[0:N+1]
        
        # propagate initial solution
        for i in range(N+1):
            u_sdc[left+i+1] = phi0[i] * u_sdc[left+i] + phi1[i] * fn(u_sdc[left+i])
    
        for k in range(M): # sweeps
            u_sdc_slice = np.copy(u_sdc[left:right]) # save φk
        
            for i in range(N+1): # time substeps
                # 
                u_sdc[left+i+1] = phi0[i] * u_sdc[left+i] + phi1[i] * (fn(u_sdc[left+i]) - fn(u_sdc_slice[i])) + np.sum( weights_block[i] * fn( u_sdc_slice ))
                                        
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

def plot_phi(plot_name, method_names, method_solutions, shared_t):
    """
    Helper function: plots Re(u) and Im(u) on two subplots
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle(plot_name, fontsize=10)
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

def benchmark(solver, solver_args, analytical_solution, folds, verbose=False):
    """
    Helper function: dt, dt/2, dt/4 timing + analytical() error L_inf
    """
    N, M, t, u0, op_list = solver_args
    bench_t = t.copy()
    if verbose:
        print(M, "sweeps")
        print("    dt     |   max(E)   | cpu time [ms]")
        print("---------------------------------------")
    errors = np.zeros((folds))
    timesteps = np.zeros((folds))
    
    for fold in range(folds):
        timesteps[fold] = np.abs(bench_t[1] - bench_t[0])
        t_0 = time.time_ns()
        tau_solver, u_solver = solver(N, M, bench_t, u0, op_list)
        t_1 = time.time_ns()
        errors[fold] = np.max(np.abs(analytical_solution(tau_solver) - u_solver))
        if verbose:
            print("%.4e | %.4e | %.2f" % (np.abs(bench_t[1] - bench_t[0]), \
                                    np.max(np.abs(analytical_solution(tau_solver) - u_solver)), (t_1 - t_0) * 1e-6))
        
        bench_t /= 2
        bench_t = np.append(bench_t, (bench_t + bench_t[-1])[1:])
    return errors, timesteps

def plot_slopes(plot_name, errors, slopes, N, M, folds, tau, savepath, M_start=1):
    for i in range(M_start, M):
        plt.plot(tau, errors[i], label='m=%d, ñ=%.2f' % (i, slopes[i].real), linestyle="-.", marker="*")
        
    plt.title(plot_name)
    plt.grid()
    plt.legend(loc=2)
    plt.xlabel("dt")
    plt.ylabel("E(φ)")
    plt.xscale('log')
    plt.yscale('log')
    # turn off autoscaling and plot reference slopes
    plt.autoscale(False)
    
    for i in range(M_start, M):
        plt.plot([tau[0], tau[-1]], [errors[i][0], (errors[i][0] * (tau[-1] / tau[0])**(i+1))], color="0.5", alpha=0.5, linestyle="--")
    
    plt.savefig(savepath, dpi=600)
    plt.show()
    return