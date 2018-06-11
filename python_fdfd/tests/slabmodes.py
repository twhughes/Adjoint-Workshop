import matplotlib.pyplot as plt
from scipy.optimize import fsolve, fminbound
from numpy import *

from ..core import assign_val
from ..eigen.beta_1d import calculate_modes as calculate_modes_1d
from ..eigen.beta_2d import calculate_modes as calculate_modes_2d

def test_slabmodes():

    # Define analytical transcendental equations
    def _F_TMe(omega,beta,d,epsr):
        betayd  = sqrt(omega**2/c_const**2*epsr-beta**2, dtype=complex)
        alphay0 = sqrt(beta**2-omega**2/c_const**2, dtype=complex)
        return abs(-1/epsr*betayd*d/2*tan(betayd*d/2)**-1-alphay0*d/2)

    def _F_TMo(omega,beta,d,epsr):
        betayd  = sqrt(omega**2/c_const**2*epsr-beta**2, dtype=complex)
        alphay0 = sqrt(beta**2-omega**2/c_const**2, dtype=complex)
        return abs(1/epsr*betayd*d/2*tan(betayd*d/2)-alphay0*d/2)

    def _F_TEe(omega,beta,d,epsr):
        betayd  = sqrt(omega**2/c_const**2*epsr-beta**2, dtype=complex)
        alphay0 = sqrt(beta**2-omega**2/c_const**2, dtype=complex)
        return abs(-betayd*d/2*tan(betayd*d/2)**-1-alphay0*d/2)

    def _F_TEo(omega,beta,d,epsr):
        betayd  = sqrt(omega**2/c_const**2*epsr-beta**2, dtype=complex)
        alphay0 = sqrt(beta**2-omega**2/c_const**2, dtype=complex)
        return abs(betayd*d/2*tan(betayd*d/2)-alphay0*d/2)

    # Specify geometry
    d = 1e-6
    epsr = 12.25

    omegas = 2*pi*linspace(1e13, 1e14, 50)

    n = sqrt(epsr)

    # Perform analytical zero solve

    betas_TMe = zeros(len(omegas))
    betas_TMo = zeros(len(omegas))
    betas_TEe = zeros(len(omegas))
    betas_TEo = zeros(len(omegas))

    for i in range(0, omegas.size):
        wrap_F_TMe = lambda x: _F_TMe(omegas[i], x, d, epsr)
        wrap_F_TMo = lambda x: _F_TMo(omegas[i], x, d, epsr)
        wrap_F_TEe = lambda x: _F_TEe(omegas[i], x, d, epsr)
        wrap_F_TEo = lambda x: _F_TEo(omegas[i], x, d, epsr)

        betas_TMe[i] = abs(fminbound(wrap_F_TMe, omegas[i]/c_const, omegas[i]/c_const*n)) #4
        betas_TEe[i] = abs(fminbound(wrap_F_TEe, betas_TMe[i], omegas[i]/c_const*n)) #3
        betas_TMo[i] = abs(fminbound(wrap_F_TMo, betas_TEe[i], omegas[i]/c_const*n)) #2
        betas_TEo[i] = abs(fminbound(wrap_F_TEo, betas_TMo[i], omegas[i]/c_const*n)) #1

    vals0_betas_TE = vstack([betas_TEe,betas_TEo]).transpose()
    vals0_betas_TM = vstack([betas_TMe,betas_TMo]).transpose()

    # 1D
    Neigs = 2
    N = 200
    xrange = (-10e-6, 10e-6)
    Npml = 15

    eps_space = ones(N, dtype=complex)
    rect1 = [0-d/2, 0+d/2]
    within_rect1 = lambda x: logical_and.reduce((x>rect1[0], x<rect1[1]))
    eps_space = assign_val(eps_space, within_rect1, epsr, xrange)

    beta_est = omegas/c_const*sqrt(epsr)
    (betas_te, _) = calculate_modes_1d('te', omegas, beta_est, Neigs, Npml, xrange, eps_space)
    (betas_tm, _) = calculate_modes_1d('tm', omegas, beta_est, Neigs, Npml, xrange, eps_space)

    # 2D
    Neigs = 4
    N = (200, 2)
    yrange = (0,2*diff(asarray(xrange))[0]/(N[0]+1))
    Npml = (Npml, 0)

    eps_space = ones(N, dtype=complex)
    rect2 = [0-d/2, 0+d/2]
    within_rect2 = lambda x, y: logical_and.reduce((x>rect2[0], x<rect2[1]))
    eps_space = assign_val(eps_space, within_rect2, epsr, xrange, yrange)

    beta_est = omegas/c_const*sqrt(epsr)
    (betas_2d, _, _, _, _, _, _) = calculate_modes_2d(omegas, beta_est, Neigs, xrange, yrange, eps_space, Npml)

    plt.figure(figsize=(7,7))

    plt.plot(real(vals0_betas_TE), omegas, 'ko')
    plt.plot(real(vals0_betas_TM), omegas, 'ks')

    plt.plot(real(betas_te), omegas, 'r-')
    plt.plot(real(betas_tm), omegas, 'r-')

    plt.plot(real(betas_2d), omegas, 'b--')

    betas_te.shape
    betas_te.shape

    betas_te.shape
    betas_te.shape

    betas_te.shape
