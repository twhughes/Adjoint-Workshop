from numpy import sqrt, array, prod, isscalar, zeros,  asarray, complex128, argsort, real, arange, exp, concatenate, conj
from scipy.sparse import spdiags, eye, kron, block_diag

from . import *
from . import dL
from .fdfd import createDws, solver_direct
from .pml import S_create


def _Nomegas(omegas):
    if isscalar(omegas):
        Nomegas = 1
        omegas = array([omegas])
    else:
        Nomegas = omegas.size
    return (omegas, Nomegas)


def solve_TM(omegas, xrange, yrange, eps_r, Jz, Npml, matrix_format='csc', timing=False):
    (omegas, Nomegas) = _Nomegas(omegas)

    N = asarray(eps_r.shape)  # Number of mesh cells
    M = prod(N)  # Number of unknowns

    vector_eps_z = epsilon0_const*eps_r.ravel(order='F')
    T_eps_z = spdiags(vector_eps_z, 0, M, M, format=matrix_format)

    jz = Jz.ravel(order='F')

    Hx = zeros((Nomegas, N[0], N[1]), dtype=complex128)
    Hy = zeros((Nomegas, N[0], N[1]), dtype=complex128)
    Ez = zeros((Nomegas, N[0], N[1]), dtype=complex128)

    for iw in range(0, Nomegas):
        (Sxf, Sxb, Syf, Syb) = S_create(omegas[iw], N, Npml, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A = (Dxf*1/mu0_const).dot(Dxb) \
            + (Dyf*1/mu0_const).dot(Dyb) \
            + omegas[iw]**2*T_eps_z
        b = 1j*omegas[iw]*jz

        if not b.any():  # If source is zero
            ez = zeros(b.shape)
        else:
            ez = solver_direct(A, b, timing=timing)

        hx = -1/1j/omegas[iw]/mu0_const * Dyb.dot(ez)
        hy = 1/1j/omegas[iw]/mu0_const * Dxb.dot(ez)

        Hx[iw, :, :] = hx.reshape((N[0], N[1]), order='F')
        Hy[iw, :, :] = hy.reshape((N[0], N[1]), order='F')
        Ez[iw, :, :] = ez.reshape((N[0], N[1]), order='F')

    return (Ez, Hx, Hy)


def solve_TE(omegas, xrange, yrange, eps_r, Mz, Npml, matrix_format='csc', timing=False):
    (omegas, Nomegas) = _Nomegas(omegas)

    N = asarray(eps_r.shape)  # Number of mesh cells
    M = prod(N)  # Number of unknowns

    vector_eps_x = grid_average(epsilon0_const*eps_r, 'x').ravel(order='F')
    vector_eps_y = grid_average(epsilon0_const*eps_r, 'y').ravel(order='F')

    # Setup the T_eps_x, T_eps_y, T_eps_x_inv, and T_eps_y_inv matrices
    T_eps_x = spdiags(vector_eps_x, 0, M, M, format=matrix_format)
    T_eps_y = spdiags(vector_eps_y, 0, M, M, format=matrix_format)
    T_eps_x_inv = spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
    T_eps_y_inv = spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)

    mz = Mz.ravel(order='F')

    Ex = zeros((Nomegas, N[0], N[1]), dtype=complex128)
    Ey = zeros((Nomegas, N[0], N[1]), dtype=complex128)
    Hz = zeros((Nomegas, N[0], N[1]), dtype=complex128)

    for iw in range(0, Nomegas):
        (Sxf, Sxb, Syf, Syb) = S_create(omegas[iw], N, Npml, xrange, yrange, matrix_format=matrix_format)

        # Construct derivate matrices
        Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
        Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

        A = Dxf.dot(T_eps_x_inv).dot(Dxb) \
            + Dyf.dot(T_eps_y_inv).dot(Dyb) \
            + omegas[iw]**2*mu0_const*eye(M)

        b = 1j*omegas[iw]*mz

        if not b.any():  # If source is zero
            hz = zeros(b.shape)
        else:
            hz = solver_direct(A, b, timing=timing)

        ex = 1/1j/omegas[iw]*T_eps_y_inv.dot(Dyb).dot(hz)
        ey = 1/1j/omegas[iw]*T_eps_y_inv.dot(-Dxb).dot(hz)

        Ex[iw, :, :] = ex.reshape((N[0], N[1]), order='F')
        Ey[iw, :, :] = ey.reshape((N[0], N[1]), order='F')
        Hz[iw, :, :] = hz.reshape((N[0], N[1]), order='F')

    return (Hz, Ex, Ey)

