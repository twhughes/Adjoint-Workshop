from numpy import sqrt, array, prod, isscalar, zeros,  asarray, complex128, argsort, real
from scipy.sparse import spdiags, vstack, hstack, block_diag

from . import *
from . import grid_average
from .fdfd import createDws, solver_eigs
from .pml import S_create


def assemble_matrix_1D(pol, omegas, N, Npml, xrange, T_eps, T_eps_x_inv, matrix_format='csc'):

    (Sxf, Sxb, Syf, Syb) = S_create(omegas[0], N, Npml, xrange, matrix_format=matrix_format)

    # Construct derivate matrices
    Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange), N, matrix_format=matrix_format))
    Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange), N, matrix_format=matrix_format))

    # Construct A matrix
    if pol is 'tm':
        A1 = mu0_const*T_eps
        A2 = Dxf.dot(Dxb)

    if pol is 'te':
        A1 = mu0_const*T_eps
        A2 = T_eps.dot(Dxf).dot(T_eps_x_inv).dot(Dxb)

    return (A1, A2, Dxf, Dxb)


def calculate_beta_1D(pol, omegas, beta_est, Neigs, Npml, xrange, eps_r, matrix_format='csc',
                    eps_region_funcs=None, eps_dispersion_funcs=None):
    if (eps_region_funcs is not None) and (eps_dispersion_funcs is not None):
        flag_dispersive = True
        assert(len(eps_region_funcs) == len(eps_dispersion_funcs))
        N_dispersive_regions = len(eps_region_funcs)
    else:
        flag_dispersive = False

    N = asarray(eps_r.shape)  # Number of mesh cells
    M = prod(N)  # Number of unknowns

    if isscalar(omegas):
        Nomegas = 1
        omegas = array([omegas])
    else:
        Nomegas = omegas.size

    if len(N) > 1:
        assert(N[1] > 1)

    # If eps is not dispersive we can set these outside of omega loop
    if not flag_dispersive:
        eps_r_x = grid_average(epsilon0_const*eps_r, 'x')
        vector_eps_r_x = eps_r_x.ravel(order='F')
        vector_eps_r = epsilon0_const*eps_r.ravel(order='F')
        # Setup the T_eps_x, T_eps_y, and T_eps_z matrices
        T_eps = spdiags(vector_eps_r, 0, M, M, format=matrix_format)
        T_eps_x_inv = spdiags(1/vector_eps_r_x, 0, M, M, format=matrix_format)
        (A1, A2, Dxf, Dxb) = assemble_matrix_1D(pol, omegas, N, Npml, xrange, T_eps, T_eps_x_inv, matrix_format=matrix_format)

    betas = zeros((Nomegas, Neigs), dtype=complex128)
    if pol is 'tm':
        Ey = zeros((Nomegas, Neigs, N[0]), dtype=complex128)

    if pol is 'te':
        Hy = zeros((Nomegas, Neigs, N[0]), dtype=complex128)

    for iw in range(0, Nomegas):
        if flag_dispersive:
            for ir in range(0, N_dispersive_regions):
                eps_r = assign_val(eps_r, eps_region_funcs[ir], eps_dispersion_funcs[ir](omegas[iw]), xrange)

        eps_r_x = grid_average(epsilon0_const*eps_r, 'x')
        vector_eps_r_x = eps_r_x.ravel(order='F')
        vector_eps_r = epsilon0_const*eps_r.ravel(order='F')
        # Setup the T_eps_x, T_eps_y, and T_eps_z matrices
        T_eps = spdiags(vector_eps_r, 0, M, M, format=matrix_format)
        T_eps_x_inv = spdiags(1/vector_eps_r_x, 0, M, M, format=matrix_format)

        (A1, A2, Dxf, Dxb) = assemble_matrix_1D(pol, omegas, N, Npml, xrange, T_eps, T_eps_x_inv, matrix_format=matrix_format)

        A = omegas[iw]**2*A1+A2
        if isscalar(beta_est):
            (values, vectors) = solver_eigs(A, Neigs, guess_value=beta_est**2)
        else:
            (values, vectors) = solver_eigs(A, Neigs, guess_value=beta_est[iw]**2)

        inds_sorted = argsort(real(sqrt(values)))[::-1]
        betas[iw, :] = sqrt(values[inds_sorted])

        for i in range(0, Neigs):
            if pol is 'tm':
                Ey[iw, i, :] = vectors[:, inds_sorted[i]]

            if pol is 'te':
                Hy[iw, i, :] = vectors[:, inds_sorted[i]]

    if pol is 'tm':
        return (betas, Ey)

    if pol is 'te':
        return (betas, Hy)

def assemble_matrix_2D(omegas, N, Npml, xrange, yrange,
                    T_eps_x, T_eps_y, T_eps_z_inv, matrix_format='csc'):

    (Sxf, Sxb, Syf, Syb) = S_create(omegas[0], N, Npml, xrange, yrange,
                                    matrix_format=matrix_format)

    # Construct derivate matrices
    Dyb = Syb.dot(createDws('y', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
    Dxb = Sxb.dot(createDws('x', 'b', dL(N, xrange, yrange), N, matrix_format=matrix_format))
    Dxf = Sxf.dot(createDws('x', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))
    Dyf = Syf.dot(createDws('y', 'f', dL(N, xrange, yrange), N, matrix_format=matrix_format))

    # Construct A matrix
    A1 = mu0_const*block_diag((T_eps_y, T_eps_x), format=matrix_format)
    A2 = block_diag((T_eps_y, T_eps_x), format=matrix_format) \
        .dot(vstack([-Dyf, Dxf], format=matrix_format)) \
        .dot(T_eps_z_inv) \
        .dot(hstack([-Dyb, Dxb], format=matrix_format))
    A3 = vstack([Dxb, Dyb], format=matrix_format) \
        .dot(hstack([Dxf, Dyf], format=matrix_format))

    return (A1, A2, A3, Dxf, Dxb, Dyf, Dyb)


def calculate_beta_2D(omegas, beta_est, Neigs, xrange, yrange,
                    eps_r, Npml, matrix_format='csc'):

    if isscalar(omegas):
        Nomegas = 1
        omegas = array([omegas])
    else:
        Nomegas = omegas.size

    N = asarray(eps_r.shape)  # Number of mesh cells
    M = prod(N)  # Number of unknowns

    eps_x = grid_average(epsilon0_const*eps_r, 'y')
    eps_y = grid_average(epsilon0_const*eps_r, 'x')
    eps_z = grid_average(epsilon0_const*eps_r, 'x')
    eps_z = grid_average(eps_z, 'y')

    vector_eps_x = eps_x.ravel(order='F')
    vector_eps_y = eps_y.ravel(order='F')
    vector_eps_z = eps_z.ravel(order='F')

    # Setup the T_eps_x, T_eps_y, and T_eps_z matrices
    T_eps_x = spdiags(vector_eps_x, 0, M, M, format=matrix_format)
    T_eps_y = spdiags(vector_eps_y, 0, M, M, format=matrix_format)

    T_eps_x_inv = spdiags(1/vector_eps_x, 0, M, M, format=matrix_format)
    T_eps_y_inv = spdiags(1/vector_eps_y, 0, M, M, format=matrix_format)
    T_eps_z_inv = spdiags(1/vector_eps_z, 0, M, M, format=matrix_format)

    (A1, A2, A3,
     Dxf, Dxb, Dyf, Dyb) = assemble_matrix_2D(omegas, N, Npml, xrange, yrange,
                       T_eps_x, T_eps_y, T_eps_z_inv, matrix_format=matrix_format)

    betas = zeros((Nomegas, Neigs), dtype=complex128)
    Hx = zeros((Nomegas, Neigs, N[0], N[1]), dtype=complex128)
    Hy = zeros((Nomegas, Neigs, N[0], N[1]), dtype=complex128)
    Hz = zeros((Nomegas, Neigs, N[0], N[1]), dtype=complex128)
    Ex = zeros((Nomegas, Neigs, N[0], N[1]), dtype=complex128)
    Ey = zeros((Nomegas, Neigs, N[0], N[1]), dtype=complex128)
    Ez = zeros((Nomegas, Neigs, N[0], N[1]), dtype=complex128)

    for iw in range(0, Nomegas):
        A = omegas[iw]**2*A1+A2+A3

        if isscalar(beta_est):
            (values, vectors) = solver_eigs(A, Neigs, guess_value=beta_est**2)
        else:
            (values, vectors) = solver_eigs(A, Neigs, guess_value=beta_est[iw]**2)

        inds_sorted = argsort(real(sqrt(values)))[::-1]
        betas[iw, :] = sqrt(values[inds_sorted])

        for i in range(0, Neigs):
            jk = 1j*betas[iw, i]

            hx = vectors[:M, i]
            hy = vectors[-M:, i]
            hz = 1/(jk)*(Dxf.dot(hx) + Dyf.dot(hy))

            ex = -1j/omegas[iw]*T_eps_x_inv.dot(Dyb.dot(hz)+jk*hy)
            ey = -1j/omegas[iw]*T_eps_y_inv.dot(-jk*hx-Dxb.dot(hz))
            ez = -1j/omegas[iw]*T_eps_z_inv.dot(Dxb.dot(hy)-Dyb.dot(hx))

            Hx[iw, i, :, :] = hx.reshape((N[0], N[1]), order='F')
            Hy[iw, i, :, :] = hy.reshape((N[0], N[1]), order='F')
            Hz[iw, i, :, :] = hz.reshape((N[0], N[1]), order='F')
            Ex[iw, i, :, :] = ex.reshape((N[0], N[1]), order='F')
            Ey[iw, i, :, :] = ey.reshape((N[0], N[1]), order='F')
            Ez[iw, i, :, :] = ez.reshape((N[0], N[1]), order='F')

    return (betas, Ex, Ey, Ez, Hx, Hy, Hz)
