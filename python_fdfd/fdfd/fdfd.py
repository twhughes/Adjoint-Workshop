from numpy import pi, sqrt, array, prod, diff, roll, linspace, meshgrid, real, \
                    where, isscalar, zeros, ones, reshape, complex128, inf, asarray, append, arange
from scipy.sparse import diags, spdiags, kron, eye
from scipy.sparse.linalg import eigs, spsolve
import time
from . import *

def createDws(w, s, dL, N, matrix_format='csc'):
    Nx = N[0]
    dx = dL[0]
    if len(N) is not 1:
        Ny = N[1]
        dy = dL[1]
    else:
        Ny = 1
        dy = inf

    if w is 'x':
        if s is 'f':
            dxf = diags([-1, 1, 1], [0, 1, -Nx+1], shape=(Nx, Nx))
            Dws = 1/dx*kron(eye(Ny), dxf, format=matrix_format)
        else:
            dxb = diags([1, -1, -1], [0, -1, Nx-1], shape=(Nx, Nx))
            Dws = 1/dx*kron(eye(Ny), dxb, format=matrix_format)

    if w is 'y':
        if s is 'f':
            dyf = diags([-1, 1, 1], [0, 1, -Ny+1], shape=(Ny, Ny))
            Dws = 1/dy*kron(dyf, eye(Nx), format=matrix_format)
        else:
            dyb = diags([1, -1, -1], [0, -1, Ny-1], shape=(Ny, Ny))
            Dws = 1/dy*kron(dyb, eye(Nx), format=matrix_format)

    return Dws


def solver_eigs(A, Neigs, guess_value=0, guess_vector=None, timing=False):
    if timing: start = time.time()
    
    (values, vectors) = eigs(A, k=Neigs, sigma=guess_value, v0=guess_vector, which='LM')

    if timing: end = time.time()
    if timing: print('Elapsed time for eigs() is %.4f secs' % (end - start))

    return (values, vectors)


def solver_direct(A, b, timing=False):
    if timing: start = time.time()
    
    x = spsolve(A, b)

    if timing: end = time.time()
    if timing: print('Elapsed time for spsolve() is %.4f secs' % (end - start))

    return x

