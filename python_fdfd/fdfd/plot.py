from numpy import pi, sqrt, array, prod, diff, roll, linspace, meshgrid, real, \
                    where, isscalar, zeros, ones, reshape, complex128, inf, asarray, append, squeeze, exp
from scipy.sparse import diags, spdiags, kron, eye
from scipy.sparse.linalg import eigs, spsolve
import time

import matplotlib.pyplot as plt
from matplotlib import animation

def plt_norm(Ex, Ey, Ez, x=None, y=None, w=0, m=0):
    extents = None
    normE = sqrt(abs(Ex[w, m, :, :])**2 +
                 abs(Ey[w, m, :, :])**2 +
                 abs(Ez[w, m, :, :])**2)
    if not (x is None or y is None):
        extents = [x[0], x[1], y[0], y[1]]

    h = plt.imshow(normE.transpose(), cmap=plt.get_cmap('inferno'),
                   extent=extents)
    plt.colorbar(h, label=r'$\vert \mathbf{E} \vert$')
    if extents is None:
        plt.xticks([])
        plt.yticks([])
    plt.show()

def plt_real(field_vals, x=None, y=None, w=0, m=0, cbar=True):
    field_vals = squeeze(field_vals)
    extents = None
    if not (x is None or y is None):
        extents = [x[0], x[1], y[0], y[1]]

    vmax=abs(field_vals).max()
    h = plt.imshow(real(field_vals).transpose(),cmap=plt.get_cmap('RdBu'),extent=extents,vmin=-vmax,vmax=vmax)
    if cbar:
        plt.colorbar()
    if extents is None:
        plt.xticks([])
        plt.yticks([])
    plt.show()


def plt_abs(field_vals, x=None, y=None, w=0, m=0, cbar=True):
    field_vals = squeeze(field_vals)
    extents = None
    if not (x is None or y is None):
        extents = [x[0], x[1], y[0], y[1]]

    h = plt.imshow(abs(field_vals).transpose(),cmap=plt.get_cmap('inferno'),extent=extents)
    if cbar:
        plt.colorbar()

    if extents is None:
        plt.xticks([])
        plt.yticks([])
    plt.show()


def ani_real(field_vals, x=None, y=None, cbar=False, Nframes=40, interval=80 ):
    field_vals = squeeze(field_vals).transpose()
    extents = None
    if not (x is None or y is None):
        extents = [x[0], x[1], y[0], y[1]]
    
    fig = plt.figure()
    h = plt.imshow(zeros(field_vals.shape),extent=extents)
    
    if cbar:
        plt.colorbar()
    if extents is None:
        plt.xticks([])
        plt.yticks([])

    def init():
        vmax=abs(field_vals).max()
        h.set_data(zeros(field_vals.shape))
        h.set_cmap('RdBu')
        h.set_clim(vmin=-vmax, vmax=+vmax)
        
        return (h,)

    def animate(i):
        h.set_data(real(field_vals*exp(1j*2*pi*i/(Nframes-1))))
        return (h,)
    
    plt.close()
    return animation.FuncAnimation(fig, animate, init_func=init, 
                                    frames=Nframes, interval=interval, blit=True)


def plot_bands(betas, omegas, n=None, scale='linear', figsize=(4, 4)):
    plt.figure(figsize=figsize)
    plt.plot(real(betas), omegas, '-')
    # plt.fill_between(omegas/c,omegas,omegas.max(),facecolor='#cccccc',edgecolor='none')

    beta_a = omegas/c_const
    plt.plot(beta_a, omegas, 'k-', lw=3)
    if n is not None:
        beta_d = beta_a*n
        plt.plot(beta_d, omegas, 'k-', lw=3)
    else:
        beta_d = beta_a

    betas_max = max([real(betas_te).max(), real(betas_tm).max()])

    plt.xlabel(r'Wave vector $\beta$')
    plt.ylabel(r'Frequency $\omega$')

    plt.xscale(scale)
    plt.yscale(scale)

    #if scale is 'linear':
    #    plt.xlim([0, betas_max])
    #    plt.ylim([0, omegas.max()])
    if scale is 'log':
        plt.xlim([beta_d.min(), betas_max])
        plt.ylim([0, omegas.max()])