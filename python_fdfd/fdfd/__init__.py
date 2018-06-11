from numpy import pi, sqrt, array, diff, roll, linspace, meshgrid, where, isscalar, arange
#from .eigen import calculate_beta_1D

epsilon0_const = 8.854e-12
mu0_const = pi*4e-7
c_const = sqrt(1/epsilon0_const/mu0_const)
eta0 = sqrt(mu0_const/epsilon0_const)

def dL(N, xrange, yrange=None):
    if yrange is None:
        L = array([diff(xrange)[0]])  # Simulation domain lengths
    else:
        L = array([diff(xrange)[0],
                   diff(yrange)[0]])  # Simulation domain lengths
    return L/N

def d_(N, range):
    assert(len(range)==2) 
    L = diff(range)[0]
    return L/N

def grid_average(center_array, w):
    xy = {'x': 0, 'y': 1}
    center_shifted = roll(center_array, 1, axis=xy[w])
    avg_array = (center_shifted+center_array)/2
    return avg_array


def assign_val(val_array, region_cond, val_fun, xrange, yrange=None):
    N = val_array.shape

    xe = linspace(xrange[0], xrange[1], N[0]+1)
    xc = (xe[:-1]+xe[1:])/2

    if (yrange is None) or len(N) is 1:
        # Operate on a 1D grid

        if callable(val_fun):
            val_array = where(region_cond(xc), val_fun(xc), val_array)
        else:
            assert(isscalar(val_fun))
            val_array[region_cond(xc)] = val_fun

    else:
        # Operate on a 2D grid

        ye = linspace(yrange[0], yrange[1], N[1]+1)
        yc = (ye[:-1]+ye[1:])/2

        (xc, yc) = meshgrid(xc, yc, indexing='ij')

        if callable(val_fun):
            val_array = where(region_cond(xc, yc), val_fun(xc, yc), val_array)
        else:
            assert(isscalar(val_fun))
            val_array[region_cond(xc, yc)] = val_fun

    return val_array

def mode_source(J, eps_r, xyc, omega, xrange, yrange, Npts=31, normal='x', pol='tm', beta_est=None):
    N = eps_r.shape
    src_ind_x = int((xyc[0]-xrange[0])/diff(xrange)[0]*N[0])
    src_ind_y = int((xyc[1]-yrange[0])/diff(yrange)[0]*N[1])

    Nside = int((Npts-1)/2)
    inds = arange(-Nside,Nside+1)

    if normal is 'x':
        eps_r_src = eps_r[src_ind_x,src_ind_y+inds]
        dh = d_(N[1], yrange)
    else:
        eps_r_src = eps_r[src_ind_x+inds,src_ind_y]
        dh = d_(N[0], xrange)
    
    if beta_est is None:
        beta_est = omega/c_const*abs(eps_r_src).max()
    range = array([0, Npts*dh])

    (betas, Ey) = calculate_beta_1D(pol, omega, beta_est, 1, 0, range, eps_r_src)
    J[src_ind_x,src_ind_y+inds] = Ey

    return J