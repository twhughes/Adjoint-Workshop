# AVM Workshop

This notebook is a live demo explaining the adjoint variable method (AVM) and its application to electromagnetic problems and designing laser-driven accelerators.

## To install dependencies, run
```pip install -r requirements.txt```

## To view and play around with the notebook, run
```jupyter notebook Main.ipynb```
or
```jupyter lab Main.ipynb```
if you have jupyter lab installed

Although a full overview and explanation is given in Main.ipynb, here is a copy of the notebook.


# AVM Tutorial  - Tyler Hughes (Stanford University)

Adjoint variable method (or AVM) is a method for computing the derivative of an objective function with respect to several degrees of freedom in the system.  

For a dielectric laser accelerator, for instance, one could imagine these degrees of freedom being the height or width of the grating or radius of a pillar.  

With this gradient information, one can optimize structures easily by doing a gradient-based optimization procedure, such as gradient ascent, where the structure is repeatedly updated based on gradient information.

The main advantage of AVM is that no matter how many degrees of freedom you have, you can compute the gradient with respect to **all** of them using only 2 electromagnetic simulations:

1.  The first we will call the 'original', which corresponds to the sitation that you want to optimize.  
2.  The second we will call the 'adjoint',  which contains information about the objective function that we care about maximizing or minimizing.

Because of the ability to compute derivatives with respect to several degrees of freedom with little additional computational cost, one option is to take the dielectric function or permittivity at each point in space as our degrees of freedom.  This was done in [this work](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-13-15414), which I will be briefly summarizing in this talk.

## Live Demo

AVM works because we know the mathematical form of our electromagnetic situation is **linear**, and thus can be represented on a computer by a matrix-vector multiplication.  For us, the starting point for AVM will be the finite-difference frequency-domain algorithm (or FDFD), which solves for the electromagnetic fields in steady state (frequency domain).  For more details, [this](http://www.mit.edu/~wsshin/pdf/shin_dissertation_updated.pdf) is a good reference.

Before explaining FDFD in detail, we will load the modules we will need to use to run it, including:

* ```numpy``` (python's numerical math package)
* ```matplotlib``` (for plotting)
* ```python_fdfd``` (our FDFD package)

The first two will need to be installed, but the third is included in this directory.  

We'll also be using ```progressbar2``` for convenience.

To install everything with ```pip``` via the command line, use ```pip install -r requirements.txt```.

We will be using an FDFD written in python by [Ian Williamson](https://github.com/ianwilliamson) from my group, although MATLAB versions can also be found at [my github](https://github.com/twhughes) or supplied upon request.

Finally, the full code for this tutorial is available [here](https://github.com/twhughes/AVM_Workshop).

Let's get started..


```python
# Some python notebook magic to automatically reload src files
%load_ext autoreload
%autoreload 2

# import the modules
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from python_fdfd.fdfd import assign_val, driven
import progressbar

%matplotlib inline
# I'm using my own stylesheet for plotting.  (Comment out if not connected to internet)
plt.style.use(['https://git.io/photons.mplstyle',
               'https://git.io/photons-presentation.mplstyle'])
```

We will start with just a simple FDFD simulation to understand the basics before moving to AVM.  
The domain we want to create is diagrammed in the image below.  

* On the outside, we have PMLs (perfectly matched layers), which act as an absorbing layer around our simulation, simulating an infinite space.
* A dielectric rectangle sits in the center of the domain.
* A point source is placed left of the dielectric rectangle.

<img src="./img/Scene.png" width="500" />

Now we will define some of our important variables for running the simulation.



```python
# number of x,y grid points in domain
N = (100, 100)

# spatial ranges for the domain (m)
xrange = (-2e-6, 2e-6)
yrange = (-2e-6, 2e-6)

# angular frequency of source (rad/s)
omega = 2*np.pi*200e12

# define the relative permittivity of the region (adding rectangle)
eps_r = np.ones(N, dtype=complex)
rel_permittivity = 3
within_rect1 = lambda x, y: np.logical_and.reduce((-0.5e-6<x,x<0.5e-6,-0.75e-6<y,y<0.75e-6))
eps_r = assign_val(eps_r, within_rect1, rel_permittivity, xrange, yrange)

# define the point source
Jz = np.zeros(N, dtype=complex)
Jz[N[0]//4,N[1]//2] = 1

# define number of PMLs (should have about as many PMLs points as grids in a wavelength, 10 is usually ok)
Npml = (10, 10)
```


Now that our source and domain are defined, we may use FDFD to solve for the electromagnetic fields.

Assuming an anisotripic, reciprocal, and non-magnetic system (most normal cases) FDFD solves for the fields at a given frequency <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/> by solving the equation

<p align="center"><img src="/tex/2758016ca71a696079afe6765dcf535d.svg?invert_in_darkmode&sanitize=true" align=middle width=311.01482939999994pt height=36.1865163pt/></p>

for the electric fields <img src="/tex/01397eef679b7e7b92167e765c1c414c.svg?invert_in_darkmode&sanitize=true" align=middle width=33.74056619999999pt height=24.65753399999998pt/>.  This equation is derived straightforwardly from the steady state (single <img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/>) Maxwell's equations.

FDFD does this by representing the electric fields, permittivity distributions, and source on a [Yee lattice](https://meep.readthedocs.io/en/latest/Yee_Lattice/) and then expressing the <img src="/tex/775af2dfac29c4cd9b42d2993f6e7e3e.svg?invert_in_darkmode&sanitize=true" align=middle width=60.27396869999999pt height=22.465723500000017pt/> operator as finite difference derivative matrix, <img src="/tex/c78e3e85881c66b66dda99243f0da8bd.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=31.141535699999984pt/>.

In the discrete case, we may write our original equation as a big matrix equation:

<p align="center"><img src="/tex/5c2a598067749e4d30d9f4ca355b73b8.svg?invert_in_darkmode&sanitize=true" align=middle width=159.21151949999998pt height=29.58934275pt/></p>
<p align="center"><img src="/tex/9c512e43971bf7219a30c97ccf464779.svg?invert_in_darkmode&sanitize=true" align=middle width=74.520204pt height=15.570767849999998pt/></p>

Let's break down the terms here:
* <img src="/tex/c78e3e85881c66b66dda99243f0da8bd.svg?invert_in_darkmode&sanitize=true" align=middle width=14.06623184999999pt height=31.141535699999984pt/> is a real-valued matrix that performs the finite difference derivative <img src="/tex/0ee00535db8eb75296cd75007579835c.svg?invert_in_darkmode&sanitize=true" align=middle width=76.65503505pt height=27.77565449999998pt/>.  It does not depend on our permittivity distribution.
* <img src="/tex/54ebc7af6ca66c4aec97af90166a3c4c.svg?invert_in_darkmode&sanitize=true" align=middle width=13.12984034999999pt height=22.831056599999986pt/> is a **diagonal** complex-valued matrix.  Each element along the diagonal corresponds to the relative permittivity of one cell in our domain.  Although the underlying relative permittivity may be real-valued, it becomes complex when PMLs are added as the system now has loss.
* <img src="/tex/ae48dff45ab57dda34b441bc7904377a.svg?invert_in_darkmode&sanitize=true" align=middle width=12.420021899999991pt height=22.55708729999998pt/> is a complex vector representing the (unknown) electric field distribution in the domain.
* <img src="/tex/a10ec92d13e76a02b538967f6b90b345.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=22.831056599999986pt/> is a complex vector representing the driving current source distribution.

**NOTE 1:** <img src="/tex/54ebc7af6ca66c4aec97af90166a3c4c.svg?invert_in_darkmode&sanitize=true" align=middle width=13.12984034999999pt height=22.831056599999986pt/>, <img src="/tex/ae48dff45ab57dda34b441bc7904377a.svg?invert_in_darkmode&sanitize=true" align=middle width=12.420021899999991pt height=22.55708729999998pt/>, and <img src="/tex/a10ec92d13e76a02b538967f6b90b345.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=22.831056599999986pt/> are **flattened** representations of the original objects in 2D or 3D.  They must therefore each be flattened according to the same procedure, such as   `E_vec = E(:)`   in matlab.

**NOTE 2:** If we have <img src="/tex/fb97d38bcc19230b0acd442e17db879c.svg?invert_in_darkmode&sanitize=true" align=middle width=17.73973739999999pt height=22.465723500000017pt/> x <img src="/tex/f9c4988898e7f532b9f826a75014ed3c.svg?invert_in_darkmode&sanitize=true" align=middle width=14.99998994999999pt height=22.465723500000017pt/> grids in our simulation, the matrix <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/> is of dimension (<img src="/tex/38c940e42b166347e72f8cc587bd9732.svg?invert_in_darkmode&sanitize=true" align=middle width=32.73970589999999pt height=22.465723500000017pt/>) x (<img src="/tex/38c940e42b166347e72f8cc587bd9732.svg?invert_in_darkmode&sanitize=true" align=middle width=32.73970589999999pt height=22.465723500000017pt/>).  If M and N are small (lets say 100), then we still will have a matrix <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/> with 10,000 x 10,000 = 100,000,000 elements, which is getting quite large! (especially for the next step..)  Therefore, we need to make sure that we construct <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/> as a **sparse** matrix

Now that we understand how to set up our simulation, we must solve for the fields.  

This can be written simply as:

<p align="center"><img src="/tex/25e67b39243ba12ba0934ab88ed9481b.svg?invert_in_darkmode&sanitize=true" align=middle width=74.81712975pt height=15.570767849999998pt/></p>

Thankfully this can be done very simply through a direct, sparse linear solver:
* MATLAB:  `E = A\b`
* Python:  `E = scipy.sparse.linalg.spsolve(A,b)`

These numerical packages will take care of the work for you.  **DO NOT** solve for <img src="/tex/471d65ea6d03a4f1ea1dd8be931d26c9.svg?invert_in_darkmode&sanitize=true" align=middle width=29.155366349999987pt height=26.76175259999998pt/> directly, for instance by doing `E = inv(A)*b` as this is highly inefficient.  

Also, Since these built-in solvers perform LU decomposition on A, instead of storing `inv(A)`, you may save this factorization for later solves with different sources if you desire.

For 3D simulations, one will generally have to use an iterative solver instead.

Now we are finally ready to do the field solution:

With this fdfd package, we simply call a function with our variables and it will do all of this behind the scenes and return the fields.



```python
# solve for the fields in our domain with FDFD
(Ez,Hx,Hy) = driven.solve_TM(omega, xrange, yrange, eps_r, Jz, Npml)
```


```python
# plot the field components
f = plt.figure(figsize=(15,5))
ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)
Ezplt = ax1.imshow(np.real(Ez[0]).T, cmap = "bwr", vmin = -3e-7, vmax = 3e-7)
Hxplt = ax2.imshow(np.real(Hx[0]).T, cmap = "bwr", vmin = -1e-9, vmax = 1e-9)
Hyplt = ax3.imshow(np.real(Hy[0]).T, cmap = "bwr", vmin = -1e-9, vmax = 1e-9)
ax1.set_title('<img src="/tex/10c7164b6171fa3aa7b7bd2a097a4a88.svg?invert_in_darkmode&sanitize=true" align=middle width=50.078975099999994pt height=24.65753399999998pt/>')
ax2.set_title('<img src="/tex/4da4ebe17457e22dcded1325e8c25fc7.svg?invert_in_darkmode&sanitize=true" align=middle width=52.310654549999995pt height=24.65753399999998pt/>')
ax3.set_title('<img src="/tex/58b40c1917c246b89cc37fe7eb3479da.svg?invert_in_darkmode&sanitize=true" align=middle width=51.935891699999985pt height=24.65753399999998pt/>')
ax1.set_xlabel('x points')
ax2.set_xlabel('x points')
ax3.set_xlabel('x points')
ax1.set_ylabel('y points')
plt.show()
```

## AVM

To summarize, with FDFD, we first construct our <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/> matrix

<p align="center"><img src="/tex/02c00f0e8efa35df9ef8a3b1111f5c34.svg?invert_in_darkmode&sanitize=true" align=middle width=202.68945839999998pt height=36.1865163pt/></p>

and our source vector

<p align="center"><img src="/tex/bbd6ae6d77a52ea8149a8e70f69cbceb.svg?invert_in_darkmode&sanitize=true" align=middle width=80.259729pt height=16.438356pt/></p>

and then we solve for the electric fields using a linear solve:

<p align="center"><img src="/tex/25e67b39243ba12ba0934ab88ed9481b.svg?invert_in_darkmode&sanitize=true" align=middle width=74.81712975pt height=15.570767849999998pt/></p>

However, in many optimization problems, we would like to evaluate an 'objective function' that depends on the electric field distribution, we will call this <img src="/tex/c81a901e4aa9367d2060c067eab671cb.svg?invert_in_darkmode&sanitize=true" align=middle width=35.90180549999999pt height=24.65753399999998pt/> for now.

If we are doing optimization, we would like <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> to be a real-valued scalar quantity that we can either minimize or maximize with respect to the degrees of freedom in our system.

For sake of argument, lets say we would like to minimize <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> with respect to the relative permittivity at position <img src="/tex/37bb59bcb3c8c723506af9f3796c92c5.svg?invert_in_darkmode&sanitize=true" align=middle width=11.66291774999999pt height=24.7161288pt/> in our system.  

What we would like to do is to compute the derivative of <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> with respect to <img src="/tex/eaed87c3f3701cdac038d20526d76dbc.svg?invert_in_darkmode&sanitize=true" align=middle width=39.22197509999999pt height=24.7161288pt/>.  Let's write this out:

<p align="center"><img src="/tex/0b3209a0c05fc522250eae529c669126.svg?invert_in_darkmode&sanitize=true" align=middle width=253.46335905pt height=37.9216761pt/></p>

where we have assumed <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> does not depend explicitly on the permittivity distribution.

Because we assume both <img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/> and <img src="/tex/eaed87c3f3701cdac038d20526d76dbc.svg?invert_in_darkmode&sanitize=true" align=middle width=39.22197509999999pt height=24.7161288pt/> are real, this becomes:

<p align="center"><img src="/tex/cd69608c8ca5f026dece521e60bd907d.svg?invert_in_darkmode&sanitize=true" align=middle width=198.94483785pt height=39.452455349999994pt/></p>

As a sanity check, <img src="/tex/e95fb685fb668cfa7b7cd488af28c56a.svg?invert_in_darkmode&sanitize=true" align=middle width=17.44463325pt height=28.92634470000001pt/> is a row vector, which only depends on the choice of objective function and <img src="/tex/f7e3a84138a66e3c330aa18b0a39b474.svg?invert_in_darkmode&sanitize=true" align=middle width=40.12470825pt height=28.92634470000001pt/> is a vector that we will evaluate by examining our FDFD simulation.  Therefore, our <img src="/tex/a52e996f41dc8914a482f1be02842fa6.svg?invert_in_darkmode&sanitize=true" align=middle width=40.12470825pt height=28.92634470000001pt/> quantity is a real-valued scalar, as desired.

To evaluate <img src="/tex/f7e3a84138a66e3c330aa18b0a39b474.svg?invert_in_darkmode&sanitize=true" align=middle width=40.12470825pt height=28.92634470000001pt/>, we do a bit of calculus

<p align="center"><img src="/tex/53867c199b29b9c968cf693a6e1f704b.svg?invert_in_darkmode&sanitize=true" align=middle width=145.0505595pt height=51.3856596pt/></p>

<p align="center"><img src="/tex/5108019aed7d5f6cccae29a491ba83bf.svg?invert_in_darkmode&sanitize=true" align=middle width=152.3167734pt height=42.07691565pt/></p>
where we have used the fact that <img src="/tex/a10ec92d13e76a02b538967f6b90b345.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=22.831056599999986pt/> does not depend on <img src="/tex/67f1777f6dfce783fa8fe87d2ab5f12f.svg?invert_in_darkmode&sanitize=true" align=middle width=13.12984034999999pt height=14.15524440000002pt/>

Noticing that <img src="/tex/8d1284da3bbac027015a7100c7401424.svg?invert_in_darkmode&sanitize=true" align=middle width=74.81712974999999pt height=31.141535699999984pt/> and resubstituting, we arrive at:

<p align="center"><img src="/tex/e1a42bf28eb24d144b3dbfa41467fd60.svg?invert_in_darkmode&sanitize=true" align=middle width=178.5738999pt height=42.07691565pt/></p>

Finally, in our original equation now, we have:

<p align="center"><img src="/tex/7d0011f1a294cb0fe7046e0a75964619.svg?invert_in_darkmode&sanitize=true" align=middle width=255.9540984pt height=49.315569599999996pt/></p>

Because of reciprocity, we are guarenteed that <img src="/tex/ca2186b1af0b0774d3347edbd138bc1b.svg?invert_in_darkmode&sanitize=true" align=middle width=57.15919439999999pt height=31.141535699999984pt/> and <img src="/tex/9c4a3818c321d08df1fb0e62b1aafd7b.svg?invert_in_darkmode&sanitize=true" align=middle width=84.03137984999998pt height=31.141535699999984pt/>.  Thus, we can group the terms on the left of this equation.

<p align="center"><img src="/tex/240016a67121ed2c59cbe797b876cd80.svg?invert_in_darkmode&sanitize=true" align=middle width=124.56005924999998pt height=33.81208709999999pt/></p>

or, to write in a different way (taking transpose)

<p align="center"><img src="/tex/f60357fa936f2acb9a4c306e058352f0.svg?invert_in_darkmode&sanitize=true" align=middle width=164.50467824999998pt height=49.315569599999996pt/></p>

We then notice that <img src="/tex/a4bf71e9ddb65afc4f6f25a7169def2c.svg?invert_in_darkmode&sanitize=true" align=middle width=25.65491279999999pt height=22.55708729999998pt/> is the field solution that we get from solving

<p align="center"><img src="/tex/19d29c1aa32443e978c2385439c16537.svg?invert_in_darkmode&sanitize=true" align=middle width=117.26720445pt height=37.61121705pt/></p>

Which is a simulation with the same system (<img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/>) but now instead of a source <img src="/tex/a10ec92d13e76a02b538967f6b90b345.svg?invert_in_darkmode&sanitize=true" align=middle width=10.502226899999991pt height=22.831056599999986pt/>, the source is <img src="/tex/a5d4ed94e28ef9eb48db780830ac8bb2.svg?invert_in_darkmode&sanitize=true" align=middle width=51.928164749999986pt height=36.5245749pt/>, which depends on our objective function.  This field solution is what is called the **adjoint** field and the equation above for solving for it is the **adjoint** problem.

Thus, we can write the change in objective function with respect to the permittivity at point <img src="/tex/37bb59bcb3c8c723506af9f3796c92c5.svg?invert_in_darkmode&sanitize=true" align=middle width=11.66291774999999pt height=24.7161288pt/> as the overlap between two electric fields <img src="/tex/ae48dff45ab57dda34b441bc7904377a.svg?invert_in_darkmode&sanitize=true" align=middle width=12.420021899999991pt height=22.55708729999998pt/> and <img src="/tex/a43695c3016bb266352f27fe8c0ee3c5.svg?invert_in_darkmode&sanitize=true" align=middle width=25.65491279999999pt height=22.55708729999998pt/>:

<p align="center"><img src="/tex/a8fdcc36676e10ff6ade4126b5e49d46.svg?invert_in_darkmode&sanitize=true" align=middle width=205.44339254999997pt height=49.315569599999996pt/></p>

The matrix <img src="/tex/daca67ba5a0eb1ab33a6f79f780ca24e.svg?invert_in_darkmode&sanitize=true" align=middle width=40.12470825pt height=34.74372989999999pt/> can be evaluated simply by inspecting the form of <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/>, since <img src="/tex/6c9593d82fc74cb581359f835452e977.svg?invert_in_darkmode&sanitize=true" align=middle width=12.55717814999999pt height=31.141535699999984pt/> contains a matrix with <img src="/tex/179588e16e1fabb29de4ad34840ef549.svg?invert_in_darkmode&sanitize=true" align=middle width=34.610099699999985pt height=24.65753399999998pt/> along the diagonal, when we differentiate this matrix with respect to <img src="/tex/eaed87c3f3701cdac038d20526d76dbc.svg?invert_in_darkmode&sanitize=true" align=middle width=39.22197509999999pt height=24.7161288pt/> it will give a matrix where there is <img src="/tex/ea5e523e3737f2f3a23cd7d1d52e958f.svg?invert_in_darkmode&sanitize=true" align=middle width=44.206737299999986pt height=26.76175259999998pt/> along the diagonal where <img src="/tex/c5a179081988c08b3bec25ddd4f43c52.svg?invert_in_darkmode&sanitize=true" align=middle width=42.27541559999999pt height=24.7161288pt/> and 0 everywhere else. 

We can call this matrix <img src="/tex/b5c9a6ab927de9a9ecdf6f3f7ca38d6d.svg?invert_in_darkmode&sanitize=true" align=middle width=72.07026914999999pt height=31.50689519999998pt/> where it is like a kronecker delta.

Inserting this, we get some nice simplification of the expression for the derivative, which now only depends on the orginal and the adjoint fields as.

<p align="center"><img src="/tex/bad47b3760b8d8b2f97f45b9ff4f01f2.svg?invert_in_darkmode&sanitize=true" align=middle width=412.36133565pt height=37.9216761pt/></p>

The real power of this expression is that if we want to now look at the change in our objective function with respect to the permittivity at any general point, <img src="/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/>, we can reuse our field solutions 

<p align="center"><img src="/tex/1a19fb8cf67b211545480297c2b99503.svg?invert_in_darkmode&sanitize=true" align=middle width=225.4081302pt height=37.9216761pt/></p>

In this way, we are able to get the change in objective function with respect to **each** pixel in our permittivity distribution **at once** all by doing two field simulations and using the equation above.  Even if we have tons of pixels, once we solve for <img src="/tex/a43695c3016bb266352f27fe8c0ee3c5.svg?invert_in_darkmode&sanitize=true" align=middle width=25.65491279999999pt height=22.55708729999998pt/>, we have our derivatives.


## Example

To see how this works, let's look at a very simple example.

Let's say that we're interested in the same situation as before (point source next to a dielectric rectangle) but now we want to maximize the electric field intensity at a point on the other side of the rectangle.

<img src="./img/Objective.png" width="500" />


To simplify the problem, we may write down a vector <img src="/tex/66fa5ab6b878d0218ecc58a66127814a.svg?invert_in_darkmode&sanitize=true" align=middle width=10.471830599999988pt height=14.611878600000017pt/> that gives 1 at that target position and 0 elsewhere.

Now, we can express an objective function as 

<p align="center"><img src="/tex/b16e453e80263741e1ac503dbff65285.svg?invert_in_darkmode&sanitize=true" align=middle width=238.24663334999997pt height=23.755462499999997pt/></p>

To construct our adjoint source, one can show

<p align="center"><img src="/tex/3f42666685fd40c1d38fcc9ee6f81897.svg?invert_in_darkmode&sanitize=true" align=middle width=232.83560354999997pt height=37.61121705pt/></p>

Therefore, the source for our adjoint field is at the probe location, <img src="/tex/66fa5ab6b878d0218ecc58a66127814a.svg?invert_in_darkmode&sanitize=true" align=middle width=10.471830599999988pt height=14.611878600000017pt/>, and has a (complex-valued) amplitude that is -2 times the complex conjugate of the forward field evaluated at the proble location.

We can easily code this up with our FDFD solver:


```python
eta = np.zeros(N)
eta[3*N[0]//4,N[1]//2] = 1
Jz_aj = -2*np.conj(Ez*eta)*eta/1j/omega

(Ez_aj,Hx_aj,Hy_aj) = driven.solve_TM(omega, xrange, yrange, eps_r, Jz_aj, Npml)
```

Lets plot the adjoint fields


```python
f = plt.figure(figsize=(15,5))
ax1 = f.add_subplot(131)
ax2 = f.add_subplot(132)
ax3 = f.add_subplot(133)
Ezplt = ax1.imshow(np.real(Ez_aj[0]).T, cmap = "bwr")#, vmin = -2e-11, vmax = 2e-11)
Hxplt = ax2.imshow(np.real(Hx_aj[0]).T, cmap = "bwr")#, vmin = -5e-14, vmax = 5e-14)
Hyplt = ax3.imshow(np.real(Hy_aj[0]).T, cmap = "bwr")#, vmin = -5e-14, vmax = 5e-14)
ax1.set_title('adjoint <img src="/tex/10c7164b6171fa3aa7b7bd2a097a4a88.svg?invert_in_darkmode&sanitize=true" align=middle width=50.078975099999994pt height=24.65753399999998pt/>')
ax2.set_title('adjoint <img src="/tex/4da4ebe17457e22dcded1325e8c25fc7.svg?invert_in_darkmode&sanitize=true" align=middle width=52.310654549999995pt height=24.65753399999998pt/>')
ax3.set_title('adjoint <img src="/tex/58b40c1917c246b89cc37fe7eb3479da.svg?invert_in_darkmode&sanitize=true" align=middle width=51.935891699999985pt height=24.65753399999998pt/>')
ax1.set_xlabel('x points')
ax2.set_xlabel('x points')
ax3.set_xlabel('x points')
ax1.set_ylabel('y points')
plt.show()
```

Note that the adjoint is sourced at the right of the dielectric square, where we are measuring the intensity for our objective function.

And now, the sensitivity map can be plotted.


```python
#plot the sensitivity map
f = plt.figure(figsize=(6,5))
max_abs = np.max(np.max(np.abs(Ez_aj[0]*Ez[0])))
Ezplt = plt.imshow(np.real(Ez_aj[0]*Ez[0]/max_abs).T, cmap = "bwr", vmin = -1/2, vmax = 1/2)
plt.title('dJ/d<img src="/tex/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode&sanitize=true" align=middle width=6.672392099999992pt height=14.15524440000002pt/> =  <img src="/tex/f4f3931463524e58bc8d844c0d9a4423.svg?invert_in_darkmode&sanitize=true" align=middle width=81.51564464999998pt height=24.65753399999998pt/>')
plt.xlabel('x points')
plt.ylabel('y points')
plt.colorbar()
plt.show()
```

This shows us how the objective function (<img src="/tex/8eb543f68dac24748e65e2e4c5fc968c.svg?invert_in_darkmode&sanitize=true" align=middle width=10.69635434999999pt height=22.465723500000017pt/>) will change when we change the relative permittivity of each point in the domain.

## Frequency Sweep

Your adjoint sensitivity should be consistent with that found by taking a numerical derivative, where the permittivity is manually changed by some small amount and the change in objective function is measured.

<p align="center"><img src="/tex/d0864204e664550bb53f9cbf6de3df54.svg?invert_in_darkmode&sanitize=true" align=middle width=315.44658915pt height=42.4879191pt/></p>

To do this, we'll compare AVM with numerical derivative when we scan the frequency.  For example, let's take our degree of fredom to be the permittivity of the entire box.

In this case, our total objective function can be evaluated by just summing over <img src="/tex/ae48dff45ab57dda34b441bc7904377a.svg?invert_in_darkmode&sanitize=true" align=middle width=12.420021899999991pt height=22.55708729999998pt/> and <img src="/tex/a43695c3016bb266352f27fe8c0ee3c5.svg?invert_in_darkmode&sanitize=true" align=middle width=25.65491279999999pt height=22.55708729999998pt/> over the box location.

<p align="center"><img src="/tex/fc2ac7a3b327d29255f421f64459b0f9.svg?invert_in_darkmode&sanitize=true" align=middle width=242.74165079999997pt height=43.17873615pt/></p>

This will take a minute or so


```python
# frequency scan
Nf = 50           # number of frequencies to scan over
freqs = np.linspace(2*np.pi*150e12, 2*np.pi*250e12, Nf)
dJde_numerical = []
dJde_AVM = []

deps = 1e-6       # size of numerical permittivity update
er = 3            # relative permittivity
e0 = 8.854e-12

# loop through freqencies (use progress bar to track progress)
pgb = progressbar.ProgressBar(max_value=Nf)
for index, w in enumerate(freqs):
    pgb.update(index+1)
    
    # make source
    Jz = np.zeros(N, dtype=complex)
    Jz[N[0]//4,N[1]//2] = 1
    
    # numerical derivative    
    
    eps_up = np.ones(N, dtype=complex)    
    eps_up = assign_val(eps_up, within_rect1, er+deps/2, xrange, yrange)
    # compute fields and objective function when permittivity increased    
    (Ez_up,Hx,Hy) = driven.solve_TM(w, xrange, yrange, eps_up, Jz, Npml)
    J_up = np.abs(np.sum(np.sum(eta*Ez_up[0])))**2
    eps_down = np.ones(N, dtype=complex)        
    eps_down = assign_val(eps_down, within_rect1, er-deps/2, xrange, yrange)
    # compute fields and objective function when permittivity decreased    
    (Ez_down,Hx,Hy) = driven.solve_TM(w, xrange, yrange, eps_down, Jz, Npml)
    J_down = np.abs(np.sum(np.sum(eta*Ez_down[0])))**2
    # compute numerical derivative
    dJde_numerical.append((J_up - J_down)/deps)
    
    # adjoint derivative
    
    # compute original fields
    eps_r = np.ones(N, dtype=complex)        
    eps_r = assign_val(eps_r, within_rect1, er, xrange, yrange)  
    (Ez,Hx,Hy) = driven.solve_TM(w, xrange, yrange, eps_r, Jz, Npml)
    # construct adjoint soure (note the omega dependence..)
    Jz_aj = -2*np.conj(Ez[0]*eta)*eta/1j/w
    # solve adoint fields
    (Ez_aj,Hx_aj,Hy_aj) = driven.solve_TM(w, xrange, yrange, eps_r, Jz_aj, Npml)
    # compute adjoint derivative (summing field overlap over box region)
    dJde_AVM.append(w**2*e0*np.real(np.sum(np.sum((Ez[0]*Ez_aj[0]*(eps_r>1))))))
    
```


```python
# Plot the results (should be perfect agreement)
num = plt.plot([f/1e12/2/np.pi for f in freqs],dJde_numerical)
avm = plt.plot([f/1e12/2/np.pi for f in freqs],dJde_AVM,'o')
plt.xlabel('<img src="/tex/ae4fb5973f393577570881fc24fc2054.svg?invert_in_darkmode&sanitize=true" align=middle width=10.82192594999999pt height=14.15524440000002pt/> (2<img src="/tex/f30fdded685c83b0e7b446aa9c9aa120.svg?invert_in_darkmode&sanitize=true" align=middle width=9.96010619999999pt height=14.15524440000002pt/> THz)')
plt.ylabel('sensitivity (dJ/d<img src="/tex/d824f57c81e410d76c1b5a4c63b59d2d.svg?invert_in_darkmode&sanitize=true" align=middle width=26.39620334999999pt height=14.15524440000002pt/>)')
plt.legend(['numerical derivative', 'AVM derivative'])
plt.show()
```

## Optimization / Inverse Design Demo

The power of this formalism is that it allows us to now optimize structures using each permittivity as our degree of freedom.

Let's take the same situation as before, but now design the central square region to maximize the intensity concentration at the target position.

We do this by first computing:

<p align="center"><img src="/tex/29498c3e0a4db608f52f3f352fc6fa84.svg?invert_in_darkmode&sanitize=true" align=middle width=43.16606415pt height=37.9216761pt/></p> for each <img src="/tex/89f2e0d2d24bcf44db73aab8fc03252c.svg?invert_in_darkmode&sanitize=true" align=middle width=7.87295519999999pt height=14.15524440000002pt/> in our design region using adjoints.

Then, we update the permittivity in that region by a simple gradient ascent update rule:

<p align="center"><img src="/tex/2948f557c03a6c76244a864e20b33a7e.svg?invert_in_darkmode&sanitize=true" align=middle width=171.51040665pt height=37.9216761pt/></p>

Where <img src="/tex/c745b9b57c145ec5577b82542b2df546.svg?invert_in_darkmode&sanitize=true" align=middle width=10.57650494999999pt height=14.15524440000002pt/> is some small step size.

**NOTE**: the original and adjoint fields can be stored to compute the derivative with respect to the permittivity in each pixel.  One can do this either with a for loop, or (better) by just directly element-wise multiplying the fields together.

We continue this process until we have converged on a structure.

If <img src="/tex/179588e16e1fabb29de4ad34840ef549.svg?invert_in_darkmode&sanitize=true" align=middle width=34.610099699999985pt height=24.65753399999998pt/> either gets smaller than 1 or larger than some cutoff during our iterations, we set them back inside the range.


```python
# inverse design demo: maximize intensity at probe location

N_iterations = 1000
alpha = 1e14             # gradient ascent step size
eps_cutoff = 5           # maximum relative permittivity
eps_start = 3            # starting relative permittivity of box
omega = 2*np.pi*200e12

eps_r = np.ones(N, dtype=complex)        
eps_r = assign_val(eps_r, within_rect1, eps_start, xrange, yrange)

J_list = []

# Create a named display and plots to update in loop
handle = display(None, display_id=True)
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5))
perm_plot = ax1.imshow(np.real(eps_r.T), cmap="bwr")
field_plot = ax2.imshow(np.abs(Ez[0].T), cmap="magma")
J_line, = ax3.plot(range(N_iterations), [0 for _ in range(N_iterations)])
ax1.set_title('permittivity')
ax2.set_title('|E_z|')    
ax3.set_ylim([0,8e-13])
ax3.set_xlabel('iteration')
ax3.set_ylabel('objective function')

# run for several iterations
for iteration_counter in range(N_iterations):

    # solve for the original fields and objective function
    (Ez,Hx,Hy) = driven.solve_TM(w, xrange, yrange, eps_r, Jz, Npml)
    obj_fun = np.abs(np.sum(np.sum(eta*Ez[0])))**2
    J_list.append(obj_fun)
    
    # compute adjoint source and fields
    Jz_aj = -2*np.conj(Ez[0]*eta)*eta/1j/w
    (Ez_aj,Hx_aj,Hy_aj) = driven.solve_TM(w, xrange, yrange, eps_r, Jz_aj, Npml)
    
    # compute adjoint gradient with respect to ALL pixels in box region
    dJde_AVM = w**2*e0*np.real(Ez[0]*Ez_aj[0]*(eps_r>1))    
    
    # update permittivit with gradients, set outsiders back in range
    eps_r = eps_r + alpha*dJde_AVM
    eps_r[eps_r<1] = 1
    eps_r[eps_r>eps_cutoff] = eps_cutoff
    
    # update plots
    perm_plot.set_array(np.real(eps_r).T)
    field_plot.set_array(np.abs(Ez[0].T))
    J_line.set_ydata([J_list[i] for i in range(iteration_counter)] + [0 for i in range(N_iterations-iteration_counter)])    
    ax3.set_xlim([0,iteration_counter-1])    

    
    # Update the named display with a new figure
    handle.update(fig)
    
```

One can see that a funny looking structure was created. (left)
If you are lucky, the permittivity distribution will converge onto either vacuum or max permittivity values with large feature sizes.  If not, you may need to use fancier techniques.

By looking at the intensity plots, it seems this structure does indeed focus at the probe location to some extent.

## Application to Accelerators

Due to the interest of time, I will give a brief description on how this may now be applied to designing dielectric laser accelerator structures.  

For a very detailed explanation please read my [paper](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-25-13-15414) on the subject.

The code used to make the figures is also given [here](https://github.com/twhughes/DLA-Structure-Optimization) and you can read the documentation in README.md

**NOTE:** The FDFD convention and coordinates are different in the paper than here.  In the paper, the electron moves in the <img src="/tex/f84e86b97e20e45cc17d297dc794b3e8.svg?invert_in_darkmode&sanitize=true" align=middle width=9.39498779999999pt height=22.831056599999986pt/> direction and the constant in front of the adjoint sensitivity is different.

For accelerators, we take a simple case where we want to maximize the acceleration gradient of a charged particle moving along the central gap of a DLA structure.

In the frequency domain, we can write the acceleration gradient (our objective function) as
<p align="center"><img src="/tex/1c74fb14891285112de99c0ddce86bcd.svg?invert_in_darkmode&sanitize=true" align=middle width=484.99340505000004pt height=49.315569599999996pt/></p>

Taking <img src="/tex/a64a590422510056485cda8e72ce9a95.svg?invert_in_darkmode&sanitize=true" align=middle width=47.30584319999999pt height=22.831056599999986pt/> as an arbitrary choice, this can further be expressed as the inner product between <img src="/tex/ae48dff45ab57dda34b441bc7904377a.svg?invert_in_darkmode&sanitize=true" align=middle width=12.420021899999991pt height=22.55708729999998pt/> and <img src="/tex/66fa5ab6b878d0218ecc58a66127814a.svg?invert_in_darkmode&sanitize=true" align=middle width=10.471830599999988pt height=14.611878600000017pt/> where

<p align="center"><img src="/tex/1f8b2fdcf5d0265ebdd5d7ebf52c382f.svg?invert_in_darkmode&sanitize=true" align=middle width=301.1625507pt height=49.315569599999996pt/></p>

<p align="center"><img src="/tex/16e546a7883ff5bf28569b3c2d57c96e.svg?invert_in_darkmode&sanitize=true" align=middle width=277.48266809999996pt height=39.452455349999994pt/></p>

Thus, by following the procedure above, we may identify the adjoint source as <img src="/tex/ff9487827c7a09c96347b2017d0d463b.svg?invert_in_darkmode&sanitize=true" align=middle width=23.25726314999999pt height=19.1781018pt/> and the same techniques can be applied.

In this case, the current source for the adjoint field corresponds to an electric current, <img src="/tex/dcc122b1d1f1c034ffc1e33654a003d8.svg?invert_in_darkmode&sanitize=true" align=middle width=15.86765564999999pt height=22.465723500000017pt/> at the center of the gap with an <img src="/tex/1549b80f29b492db1f0b1e011e91c921.svg?invert_in_darkmode&sanitize=true" align=middle width=79.35464624999999pt height=37.80850590000001pt/> dependence.

One can show that this is exactly proportional to the current source of a point particle moving through the center of the gap with speed <img src="/tex/3f7903a2571aab93e88bf15d11268475.svg?invert_in_darkmode&sanitize=true" align=middle width=26.307162749999986pt height=22.831056599999986pt/>.

Thus, for maximizing acceleration gradient, the adjoint fields are proportional to the radiation from the electron beam that you are trying to accelerate!

One can maximize structures for both acceleration gradient and acceleration factor (gradient divided by maximum electric field amplitude), yielding structures that look like this:

<img src="./img/Grad.png" width="500" />

- left = gradient alone maximized
- right = gradient / max |E| in optimization region maximized.

The second requires defining a special, differential approximation to the max function and taking some nasty derivatives.

### Other Interesting AVM Applications

- We can use the same analysis as above with objective functions of the same form: <img src="/tex/3e03b84eb83c39d002f66d5921ee9c21.svg?invert_in_darkmode&sanitize=true" align=middle width=106.2296631pt height=24.65753399999998pt/>. For example, now <img src="/tex/66fa5ab6b878d0218ecc58a66127814a.svg?invert_in_darkmode&sanitize=true" align=middle width=10.471830599999988pt height=14.611878600000017pt/> may represent a desired focusing field.

- One can also do hybrid approaches, such as maximizing the sum of the accelerating and focusing fields.

- As we show [here](https://arxiv.org/abs/1805.09943), for a system parameterized by optical phase shifters, we can compute how the objective function will change with respect to each phase shifter by summing the adjoint sensitivities over a phase shifter location.

- Furthermore, in [the same work](https://arxiv.org/abs/1805.09943), we also show that the adjoint senstivity can be read out as an intensity measurement in the device by inerfering <img src="/tex/5b447d3ce41d3cead29dd174bc04c3a9.svg?invert_in_darkmode&sanitize=true" align=middle width=25.73459459999999pt height=22.55708729999998pt/> with <img src="/tex/ed42a733e2c6667e7d5c634aac3585e6.svg?invert_in_darkmode&sanitize=true" align=middle width=25.65491279999999pt height=22.63846199999998pt/>.

## Conclusion

I hope that at least gives a flavor for how the adjoint variable method works.  

You can play around with this code or use it + our FDFD to implementing adjoint optmization in your own research.

If you want to discuss further or have questions, feel free to contact me at ```twhughes_at_stanford.edu```




Enjoy!