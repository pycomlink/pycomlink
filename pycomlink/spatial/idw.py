from __future__ import division
from builtins import range
from builtins import object
import numpy as np
from scipy.spatial import cKDTree as KDTree
from numba import jit


class Invdisttree(object):
    """inverse-distance-weighted interpolation using KDTree:

    Copied from http://stackoverflow.com/questions/3104781/
    inverse-distance-weighted-idw-interpolation-with-python

    Usage granted by original author here:
    https://github.com/scipy/scipy/issues/2022#issuecomment-296373506

    invdisttree = Invdisttree( X, z )  -- data points, values
    interpol = invdisttree( q, nnear=3, eps=0, p=1, weights=None, stat=0 )
        interpolates z from the 3 points nearest each query point q;
        For example, interpol[ a query point q ]
        finds the 3 data points nearest q, at distances d1 d2 d3
        and returns the IDW average of the values z1 z2 z3
            (z1/d1 + z2/d2 + z3/d3)
            / (1/d1 + 1/d2 + 1/d3)
            = .55 z1 + .27 z2 + .18 z3  for distances 1 2 3

        q may be one point, or a batch of points.
        eps: approximate nearest, dist <= (1 + eps) * true nearest
        p: use 1 / distance**p
        weights: optional multipliers for 1 / distance**p, of the same shape as q
        stat: accumulate wsum, wn for average weights

    How many nearest neighbors should one take ?
    a) start with 8 11 14 .. 28 in 2d 3d 4d .. 10d; see Wendel's formula
    b) make 3 runs with nnear= e.g. 6 8 10, and look at the results --
        |interpol 6 - interpol 8| etc., or |f - interpol*| if you have f(q).
        I find that runtimes don't increase much at all with nnear -- ymmv.

    p=1, p=2 ?
        p=2 weights nearer points more, farther points less.
        In 2d, the circles around query points have areas ~ distance**2,
        so p=2 is inverse-area weighting. For example,
            (z1/area1 + z2/area2 + z3/area3)
            / (1/area1 + 1/area2 + 1/area3)
            = .74 z1 + .18 z2 + .08 z3  for distances 1 2 3
        Similarly, in 3d, p=3 is inverse-volume weighting.

    Scaling:
        if different X coordinates measure different things, Euclidean distance
        can be way off.  For example, if X0 is in the range 0 to 1
        but X1 0 to 1000, the X1 distances will swamp X0;
        rescale the data, i.e. make X0.std() ~= X1.std() .

    A nice property of IDW is that it's scale-free around query points:
    if I have values z1 z2 z3 from 3 points at distances d1 d2 d3,
    the IDW average
        (z1/d1 + z2/d2 + z3/d3)
        / (1/d1 + 1/d2 + 1/d3)
    is the same for distances 1 2 3, or 10 20 30 -- only the ratios matter.
    In contrast, the commonly-used Gaussian kernel exp( - (distance/h)**2 )
    is exceedingly sensitive to distance and to h.

    """

    # anykernel( dj / av dj ) is also scale-free
    # error analysis, |f(x) - idw(x)| ? todo: regular grid, nnear ndim+1, 2*ndim

    def __init__(self, X, leafsize=10, stat=0):
        self.X = X
        self.tree = KDTree(X, leafsize=leafsize)  # build the tree
        self.stat = stat
        self.wn = 0
        self.wsum = None
        self.q = None

    def __call__(self, q, z, nnear=6, eps=0, p=1, weights=None, max_distance=None):
        # nnear nearest neighbours of each query point --
        assert len(self.X) == len(z), "len(X) %d != len(z) %d" % (len(self.X), len(z))

        if max_distance is None:
            max_distance = np.inf

        if nnear <= 1:
            raise ValueError("`nnear` must be greater than 1")

        q = np.asarray(q)
        z = np.asarray(z)
        qdim = q.ndim
        if qdim == 1:
            q = np.array([q])
        if self.wsum is None:
            self.wsum = np.zeros(nnear)

        # Do not recalculate the distance matrix if it has already
        # been calculated for the same parameters
        if (
            hasattr(self, "q")
            and np.array_equal(q, self.q)
            and hasattr(self, "nnear")
            and np.array_equal(nnear, self.nnear)
            and hasattr(self, "eps")
            and np.array_equal(eps, self.eps)
        ):
            # Do nothing here
            # print 'reuse `distances`'
            pass
        else:
            self.distances, self.ix = self.tree.query(
                q, k=nnear, eps=eps, distance_upper_bound=max_distance
            )
            self.q = q
            self.nnear = nnear
            self.eps = eps

        self.z = z

        if weights is None:
            weights = np.ones(len(q))

        interpol, self.wn, self.wsum = _numba_idw_loop(
            distances=self.distances,
            ixs=self.ix,
            z=self.z,
            z_shape=z[0].shape,
            p=p,
            weights=weights,
            wn=self.wn,
            wsum=self.wsum,
        )

        return interpol if qdim > 1 else interpol[0]


@jit(nopython=True)
def _numba_idw_loop(distances, ixs, z, z_shape, p, weights, wn, wsum):
    interpol = np.zeros((len(distances),) + z_shape)
    jinterpol = 0
    for i in range(len(distances)):
        dist = distances[i]
        ix = ixs[i]
        if dist[0] < 1e-10:
            wz = z[ix[0]]
        else:  # weight z s by 1/dist --
            w = 1 / dist ** p
            w *= weights[ix]  # >= 0
            w /= np.sum(w)
            wz = np.dot(w, z[ix])
            wn += 1
            wsum += w
        interpol[jinterpol] = wz
        jinterpol += 1
    return interpol, wn, wsum
