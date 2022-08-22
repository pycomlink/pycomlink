import numpy as np

from pycomlink.spatial import idw


def test_idw_standard_method():
    # fmt: off

    # simple example in 1D
    x = np.array([[0, ], [1, ]])
    y = np.array([1, 0])
    interpolator = idw.Invdisttree(x)

    np.testing.assert_almost_equal(
        np.array([1. , 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.]),
        interpolator(q=np.linspace([0, ], [1, ], 11), z=y, p=1)
    )
    np.testing.assert_almost_equal(
        np.array([1. , 0.98780488, 0.94117647, 0.84482759, 0.69230769, 0.5 , 0.30769231, 0.15517241, 0.05882353, 0.01219512, 0.]),
        interpolator(q=np.linspace([0, ], [1, ], 11), z=y, p=2)
    )

    # simple example in 2D
    x = np.array([[1, -1], [1, 1], [-1, -1], [-1, 1]])
    y = np.array([1, 0, 0, 1])
    interpolator = idw.Invdisttree(x)

    # test with increasing number of points. We do this because there
    # was a bug if only 1 to 3 or so interpolation points were supplied.
    xi = np.array([[0, 0], [0.5, 0.5], [1, 0], [2, 2], [1, 1.1], [0.2, 1], [1, 0.3]])
    for i in range(len(xi)):
        np.testing.assert_almost_equal(
            interpolator(q=xi[:i + 1], z=y, p=2),
            np.array([0.5, 0.26470588, 0.5, 0.26470588, 0.00473317, 0.34256927, 0.26870145])[:i + 1]
        )

    # fmt: on
