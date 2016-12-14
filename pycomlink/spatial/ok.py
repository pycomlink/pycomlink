#----------------------------------------------------------------------------
# Name:
# Purpose:
#
# Authors: Felix Keis, Christian Chwala
#
# Created:
# Copyright:    (c) Christian Chwala 2016
# Licence:      The MIT License
#----------------------------------------------------------------------------


from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals

import numpy as np
from scipy.optimize import minimize
import scipy

import matplotlib.pyplot as plt


def kriging(x, y, z, xgrid, ygrid, n_closest_points, **kwargs):
    """Calculate Ordinary Kriging interpolation

    Parameters
    ----------

    TODO: THIS DOC IS OUTDATED!!!!!!!!!!!!!!!!

    sample_points : iterable of floats
                    Locations of sample points (Lon/Lat)

                    Example
                    -------
                    >>> x = np.random.rand(number_of_points)
                        y = np.random.rand(number_of_points)
                        sample_points = np.vstack((x,y)).T

    sample_values : iterable of floats
                    Values at sample_points
    grid : iterable of floats
            Gridpoint locations

                    Example
                    -------
                    >>> xcoords = np.arange(xstart, xstop, dx)
                        ycoords = np.arange(ystart, ystop, dy)
                        xgrid, ygrid = np.meshgrid(xcoords, ycoords)
                        xi, yi = xgrid.flatten(), ygrid.flatten()
                        grid = np.vstack((xi, yi)).T
    n_closest_points : int
                Parameters for Kriging interpolation. See OrdinaryKriging
                documentation for information.
    kwargs : kriging parameters, optional
                See https://github.com/bsmurphy/PyKrige for details

    Returns
    -------
    array of floats
        Interpolated values at grid points

    """

    try:
        OK = OrdinaryKriging(x, y, z,
                             verbose=False, enable_plotting=False,
                             **kwargs)

        z, s_kr = OK.execute(xpoints=xgrid.flatten(),
                             ypoints=ygrid.flatten(),
                             style='points',
                             backend='loop',
                             n_closest_points=n_closest_points)
        z = np.reshape(z, xgrid.shape)
    except ValueError:
        pass
        z = np.zeros_like(xgrid)

    return z


def adjust_for_anisotropy(x, y, xcenter, ycenter, scaling, angle):
    """Helper function for Kriging Interpolation:
       Adjusts data coordinates to take into account anisotropy.
    Can also be used to take into account data scaling."""

    x -= xcenter
    y -= ycenter
    xshape = x.shape
    yshape = y.shape
    x = x.flatten()
    y = y.flatten()

    coords = np.vstack((x, y))
    stretch = np.array([[1, 0], [0, scaling]])
    rotate = np.array([[np.cos(-angle * np.pi/180.0), -np.sin(-angle * np.pi/180.0)],
                       [np.sin(-angle * np.pi/180.0), np.cos(-angle * np.pi/180.0)]])
    rotated_coords = np.dot(stretch, np.dot(rotate, coords))
    x = rotated_coords[0, :].reshape(xshape)
    y = rotated_coords[1, :].reshape(yshape)
    x += xcenter
    y += ycenter

    return x, y


def adjust_for_anisotropy_3d(x, y, z, xcenter, ycenter, zcenter, scaling_y,
                             scaling_z, angle_x, angle_y, angle_z):
    """Helper function for Kriging Interpolation:
       Adjusts data coordinates to take into account anisotropy.
    Can also be used to take into account data scaling."""

    x -= xcenter
    y -= ycenter
    z -= zcenter
    xshape = x.shape
    yshape = y.shape
    zshape = z.shape
    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    coords = np.vstack((x, y, z))
    stretch = np.array([[1., 0., 0.], [0., scaling_y, 0.], [0., 0., scaling_z]])
    rotate_x = np.array([[1., 0., 0.],
                         [0., np.cos(-angle_x * np.pi/180.), -np.sin(-angle_x * np.pi/180.)],
                         [0., np.sin(-angle_x * np.pi/180.), np.cos(-angle_x * np.pi/180.)]])
    rotate_y = np.array([[np.cos(-angle_y * np.pi/180.), 0., np.sin(-angle_y * np.pi/180.)],
                         [0., 1., 0.],
                         [-np.sin(-angle_y * np.pi/180.), 0., np.cos(-angle_y * np.pi/180.)]])
    rotate_z = np.array([[np.cos(-angle_z * np.pi/180.), -np.sin(-angle_z * np.pi/180.), 0.],
                         [np.sin(-angle_z * np.pi/180.), np.cos(-angle_z * np.pi/180.), 0.],
                         [0., 0., 1.]])
    rot_tot = np.dot(rotate_z, np.dot(rotate_y, rotate_x))
    rotated_coords = np.dot(stretch, np.dot(rot_tot, coords))
    x = rotated_coords[0, :].reshape(xshape)
    y = rotated_coords[1, :].reshape(yshape)
    z = rotated_coords[2, :].reshape(zshape)
    x += xcenter
    y += ycenter
    z += zcenter

    return x, y, z


def initialize_variogram_model(x, y, z, variogram_model, variogram_model_parameters,
                               variogram_function, nlags, weight):
    """Helper function for Kriging Interpolation:
       Initializes the variogram model for kriging according
    to user specifications or to defaults"""

    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    z1, z2 = np.meshgrid(z, z)

    dx = x1 - x2
    dy = y1 - y2
    dz = z1 - z2
    d = np.sqrt(dx**2 + dy**2)
    g = 0.5 * dz**2

    indices = np.indices(d.shape)
    d = d[(indices[0, :, :] > indices[1, :, :])]
    g = g[(indices[0, :, :] > indices[1, :, :])]

    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin)/nlags
    bins = [dmin + n*dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    lags = np.zeros(nlags)
    semivariance = np.zeros(nlags)

    for n in range(nlags):

        if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
            lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
            semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
        else:
            lags[n] = np.nan
            semivariance[n] = np.nan

    lags = lags[~np.isnan(semivariance)]
    semivariance = semivariance[~np.isnan(semivariance)]

    if variogram_model_parameters is not None:
        if variogram_model == 'linear' and len(variogram_model_parameters) != 2:
            raise ValueError("Exactly two parameters required "
                             "for linear variogram model")
        elif (variogram_model == 'power' or variogram_model == 'spherical' or variogram_model == 'exponential'
              or variogram_model == 'gaussian') and len(variogram_model_parameters) != 3:
            raise ValueError("Exactly three parameters required "
                             "for %s variogram model" % variogram_model)
    else:
        if variogram_model == 'custom':
            raise ValueError("Variogram parameters must be specified when implementing custom variogram model.")
        else:
            variogram_model_parameters = calculate_variogram_model(lags, semivariance, variogram_model,
                                                                   variogram_function, weight)

    return lags, semivariance, variogram_model_parameters


def initialize_variogram_model_3d(x, y, z, values, variogram_model, variogram_model_parameters,
                                  variogram_function, nlags, weight):
    """Helper function for Kriging Interpolation:
       Initializes the variogram model for kriging according
    to user specifications or to defaults"""

    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    z1, z2 = np.meshgrid(z, z)
    val1, val2 = np.meshgrid(values, values)
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    g = 0.5 * (val1 - val2)**2

    indices = np.indices(d.shape)
    d = d[(indices[0, :, :] > indices[1, :, :])]
    g = g[(indices[0, :, :] > indices[1, :, :])]

    dmax = np.amax(d)
    dmin = np.amin(d)
    dd = (dmax - dmin)/nlags
    bins = [dmin + n*dd for n in range(nlags)]
    dmax += 0.001
    bins.append(dmax)

    lags = np.zeros(nlags)
    semivariance = np.zeros(nlags)

    for n in range(nlags):

        if d[(d >= bins[n]) & (d < bins[n + 1])].size > 0:
            lags[n] = np.mean(d[(d >= bins[n]) & (d < bins[n + 1])])
            semivariance[n] = np.mean(g[(d >= bins[n]) & (d < bins[n + 1])])
        else:
            lags[n] = np.nan
            semivariance[n] = np.nan

    lags = lags[~np.isnan(semivariance)]
    semivariance = semivariance[~np.isnan(semivariance)]

    if variogram_model_parameters is not None:
        if variogram_model == 'linear' and len(variogram_model_parameters) != 2:
            raise ValueError("Exactly two parameters required "
                             "for linear variogram model")
        elif (variogram_model == 'power' or variogram_model == 'spherical' or variogram_model == 'exponential'
              or variogram_model == 'gaussian') and len(variogram_model_parameters) != 3:
            raise ValueError("Exactly three parameters required "
                             "for %s variogram model" % variogram_model)
    else:
        if variogram_model == 'custom':
            raise ValueError("Variogram parameters must be specified when implementing custom variogram model.")
        else:
            variogram_model_parameters = calculate_variogram_model(lags, semivariance, variogram_model,
                                                                   variogram_function, weight)

    return lags, semivariance, variogram_model_parameters


def variogram_function_error(params, x, y, variogram_function, weight):
    """Helper function for Kriging Interpolation:
       Function used to in fitting of variogram model.
    Returns RMSE between calculated fit and actual data."""

    diff = variogram_function(params, x) - y

    if weight:
        weights = np.arange(x.size, 0.0, -1.0)
        weights /= np.sum(weights)
        rmse = np.sqrt(np.average(diff**2, weights=weights))
    else:
        rmse = np.sqrt(np.mean(diff**2))

    return rmse


def calculate_variogram_model(lags, semivariance, variogram_model, variogram_function, weight):
    """Helper function for Kriging Interpolation:
       Function that fits a variogram model when parameters are not specified."""

    if variogram_model == 'linear':
        x0 = [(np.amax(semivariance) - np.amin(semivariance))/(np.amax(lags) - np.amin(lags)),
              np.amin(semivariance)]
        bnds = ((0.0, 1000000000.0), (0.0, np.amax(semivariance)))
    elif variogram_model == 'power':
        x0 = [(np.amax(semivariance) - np.amin(semivariance))/(np.amax(lags) - np.amin(lags)),
              1.1, np.amin(semivariance)]
        bnds = ((0.0, 1000000000.0), (0.01, 1.99), (0.0, np.amax(semivariance)))
    else:
        x0 = [np.amax(semivariance), 0.5*np.amax(lags), np.amin(semivariance)]
        bnds = ((0.0, 10*np.amax(semivariance)), (0.0, np.amax(lags)), (0.0, np.amax(semivariance)))

    res = minimize(variogram_function_error, x0, args=(lags, semivariance, variogram_function, weight),
                   method='SLSQP', bounds=bnds)

    return res.x


def krige(x, y, z, coords, variogram_function, variogram_model_parameters):
    """Helper function for Kriging Interpolation:
       Sets up and solves the kriging matrix for the given coordinate pair.
        This function is now only used for the statistics calculations."""

    zero_index = None
    zero_value = False

    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2)
    bd = np.sqrt((x - coords[0])**2 + (y - coords[1])**2)
    if np.any(np.absolute(bd) <= 1e-10):
        zero_value = True
        zero_index = np.where(bd <= 1e-10)[0][0]

    n = x.shape[0]
    a = np.zeros((n+1, n+1))
    a[:n, :n] = - variogram_function(variogram_model_parameters, d)
    np.fill_diagonal(a, 0.0)
    a[n, :] = 1.0
    a[:, n] = 1.0
    a[n, n] = 0.0

    b = np.zeros((n+1, 1))
    b[:n, 0] = - variogram_function(variogram_model_parameters, bd)
    if zero_value:
        b[zero_index, 0] = 0.0
    b[n, 0] = 1.0

    x_ = np.linalg.solve(a, b)
    zinterp = np.sum(x_[:n, 0] * z)
    sigmasq = np.sum(x_[:, 0] * -b[:, 0])

    return zinterp, sigmasq


def krige_3d(x, y, z, vals, coords, variogram_function, variogram_model_parameters):   
    """Helper function for Kriging Interpolation:
       Sets up and solves the kriging matrix for the given coordinate pair.
        This function is now only used for the statistics calculations."""

    zero_index = None
    zero_value = False

    x1, x2 = np.meshgrid(x, x)
    y1, y2 = np.meshgrid(y, y)
    z1, z2 = np.meshgrid(z, z)
    d = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    bd = np.sqrt((x - coords[0])**2 + (y - coords[1])**2 + (z - coords[2])**2)
    if np.any(np.absolute(bd) <= 1e-10):
        zero_value = True
        zero_index = np.where(bd <= 1e-10)[0][0]

    n = x.shape[0]
    a = np.zeros((n+1, n+1))
    a[:n, :n] = - variogram_function(variogram_model_parameters, d)
    np.fill_diagonal(a, 0.0)
    a[n, :] = 1.0
    a[:, n] = 1.0
    a[n, n] = 0.0

    b = np.zeros((n+1, 1))
    b[:n, 0] = - variogram_function(variogram_model_parameters, bd)
    if zero_value:
        b[zero_index, 0] = 0.0
    b[n, 0] = 1.0

    x_ = np.linalg.solve(a, b)
    zinterp = np.sum(x_[:n, 0] * vals)
    sigmasq = np.sum(x_[:, 0] * -b[:, 0])

    return zinterp, sigmasq


def find_statistics(x, y, z, variogram_function, variogram_model_parameters):
    
    """Helper function for Kriging Interpolation:
          Calculates variogram fit statistics."""

    delta = np.zeros(z.shape)
    sigma = np.zeros(z.shape)

    for n in range(z.shape[0]):
        if n == 0:
            delta[n] = 0.0
            sigma[n] = 0.0
        else:
            z_, ss_ = krige(x[:n], y[:n], z[:n], (x[n], y[n]), variogram_function, variogram_model_parameters)
            d = z[n] - z_
            delta[n] = d
            sigma[n] = np.sqrt(ss_)

    delta = delta[1:]
    sigma = sigma[1:]
    epsilon = delta/sigma

    return delta, sigma, epsilon


def find_statistics_3d(x, y, z, vals, variogram_function, variogram_model_parameters):
    """Helper function for Kriging Interpolation:
       Calculates variogram fit statistics for 3D problems."""

    delta = np.zeros(vals.shape)
    sigma = np.zeros(vals.shape)

    for n in range(z.shape[0]):
        if n == 0:
            delta[n] = 0.0
            sigma[n] = 0.0
        else:
            z_, ss_ = krige_3d(x[:n], y[:n], z[:n], vals[:n], (x[n], y[n], z[n]),
                               variogram_function, variogram_model_parameters)
            d = z[n] - z_
            delta[n] = d
            sigma[n] = np.sqrt(ss_)

    delta = delta[1:]
    sigma = sigma[1:]
    epsilon = delta/sigma

    return delta, sigma, epsilon


def calcQ1(epsilon):
    """Helper function for Kriging Interpolation"""
    return abs(np.sum(epsilon)/(epsilon.shape[0] - 1))


def calcQ2(epsilon):
    """Helper function for Kriging Interpolation"""
    return np.sum(epsilon**2)/(epsilon.shape[0] - 1)


def calc_cR(Q2, sigma):
    """Helper function for Kriging Interpolation"""
    return Q2 * np.exp(np.sum(np.log(sigma**2))/sigma.shape[0])


def linear_variogram_model(params, dist):
    """Helper function for Kriging Interpolation"""
    return float(params[0])*dist + float(params[1])


def power_variogram_model(params, dist):
    """Helper function for Kriging Interpolation"""
    return float(params[0])*(dist**float(params[1])) + float(params[2])


def gaussian_variogram_model(params, dist):
    """Helper function for Kriging Interpolation"""
    return (float(params[0]) - float(params[2]))*(1 - np.exp(-dist**2/(float(params[1])*4.0/7.0)**2)) + \
            float(params[2])


def exponential_variogram_model(params, dist):
    """Helper function for Kriging Interpolation"""
    return (float(params[0]) - float(params[2]))*(1 - np.exp(-dist/(float(params[1])/3.0))) + \
            float(params[2])


def spherical_variogram_model(params, dist):
    """Helper function for Kriging Interpolation"""
    return np.piecewise(dist, [dist <= float(params[1]), dist > float(params[1])],
                        [lambda x: (float(params[0]) - float(params[2])) *
                                   ((3*x)/(2*float(params[1])) - (x**3)/(2*float(params[1])**3)) + float(params[2]),
                         float(params[0])])



class OrdinaryKriging:
    """class OrdinaryKriging Convenience class for easy access to 2D Ordinary Kriging
    
    Parameters
    ----------
        X : array-like
            X-coordinates of data points.
        Y : array-like
            Y-coordinates of data points.
        Z : array-like
            Values at data points.
        variogram_model : string, optional
            Specified which variogram model to use;
            may be one of the following: linear, power, gaussian, spherical,
            exponential. Default is linear variogram model. To utilize as custom variogram
            model, specify 'custom'; you must also provide variogram_parameters and
            variogram_function.
        variogram_parameters : list, optional
            Parameters that define the
            specified variogram model. If not provided, parameters will be automatically
            calculated such that the root-mean-square error for the fit variogram
            function is minimized.
                linear - [slope, nugget]
                power - [scale, exponent, nugget]
                gaussian - [sill, range, nugget]
                spherical - [sill, range, nugget]
                exponential - [sill, range, nugget]
            For a custom variogram model, the parameters are required, as custom variogram
            models currently will not automatically be fit to the data. The code does not
            check that the provided list contains the appropriate number of parameters for
            the custom variogram model, so an incorrect parameter list in such a case will
            probably trigger an esoteric exception someplace deep in the code.
        variogram_function : callable, optional
            A callable function that must be provided
            if variogram_model is specified as 'custom'. The function must take only two
            arguments: first, a list of parameters for the variogram model; second, the
            distances at which to calculate the variogram model. The list provided in
            variogram_parameters will be passed to the function as the first argument.
        nlags : int, optional
            Number of averaging bins for the semivariogram.
            Default is 6.
        weight : boolean, optional
            Flag that specifies if semivariance at smaller lags
            should be weighted more heavily when automatically calculating variogram model.
            True indicates that weights will be applied. Default is False.
            (Kitanidis suggests that the values at smaller lags are more important in
            fitting a variogram model, so the option is provided to enable such weighting.)
        anisotropy_scaling : float, optional
            Scalar stretching value to take
            into account anisotropy. Default is 1 (effectively no stretching).
            Scaling is applied in the y-direction in the rotated data frame
            (i.e., after adjusting for the anisotropy_angle, if anisotropy_angle
            is not 0).
        anisotropy_angle : float, optional
            CCW angle (in degrees) by which to
            rotate coordinate system in order to take into account anisotropy.
            Default is 0 (no rotation). Note that the coordinate system is rotated.
        verbose : Boolean, optional
            Enables program text output to monitor
            kriging process. Default is False (off).
        enable_plotting : Boolean, optional
            Enables plotting to display
            variogram. Default is False (off).
        enable_statistics : Boolean, optional
            Default is False

    Note
    ----
    This code was adopted from the PyKrige modul (https://github.com/bsmurphy/PyKrige)    
    
    
    """

    eps = 1.e-10   # Cutoff for comparison to zero
    variogram_dict = {'linear': linear_variogram_model,
                      'power': power_variogram_model,
                      'gaussian': gaussian_variogram_model,
                      'spherical': spherical_variogram_model,
                      'exponential': exponential_variogram_model}

    def __init__(self, x, y, z, variogram_model='linear', variogram_parameters=None,
                 variogram_function=None, nlags=6, weight=False, anisotropy_scaling=1.0,
                 anisotropy_angle=0.0, verbose=False, enable_plotting=False,
                 enable_statistics=False):

        # Code assumes 1D input arrays. Ensures that any extraneous dimensions
        # don't get in the way. Copies are created to avoid any problems with
        # referencing the original passed arguments.
        self.X_ORIG = np.atleast_1d(np.squeeze(np.array(x, copy=True)))
        self.Y_ORIG = np.atleast_1d(np.squeeze(np.array(y, copy=True)))
        self.Z = np.atleast_1d(np.squeeze(np.array(z, copy=True)))

        self.verbose = verbose
        self.enable_plotting = enable_plotting
        if self.enable_plotting and self.verbose:
            print("Plotting Enabled\n")

        self.XCENTER = (np.amax(self.X_ORIG) + np.amin(self.X_ORIG))/2.0
        self.YCENTER = (np.amax(self.Y_ORIG) + np.amin(self.Y_ORIG))/2.0
        self.anisotropy_scaling = anisotropy_scaling
        self.anisotropy_angle = anisotropy_angle
        if self.verbose:
            print("Adjusting data for anisotropy...")
        self.X_ADJUSTED, self.Y_ADJUSTED = \
            adjust_for_anisotropy(np.copy(self.X_ORIG), np.copy(self.Y_ORIG),
                                       self.XCENTER, self.YCENTER,
                                       self.anisotropy_scaling, self.anisotropy_angle)

        self.variogram_model = variogram_model
        if self.variogram_model not in self.variogram_dict.keys() and self.variogram_model != 'custom':
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                raise ValueError("Must specify callable function for custom variogram model.")
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]
        if self.verbose:
            print("Initializing variogram model...")
        self.lags, self.semivariance, self.variogram_model_parameters = \
            initialize_variogram_model(self.X_ADJUSTED, self.Y_ADJUSTED, self.Z,
                                            self.variogram_model, variogram_parameters,
                                            self.variogram_function, nlags, weight)
        if self.verbose:
            if self.variogram_model == 'linear':
                print("Using '%s' Variogram Model" % 'linear')
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], '\n')
            elif self.variogram_model == 'power':
                print("Using '%s' Variogram Model" % 'power')
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
            elif self.variogram_model == 'custom':
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Sill:", self.variogram_model_parameters[0])
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        if enable_statistics:
            self.delta, self.sigma, self.epsilon = find_statistics(self.X_ADJUSTED, self.Y_ADJUSTED,
                                                                        self.Z, self.variogram_function,
                                                                        self.variogram_model_parameters)
            self.Q1 = calcQ1(self.epsilon)
            self.Q2 = calcQ2(self.epsilon)
            self.cR = calc_cR(self.Q2, self.sigma)
            if self.verbose:
                print("Q1 =", self.Q1)
                print("Q2 =", self.Q2)
                print("cR =", self.cR, '\n')
        else:
            self.delta, self.sigma, self.epsilon, self.Q1, self.Q2, self.cR = [None]*6

    def update_variogram_model(self, variogram_model, variogram_parameters=None,
                               variogram_function=None, nlags=6, weight=False,
                               anisotropy_scaling=1.0, anisotropy_angle=0.0):
        """Allows user to update variogram type and/or variogram model parameters."""

        if anisotropy_scaling != self.anisotropy_scaling or \
           anisotropy_angle != self.anisotropy_angle:
            if self.verbose:
                print("Adjusting data for anisotropy...")
            self.anisotropy_scaling = anisotropy_scaling
            self.anisotropy_angle = anisotropy_angle
            self.X_ADJUSTED, self.Y_ADJUSTED = \
                adjust_for_anisotropy(np.copy(self.X_ORIG),
                                           np.copy(self.Y_ORIG),
                                           self.XCENTER, self.YCENTER,
                                           self.anisotropy_scaling,
                                           self.anisotropy_angle)

        self.variogram_model = variogram_model
        if self.variogram_model not in self.variogram_dict.keys() and self.variogram_model != 'custom':
            raise ValueError("Specified variogram model '%s' is not supported." % variogram_model)
        elif self.variogram_model == 'custom':
            if variogram_function is None or not callable(variogram_function):
                raise ValueError("Must specify callable function for custom variogram model.")
            else:
                self.variogram_function = variogram_function
        else:
            self.variogram_function = self.variogram_dict[self.variogram_model]
        if self.verbose:
            print("Updating variogram mode...")
        self.lags, self.semivariance, self.variogram_model_parameters = \
            initialize_variogram_model(self.X_ADJUSTED, self.Y_ADJUSTED, self.Z,
                                            self.variogram_model, variogram_parameters,
                                            self.variogram_function, nlags, weight)
        if self.verbose:
            if self.variogram_model == 'linear':
                print("Using '%s' Variogram Model" % 'linear')
                print("Slope:", self.variogram_model_parameters[0])
                print("Nugget:", self.variogram_model_parameters[1], '\n')
            elif self.variogram_model == 'power':
                print("Using '%s' Variogram Model" % 'power')
                print("Scale:", self.variogram_model_parameters[0])
                print("Exponent:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
            elif self.variogram_model == 'custom':
                print("Using Custom Variogram Model")
            else:
                print("Using '%s' Variogram Model" % self.variogram_model)
                print("Sill:", self.variogram_model_parameters[0])
                print("Range:", self.variogram_model_parameters[1])
                print("Nugget:", self.variogram_model_parameters[2], '\n')
        if self.enable_plotting:
            self.display_variogram_model()

        if self.verbose:
            print("Calculating statistics on variogram model fit...")
        self.delta, self.sigma, self.epsilon = find_statistics(self.X_ADJUSTED, self.Y_ADJUSTED,
                                                                    self.Z, self.variogram_function,
                                                                    self.variogram_model_parameters)
        self.Q1 = calcQ1(self.epsilon)
        self.Q2 = calcQ2(self.epsilon)
        self.cR = calc_cR(self.Q2, self.sigma)
        if self.verbose:
            print("Q1 =", self.Q1)
            print("Q2 =", self.Q2)
            print("cR =", self.cR, '\n')

    def display_variogram_model(self):
        """Displays variogram model with the actual binned data"""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(self.lags, self.semivariance, 'r*')
        ax.plot(self.lags,
                self.variogram_function(self.variogram_model_parameters, self.lags), 'k-')
        plt.show()

    def switch_verbose(self):
        """Allows user to switch code talk-back on/off. Takes no arguments."""
        self.verbose = not self.verbose

    def switch_plotting(self):
        """Allows user to switch plot display on/off. Takes no arguments."""
        self.enable_plotting = not self.enable_plotting

    def get_epsilon_residuals(self):
        """Returns the epsilon residuals for the variogram fit."""
        return self.epsilon

    def plot_epsilon_residuals(self):
        """Plots the epsilon residuals for the variogram fit."""
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(range(self.epsilon.size), self.epsilon, c='k', marker='*')
        ax.axhline(y=0.0)
        plt.show()

    def get_statistics(self):
        return self.Q1, self.Q2, self.cR

    def print_statistics(self):
        print("Q1 =", self.Q1)
        print("Q2 =", self.Q2)
        print("cR =", self.cR)

    def _get_kriging_matrix(self, n):
        """Assembles the kriging matrix."""

        xy = np.concatenate((self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1)
        d = scipy.spatial.distance.cdist(xy, xy, 'euclidean')
        a = np.zeros((n+1, n+1))
        a[:n, :n] = - self.variogram_function(self.variogram_model_parameters, d)
        np.fill_diagonal(a, 0.)
        a[n, :] = 1.0
        a[:, n] = 1.0
        a[n, n] = 0.0

        return a

    def _exec_loop(self, a, bd_all, mask):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""

        npt = bd_all.shape[0]
        n = self.X_ADJUSTED.shape[0]
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        a_inv = scipy.linalg.inv(a)

        for j in np.nonzero(~mask)[0]:   # Note that this is the same thing as range(npt) if mask is not defined,
            bd = bd_all[j]               # otherwise it takes the non-masked elements.
            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False

            b = np.zeros((n+1, 1))
            b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0
            x = np.dot(a_inv, b)
            zvalues[j] = np.sum(x[:n, 0] * self.Z)
            sigmasq[j] = np.sum(x[:, 0] * -b[:, 0])

        return zvalues, sigmasq

    def _exec_loop_moving_window(self, a_all, bd_all, mask, bd_idx):
        """Solves the kriging system by looping over all specified points.
        Less memory-intensive, but involves a Python-level loop."""
        import scipy.linalg.lapack

        npt = bd_all.shape[0]
        n = bd_idx.shape[1]
        zvalues = np.zeros(npt)
        sigmasq = np.zeros(npt)

        for i in np.nonzero(~mask)[0]:   # Note that this is the same thing as range(npt) if mask is not defined,
            b_selector = bd_idx[i]       # otherwise it takes the non-masked elements.
            bd = bd_all[i]

            a_selector = np.concatenate((b_selector, np.array([a_all.shape[0] - 1])))
            a = a_all[a_selector[:, None], a_selector]

            if np.any(np.absolute(bd) <= self.eps):
                zero_value = True
                zero_index = np.where(np.absolute(bd) <= self.eps)
            else:
                zero_index = None
                zero_value = False
            b = np.zeros((n+1, 1))
            b[:n, 0] = - self.variogram_function(self.variogram_model_parameters, bd)
            if zero_value:
                b[zero_index[0], 0] = 0.0
            b[n, 0] = 1.0

            x = scipy.linalg.solve(a, b)

            zvalues[i] = x[:n, 0].dot(self.Z[b_selector])
            sigmasq[i] = - x[:, 0].dot(b[:, 0])

        return zvalues, sigmasq

    def execute(self, style, xpoints, ypoints, mask=None, n_closest_points=None):
        """
        Parameters
        ----------
            style : string
                Specifies how to treat input kriging points.
                Specifying 'grid' treats xpoints and ypoints as two arrays of
                x and y coordinates that define a rectangular grid.
                Specifying 'points' treats xpoints and ypoints as two arrays
                that provide coordinate pairs at which to solve the kriging system.
                Specifying 'masked' treats xpoints and ypoints as two arrays of
                x and y coordinates that define a rectangular grid and uses mask
                to only evaluate specific points in the grid.
            xpoints : array-like
                If style is specific as 'grid' or 'masked',
                x-coordinates of MxN grid. If style is specified as 'points',
                x-coordinates of specific points at which to solve kriging system.
            ypoints : array-like
                If style is specified as 'grid' or 'masked',
                y-coordinates of MxN grid. If style is specified as 'points',
                y-coordinates of specific points at which to solve kriging system.
                Note that in this case, xpoints and ypoints must have the same dimensions
                (i.e., M = N).
            mask : boolean array, optional
                Specifies the points in the rectangular
                grid defined by xpoints and ypoints that are to be excluded in the
                kriging calculations. Must be provided if style is specified as 'masked'.
                False indicates that the point should not be masked, so the kriging system
                will be solved at the point.
                True indicates that the point should be masked, so the kriging system should
                will not be solved at the point.
            n_closest_points : int, optional
                For kriging with a moving window, specifies the number
                of nearby points to use in the calculation. This can speed up the calculation for large
                datasets, but should be used with caution. As Kitanidis notes, kriging with a moving
                window can produce unexpected oddities if the variogram model is not carefully chosen.
                
        Returns
        -------
            zvalues numpy array
                Z-values of specified grid or at the
                specified set of points. If style was specified as 'masked', zvalues will
                be a numpy masked array.
            sigmasq : numpy array
                Variance at specified grid points or
                at the specified set of points. If style was specified as 'masked', sigmasq
                will be a numpy masked array.
        """

        if self.verbose:
            print("Executing Ordinary Kriging...\n")

        if style != 'grid' and style != 'masked' and style != 'points':
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        xpts = np.atleast_1d(np.squeeze(np.array(xpoints, copy=True)))
        ypts = np.atleast_1d(np.squeeze(np.array(ypoints, copy=True)))
        n = self.X_ADJUSTED.shape[0]
        nx = xpts.size
        ny = ypts.size
        a = self._get_kriging_matrix(n)

        if style in ['grid', 'masked']:
            if style == 'masked':
                if mask is None:
                    raise IOError("Must specify boolean masking array when style is 'masked'.")
                if mask.shape[0] != ny or mask.shape[1] != nx:
                    if mask.shape[0] == nx and mask.shape[1] == ny:
                        mask = mask.T
                    else:
                        raise ValueError("Mask dimensions do not match specified grid dimensions.")
                mask = mask.flatten()
            npt = ny*nx
            grid_x, grid_y = np.meshgrid(xpts, ypts)
            xpts = grid_x.flatten()
            ypts = grid_y.flatten()

        elif style == 'points':
            if xpts.size != ypts.size:
                raise ValueError("xpoints and ypoints must have same dimensions "
                                 "when treated as listing discrete points.")
            npt = nx
        else:
            raise ValueError("style argument must be 'grid', 'points', or 'masked'")

        xpts, ypts = adjust_for_anisotropy(xpts, ypts, self.XCENTER, self.YCENTER,
                                                self.anisotropy_scaling, self.anisotropy_angle)

        if style != 'masked':
            mask = np.zeros(npt, dtype='bool')

        xy_points = np.concatenate((xpts[:, np.newaxis], ypts[:, np.newaxis]), axis=1)
        xy_data = np.concatenate((self.X_ADJUSTED[:, np.newaxis], self.Y_ADJUSTED[:, np.newaxis]), axis=1)

        if n_closest_points is not None:
            from scipy.spatial import cKDTree
            tree = cKDTree(xy_data)
            bd, bd_idx = tree.query(xy_points, k=n_closest_points, eps=0.0)

            zvalues, sigmasq = self._exec_loop_moving_window(a, bd, mask, bd_idx)
            
        else:
            bd = scipy.spatial.distance.cdist(xy_points,  xy_data, 'euclidean')
            zvalues, sigmasq = self._exec_loop(a, bd, mask)
            
        if style == 'masked':
            zvalues = np.ma.array(zvalues, mask=mask)
            sigmasq = np.ma.array(sigmasq, mask=mask)

        if style in ['masked', 'grid']:
            zvalues = zvalues.reshape((ny, nx))
            sigmasq = sigmasq.reshape((ny, nx))

        return zvalues, sigmasq


