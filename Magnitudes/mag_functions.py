import numpy as np

"""
Created on 1st March 2020 by Antony Butcher

This module contains a series of functions required to
model source spectra and estimate MW and ML 

"""

def latlong_distn(lat1, long1, lat2, long2):
    """Calculates the distance using latitude and longitude coordinates"""
    # Convert latitude and longitude to
    # spherical coordinates in radians.
    degrees_to_radians = np.pi/180.0

    # phi = 90 - latitude
    phi1 = (90.0 - lat1)*degrees_to_radians
    phi2 = (90.0 - lat2)*degrees_to_radians

    # theta = longitude
    theta1 = long1*degrees_to_radians
    theta2 = long2*degrees_to_radians

    # Compute spherical distance from spherical coordinates.

    # For two locations in spherical coordinates
    # (1, theta, phi) and (1, theta', phi')
    # cosine( arc length ) =
    #    sin phi sin phi' cos(theta-theta') + cos phi cos phi'
    # distance = rho * arc length

    cos = (np.sin(phi1)*np.sin(phi2)*np.cos(theta1 - theta2) +
           np.cos(phi1)*np.cos(phi2))
    arc = np.arccos( cos )

    # Remember to multiply arc by the radius of the earth
    # in your favorite set of units to get length.
    return (arc*6378.137)


def brune(X,Q,fc):
    """Function for inverting Q and fc using a Brune source model"""
    tt=X[0]
    ff=X[1]
    omega_0=X[2]
    return (omega_0*np.exp((-1.*np.pi*tt*ff)/Q))/(1.+(ff/fc)**2.)

def brune_kappa(X,Q,fc):
    """Function for inverting Q and fc using a Brune source model which includes kappa"""
    tt=X[0]
    ff=X[1]
    omega_0=X[2]
    kappa=X[3]
#     kappa=0.05
    return np.log10((omega_0*np.exp((-1.*np.pi*tt*ff)/Q))/(1.+(ff/fc)**2.)*np.exp(-ff*np.pi*kappa))

def ml_nol(a,r):
    """Equation to calculte local magnitude using the NOL scale"""
    if r<=17:
        mag=(np.log10(a))+(1.17*np.log10(r))+(0.0514*r)-3
    elif r>17:
        mag=(np.log10(a))+(1.11*np.log10(r))+(0.00189*r)-2.09


    return mag

def moving_average(a, n=3):
    """Fast moving average function"""
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n