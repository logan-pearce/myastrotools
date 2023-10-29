import numpy as np
import astropy.units as u
import astropy.constants as c
import pandas as pd

def update_progress(n,max_value):
    ''' Create a progress bar
    
    Args:
        n (int): current count
        max_value (int): ultimate values
    
    '''
    import sys
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    progress = np.round(float(n/max_value),decimals=2)
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    #if progress >= 1.:
    #    progress = 1
    #    status = "Done...\r\n"
    if n == max_value:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\r{0}% ({1} of {2}): |{3}|  {4}".format(np.round(progress*100,decimals=1), 
                                                  n, 
                                                  max_value, 
                                                  "#"*block + "-"*(barLength-block), 
                                                  status)
    sys.stdout.write(text)
    sys.stdout.flush()

def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def danby_solve(f, M0, e, h, maxnum=50):
    ''' Newton-Raphson solver for eccentricity anomaly based on "Danby" method in 
        Wisdom textbook
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
        maxnum (int): if it takes more than maxnum iterations,
            use the Mikkola solver instead.
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    import numpy as np
    from myastrotools.tools import eccentricity_anomaly
    #f = eccentricity_anomaly
    k = 0.85
    E0 = M0 + np.sign(np.sin(M0))*k*e
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    delta_D = 1
    while (delta_D > h) and number < maxnum+1: 
        fx = f(nextE,e,M0) 
        fp = (1.-e*np.cos(lastE)) 
        fpp = e*np.sin(lastE)
        fppp = e*np.cos(lastE)
        lastE = nextE
        delta_N = -fx / fp
        delta_H = -fx / (fp + 0.5*fpp*delta_N)
        delta_D = -fx / (fp + 0.5*fpp*delta_H + (1./6)*fppp*delta_H**2)
        nextE = lastE + delta_D
        number=number+1
        if number >= maxnum:
            from myastrotools.tools import mikkola_solve
            nextE = mikkola_solve(M0,e)
    return nextE

def mikkola_solve(M,e):
    ''' Analytic solver for eccentricity anomaly from Mikkola 1987. Most efficient
        when M near 0/2pi and e >= 0.95.
    Inputs: 
        M (float): mean anomaly
        e (float): eccentricity
    Returns: eccentric anomaly
        Written by Logan Pearce, 2020
    '''
    # Constants:
    alpha = (1 - e) / ((4.*e) + 0.5)
    beta = (0.5*M) / ((4.*e) + 0.5)
    ab = np.sqrt(beta**2. + alpha**3.)
    z = np.abs(beta + ab)**(1./3.)

    # Compute s:
    s1 = z - alpha/z
    # Compute correction on s:
    ds = -0.078 * (s1**5) / (1 + e)
    s = s1 + ds

    # Compute E:
    E0 = M + e * ( 3.*s - 4.*(s**3.) )

    # Compute final correction to E:
    sinE = np.sin(E0)
    cosE = np.cos(E0)

    f = E0 - e*sinE - M
    fp = 1. - e*cosE
    fpp = e*sinE
    fppp = e*cosE
    fpppp = -fpp

    dx1 = -f / fp
    dx2 = -f / (fp + 0.5*fpp*dx1)
    dx3 = -f / ( fp + 0.5*fpp*dx2 + (1./6.)*fppp*(dx2**2) )
    dx4 = -f / ( fp + 0.5*fpp*dx3 + (1./6.)*fppp*(dx3**2) + (1./24.)*(fpppp)*(dx3**3) )

    return E0 + dx4

def GetPhaseAngle(MeanAnom, ecc, inc, argp):
    ''' Function for returning observed phase angle given orbital elements
    Args:
        MeanAnom (flt): Mean anomly in radians, where MeanAnom = orbit fraction*2pi, or M=2pi * time/Period
        ecc (flt): eccentricity, defined on [0,1)
        inc (flt): inclination in degrees, where inc = 90 is edge on, inc = 0 or 180 is face on orbit
        argp (flt): argument of periastron in degrees, defined on [0,360)
        
    Returns:
        flt: phase angle in degrees
    Written by Logan Pearce, 2023
    '''
    import numpy as np
    inc = np.radians(inc)
    argp = np.radians(argp)
    EccAnom = danby_solve(eccentricity_anomaly, MeanAnom, ecc, 0.001, maxnum=50)
    TrueAnom = 2*np.arctan( np.sqrt( (1+ecc)/(1-ecc) ) * np.tan(EccAnom/2) )
    Alpha = np.arccos( np.sin(inc) * np.sin(TrueAnom + argp) )
    
    return np.degrees(Alpha)



def keplerian_to_cartesian(sma,ecc,inc,argp,lon,meananom,kep, solvefunc = danby_solve, return_orbit_plane = False,
                           return_ecc_anom = False):
    """ Given a set of Keplerian orbital elements, returns the observable 3-dimensional position, velocity, 
        and acceleration at the specified time.  Accepts and arbitrary number of input orbits.  Semi-major 
        axis must be an astropy unit object in physical distance (ex: au, but not arcsec).  The observation
        time must be converted into mean anomaly before passing into function.
        Inputs:
            sma (1xN arr flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (1xN arr flt) [unitless]: eccentricity
            inc (1xN arr flt) [deg]: inclination
            argp (1xN arr flt) [deg]: argument of periastron
            lon (1xN arr flt) [deg]: longitude of ascending node
            meananom (1xN arr flt) [radians]: mean anomaly 
            kep (1xN arr flt): kepler constant = mu/m where mu = G*m1*m2 and m = [1/m1 + 1/m2]^-1 . 
                        In the limit of m1>>m2, mu = G*m1 and m = m2
        Returns:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
            vel (3xN arr) [km/s]: velocity in xyz plane.
            acc (3xN arr) [km/s/yr]: acceleration in xyz plane.
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    
    # Compute mean motion and eccentric anomaly:
    meanmotion = np.sqrt(kep / sma**3).to(1/u.s)
    try:
        E = solvefunc(eccentricity_anomaly, meananom, ecc, 0.001)
    except:
        nextE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(meananom, ecc)]
        E = np.array(nextE)

    # Compute position:
    try:
        pos_planeoforbit = np.zeros((3,len(sma)))
    # In the plane of the orbit:
        pos_planeoforbit[0,:], pos_planeoforbit[1,:] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
    except:
        pos_planeoforbit = np.zeros(3)
        pos_planeoforbit[0], pos_planeoforbit[1] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
        
    # Rotate to plane of the sky:
    pos = rotate_z(pos_planeoforbit, np.radians(argp))
    pos = rotate_x(pos, np.radians(inc))
    pos = rotate_z(pos, np.radians(lon))
    
    # compute velocity:
    try:
        vel = np.zeros((3,len(sma)))
        vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1- ecc * np.cos(E) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value
    except:
        vel = np.zeros(3)
        vel[0], vel[1] = (( -meanmotion * sma * np.sin(E) ) / ( 1- ecc * np.cos(E) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(E) ) / ( 1 - ecc * np.cos(E) )).to(u.km/u.s).value
    vel = rotate_z(vel, np.radians(argp))
    vel = rotate_x(vel, np.radians(inc))
    vel = rotate_z(vel, np.radians(lon))
    
    # Compute accelerations numerically
    # Generate a nearby future time point(s) along the orbit:
    deltat = 0.002*u.yr
    try:
        acc = np.zeros((3,len(sma)))
        futurevel = np.zeros((3,len(sma)))
    except:
        acc = np.zeros(3)
        futurevel = np.zeros(3)
    # Compute new mean anomaly at future time:
    futuremeananom = meananom + meanmotion*((deltat).to(u.s))
    # Compute new eccentricity anomaly at future time:
    try:
        futureE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(futuremeananom.value, ecc)]
        futureE = np.array(futureE)
    except:
        futureE = solvefunc(eccentricity_anomaly, futuremeananom.value, ecc, 0.001)
    # Compute new velocity at future time:
    futurevel[0], futurevel[1] = (( -meanmotion * sma * np.sin(futureE) ) / ( 1- ecc * np.cos(futureE) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(futureE) ) / ( 1 - ecc * np.cos(futureE) )).to(u.km/u.s).value
    futurevel = rotate_z(futurevel, np.radians(argp))
    futurevel = rotate_x(futurevel, np.radians(inc))
    futurevel = rotate_z(futurevel, np.radians(lon))
    acc = (futurevel-vel)/deltat.value

    if return_orbit_plane and return_ecc_anom:
        return E, pos_planeoforbit*u.au, np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)
    elif return_ecc_anom:
        return E, pos_planeoforbit*u.au, np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)
    elif return_orbit_plane:
        return pos_planeoforbit*u.au, np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)
    else:
        return np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)

def keplersconstant(m1,m2):
    '''Compute Kepler's constant for two gravitationally bound masses k = G*m1*m2/(m1+m2) = G + (m1+m2)
        Inputs:
            m1,m2 (arr,flt): masses of the two objects in solar masses.  Must be astropy objects
        Returns:
            Kepler's constant in m^3 s^(-2)
    '''
    import astropy.constants as c
    import astropy.units as u
    m1 = m1.to(u.Msun)
    m2 = m2.to(u.Msun)
    mu = c.G*m1*m2
    m = (1/m1 + 1/m2)**(-1)
    kep = mu/m
    return kep.to((u.m**3)/(u.s**2))



def rotate_z(vector,theta):
    """ Rotate a 3D vector about the +z axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    import numpy as np
    R = np.array([[np.cos(theta), -np.sin(theta), 0],
               [np.sin(theta), np.cos(theta), 0],
               [0, 0, 1]
              ], dtype = object)
    if np.ndim(vector) == 1:
        out = np.zeros(3)
        out[0] = R[0,0]*vector[0] + R[0,1]*vector[1] + R[0,2]*vector[2]
        out[1] = R[1,0]*vector[0] + R[1,1]*vector[1] + R[1,2]*vector[2]
        out[2] = R[2,0]*vector[0] + R[2,1]*vector[1] + R[2,2]*vector[2]
        
    else:
        out = np.zeros((3,vector.shape[1]))
        out[0] = R[0,0]*vector[0,:] + R[0,1]*vector[1,:] + R[0,2]*vector[2,:]
        out[1] = R[1,0]*vector[0,:] + R[1,1]*vector[1,:] + R[1,2]*vector[2,:]
        out[2] = R[2,0]*vector[0,:] + R[2,1]*vector[1,:] + R[2,2]*vector[2,:]
    
    return out

def rotate_x(vector,theta):
    """ Rotate a 3D vector about the +x axis
        Inputs:
            vector: 3d vector array
            theta [rad]: angle to rotate the vector about
        Returns: rotated vector
    """
    import numpy as np
    if np.ndim(vector) == 1:
        R = np.array([[1., 0., 0.],
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]  
              ], dtype = object)
        out = np.zeros(3)
        out[0] = R[0,0]*vector[0] + R[0,1]*vector[1] + R[0,2]*vector[2]
        out[1] = R[1,0]*vector[0] + R[1,1]*vector[1] + R[1,2]*vector[2]
        out[2] = R[2,0]*vector[0] + R[2,1]*vector[1] + R[2,2]*vector[2]
        
    else:
        R = np.array([[[1.]*len(theta), 0., 0.],
              [0., np.cos(theta), -np.sin(theta)],
              [0., np.sin(theta), np.cos(theta)]  
              ], dtype = object)
        out = np.zeros((3,vector.shape[1]))
        out[0] = R[0,0]*vector[0,:] + R[0,1]*vector[1,:] + R[0,2]*vector[2,:]
        out[1] = R[1,0]*vector[0,:] + R[1,1]*vector[1,:] + R[1,2]*vector[2,:]
        out[2] = R[2,0]*vector[0,:] + R[2,1]*vector[1,:] + R[2,2]*vector[2,:]
    return out


def ComputeFluxRatio(Rp, sep, alpha, Ag = 0.5):
    ''' For a single planet compute planet/star flux ratio using Cahoy 2010 eqn 1
    and https://ui.adsabs.harvard.edu/abs/2017ApJ...844...89C/abstract
    
    Args:
        Rp (astropy units object): planet radius
        sep (astropy units object): planet-star separation
        alpha (flt): phase angle in degrees
        Ag (flt): geometric albedo

    Returns:
        flt: planet-star contrast
    '''
    alpha = np.radians(alpha)
    angleterm = (np.sin(alpha) + (np.pi - alpha)*np.cos(alpha)) / np.pi
    Rp = Rp.to(u.km)
    sep = sep.to(u.km)
    C = Ag * ((Rp / sep)**2) * angleterm
    return C


def Get_LOD(central_wavelength, D):
    ''' Return lambda/D in mas mas for a filter and primary diameter
    Args:
        central_wavelength (flt, astropy units object): wavelength of filter
        D (flt, astropy units object): primary diameter
    Returns:
        flt: lambda over D in mas
    '''
    central_wavelength = central_wavelength.to(u.um)
    D = D.to(u.m)
    lod = 0.206*central_wavelength/D
    lod = (lod.value)*u.arcsec.to(u.mas)
    return lod


def GetFWHM(wavelength, transmission):
    ''' Given a filter curve, compute the full width half max
    Args:
        wavelength (arr): array of wavelength values
        transmission (arr): array of transmission values
    
    Returns:
        flt: full width half max
        flt: wavelength location of low end half max
        flt: wavelength location of high end half max
        flt: transmission value at half max
    '''
    from scipy.interpolate import interp1d
    # Compute value at half max:
    hm = (0.5 * (np.max(transmission)-np.min(transmission))) + np.min(transmission)
    # Find regions of transmission array above and below half max value:
    signs = np.sign(np.add(transmission, -hm))
    # find where the values change from positive to negative:
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    # for the low end:
    i = 0
    # interpolate between the two values on either side of the half max value:
    g = interp1d([transmission[zero_crossings_i[i]],transmission[zero_crossings_i[i]+1]],
                 [wavelength[zero_crossings_i[i]],wavelength[zero_crossings_i[i]+1]])
    hmlow = g(hm)
    # repeat for high end:
    i = 1
    g = interp1d([transmission[zero_crossings_i[i]],transmission[zero_crossings_i[i]+1]],
                 [wavelength[zero_crossings_i[i]],wavelength[zero_crossings_i[i]+1]])
    hmhigh = g(hm)
    # compute full width at half max:
    fwhm = np.abs(hmhigh - hmlow)
    return fwhm, hmlow, hmhigh, hm


def Get_LOD(central_wavelength, D):
    ''' Return lambda/D in mas mas for a filter and primary diameter
    Args:
        central_wavelength (flt, astropy units object): wavelength of filter
        D (flt, astropy units object): primary diameter
    Returns:
        flt: lambda over D in mas
    '''
    central_wavelength = central_wavelength.to(u.um)
    D = D.to(u.m)
    lod = 0.206*central_wavelength.value/D.value
    lod = lod*u.arcsec.to(u.mas)
    return lod



class NIRC2JFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_j.csv')
        #file = 'filter_curves/nirc2_j.csv'
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = u.um
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        #self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        self.label = 'J'

class NIRC2HFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_h.csv')
        #file = '/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/filter_curves/nirc2_h.csv'
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = u.um
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        #self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        self.label = 'H'

class NIRC2KFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_k.csv')
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = r'$\mu$m'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        #self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class NIRC2KsFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_Ks.csv')
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = r'$\mu$m'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        #self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class NIRC2KpFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_Kp.csv')
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = r'$\mu$m'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        #self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class NIRC2KcontFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_Kcont.csv')
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = r'$\mu$m'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        #self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class NIRC2LpFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_Lp.csv')
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = r'$\mu$m'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        #elf.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        
class NIRC2MsFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_Ms.csv')
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = np.array(f['Transmission']) / 100
        self.wavelength_unit = r'$\mu$m'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class SloangFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/SLOAN_SDSS.gprime_filter.dat')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission'])
        self.wavelength_unit = u.AA
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        self.label = r'g$^{\prime}$'

class SloaniFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/SLOAN_SDSS.iprime_filter.dat')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission'])
        self.wavelength_unit = u.AA
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        self.label = r'i$^{\prime}$'

class SloanrFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/SLOAN_SDSS.rprime_filter.dat')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission'])
        self.wavelength_unit = u.AA
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        self.label = r'r$^{\prime}$'

class SloanuFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/SLOAN_SDSS.uprime_filter.dat')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission'])
        self.wavelength_unit = u.AA
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        self.label = r'u$^{\prime}$'

class SloanzFilter(object):
    def __init__(self):
        import os
        #file = os.path.join(os.path.dirname(__file__), 'filter_curves/SLOAN_SDSS.zprime_filter.dat')
        file = "/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/filter_curves/Sloan_z.txt"
        f = pd.read_csv(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission']) / 100
        self.wavelength_unit = u.nm
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        eff_wavelength = np.round(np.sum(self.transmission) / np.sum(self.transmission * (1/self.wavelength**2)), 
                          decimals = 3)
        self.effective_wavelength = np.sqrt(eff_wavelength)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        self.label = r'z$^{\prime}$'


def GetPhotonsPerSec(wavelength, flux, filt, distance, radius, primary_mirror_diameter,
                    return_ergs_flux_times_filter = False, Omega = None):
    ''' Given a spectrum with wavelengths in um and flux in ergs cm^-1 s^-1 cm^-2, convolve 
    with a filter transmission curve and return photon flux in photons/sec
    
    Args:
        wavelength [arr]: wavelength array in um
        flux [arr]: flux array in ergs cm^-1 s^-1 cm^-2 from the surface of the object (NOT corrected for distance)
        filt [myastrotools filter object]: filter
        distance [astropy unit object]: distance to star with astropy unit
        radius [astropy unit object]: radius of star or planet with astropy unit
        primary_mirror_diameter [astropy unit object]: primary mirror diameter with astropy unit
        return_ergs_flux [bool]: if True, return photons/sec and the flux in ergs cm^-1 s^-1 cm^-2
                                convolved with the filter
    Returns
        astropy units object: flux in photons/sec
        
    '''
    # correct for distance:
    D = distance
    Rp = radius
    if not Omega:
        Omega = ((Rp/D).decompose())**2
    flux = flux * Omega

    # energy in ergs:
    energy_per_photon_per_wavelength = c.h.cgs * c.c.cgs / wavelength
    # Flux in photons/cm s cm^2: number of photons per area per sec per lambda:
    nphotons_per_wavelength = flux / energy_per_photon_per_wavelength
    
    # Combine flux with filter curve:
    w = filt.wavelength*filt.wavelength_unit.to(u.um)
    t = filt.transmission
    # make interpolation function:
    ind = np.where((wavelength > np.min(w)) & (wavelength < np.max(w)))[0]
    # of spectrum wavelength and Flux in photons/cm s cm^2:
    from scipy.interpolate import interp1d
    f = interp1d(wavelength[ind], nphotons_per_wavelength[ind], fill_value="extrapolate")
    # interpolate the filter flux onto the spectrum wavelength grid:
    flux_on_filter_wavelength_grid = f(w)

    # multiply flux time filter transmission
    filter_times_flux = flux_on_filter_wavelength_grid * t
    
    # Now sum:
    dl = (np.mean([w[j+1] - w[j] for j in range(len(w)-1)]) * u.um).to(u.cm)

    total_flux_in_filter = np.sum(filter_times_flux * dl.value)
    
    
    area_of_primary = np.pi * ((0.5*primary_mirror_diameter).to(u.cm))**2

    #Total flux in photons/sec:
    total_flux_in_photons_sec = total_flux_in_filter * area_of_primary.value
    
    if return_ergs_flux_times_filter:
        f = interp1d(wavelength[ind], flux[ind], fill_value="extrapolate")
        # interpolate the filter flux onto the spectrum wavelength grid:
        flux_ergs_on_filter_wavelength_grid = f(w)
        filter_times_flux_ergs = flux_ergs_on_filter_wavelength_grid * t
        
        return total_flux_in_photons_sec * (1/u.s), filter_times_flux_ergs, w
    
    return total_flux_in_photons_sec * (1/u.s)

def GetFofLambdaNaught(wavelength,flux,filt):
    ''' For a specific filter object and spectrum, get the 
    flux at the central wavelength
    
    Args:
        wavelength (arr): wavelength array in microns
        flux (arr): flux array
        filt (filter object): filter
    
    Returns:
        flt: flux at the central wavelength
    '''
    from scipy.interpolate import interp1d
    f = interp1d(wavelength,flux)
    F0 = f(filt.central_wavelength*filt.wavelength_unit.to(u.um))
    return F0

def GetGuidestarMagForIasTable(wavelength, flux, filt, distance, star_radius, primary_mirror_diameter):
    ''' Get apparent magnitude in vega mags for a specific filter

    Args:
        wavelength [arr]: wavelength array in um
        flux [arr]: flux array in ergs cm^-1 s^-1 cm^-2 from the surface of the object (NOT corrected for distance)
        filt [myastrotools filter object]: filter
        distance [astropy unit object]: distance to star with astropy unit
        radius [astropy unit object]: radius of star or planet with astropy unit
        primary_mirror_diameter [astropy unit object]: primary mirror diameter with astropy unit

    Returns:
        str: nearest guidestar magnitude for Ias table lookup
        flt: star magnitude in Vega mags
    '''
    import pysynphot as S
    star_flux_in_phot,star_flux_in_ergs, filt_w = GetPhotonsPerSec(wavelength, flux, filt, distance, star_radius, 
                                                       primary_mirror_diameter,
                                                       return_ergs_flux_times_filter=True)
    star_Flambda0 = GetFofLambdaNaught(filt_w, star_flux_in_ergs, filt)
    
    vega = S.Vega
    vega_Flam0 = vega.sample(filt.central_wavelength*filt.wavelength_unit.to(u.AA))
    vega_Flambda0 = vega_Flam0*(1/u.AA).to(1/u.cm)
    
    star_magnitude = -2.5 * np.log10(star_Flambda0/vega_Flambda0.value)
    available_mags = np.array(['0', '2.5', '5', '7', '9', '10', '11',
                        '11.5', '12', '12.5', '13','13.5', '14', '14.5', '15'])
    available_mags = np.array([float(m) for m in available_mags])
    idx = (np.abs(available_mags - star_magnitude)).argmin()
    guidestarmag = str(available_mags[idx]).replace('.0','')
    
    return guidestarmag, star_magnitude


def GetNoiseModelMap(guidestarmag, wfc):
    from astropy.io import fits
    IasMap = fits.getdata(f'/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/GMagAO-X-noise/contrast_{guidestarmag}_{wfc}.fits')
    return IasMap

def GetIasFromTable(guidestarmag, wfc, sep, pa):
    ''' For a given guide star magnitude and wfc, look up the value of the atmospheric speckle
        contribution I_as (Males et al. 2021 eqn 6) at a given separation and position angle
        
    Args:
        guidestarmag (flt or str): Guide star magnitude. Must be: ['0', '2.5', '5', '7', '9', '10', '11',
                        '11.5', '12', '12.5', '13','13.5', '14', '14.5', '15']
        wfc (str): wavefront control set up.  Either linear predictive control "lp" or simple integrator "si"
        sep (flt): separation in lambda/D
        pa (flt): position angle in degrees
    
    Returns:
        flt: value of I_as at that location
    '''
    IasMap = GetNoiseModelMap(guidestarmag, wfc)
    center = [0.5*(IasMap.shape[0]-1),0.5*(IasMap.shape[1]-1)]
    dx = sep * np.cos(np.radians(pa + 90))
    dy = sep * np.sin(np.radians(pa + 90))
    if int(np.round(center[0]+dx, decimals=0)) < 0:
        return np.nan
    try:
        return IasMap[int(np.round(center[0]+dx, decimals=0)),int(np.round(center[1]+dy,decimals=0))]
    except IndexError:
        return np.nan
    
def GetIas(guidestarmag, wfc, sep, pa, wavelength):
    '''For a given guide star magnitude, wfc, and planet-star contrast, get the SNR
        in the speckle-limited regime (Eqn 10 of Males et al. 2021)
        at a given separation and position angle.
        
    Args:
        guidestarmag (flt or str): Guide star magnitude. Must be: ['0', '2.5', '5', '7', '9', '10', '11',
                        '11.5', '12', '12.5', '13','13.5', '14', '14.5', '15']
        wfc (str): wavefront control set up.  Either linear predictive control "lp" or simple integrator "si"
        sep (flt): separation in lambda/D
        pa (flt): position angle in degrees
        Cp (flt): planet-star contrast
        deltat (flt): observation time in sec
        wavelength (astropy units object):  central wavelength of filter band
        tau_as (flt): lifetime of atmospheric speckles in sec. Default = 0.02, ave tau_as for 24.5 m telescope
                from Males et al. 2021 Fig 10
    
    Returns:
        flt: value of I_as at that location
    '''
    wavelength = wavelength.to(u.um)
    # Look up Ias from table
    Ias = GetIasFromTable(guidestarmag, wfc, sep, pa)
    # Correct for differnce in wavelength between lookup table and filter wavelength:
    Ias = Ias * (((0.8*u.um/wavelength))**2).value
    if np.isnan(Ias):
        raise Exception('Sep/PA is outside noise map boundaries')
    else:
        return Ias
    
def Get_LOD(central_wavelength, D):
    ''' Return lambda/D in mas mas for a filter and primary diameter
    Args:
        central_wavelength (flt, astropy units object): wavelength of filter
        D (flt, astropy units object): primary diameter
    Returns:
        flt: lambda over D in mas
    '''
    central_wavelength = central_wavelength.to(u.um)
    D = D.to(u.m)
    lod = 0.206*central_wavelength.value/D.value
    lod = lod*u.arcsec.to(u.mas)
    return lod
    

def ComputePlanetSNR(Ip, Istar, Ic, Ias, Iqs, tau_as, tau_qs, deltat, 
                           RN = None, Isky = None, Idc = None, texp = None):
    ''' Get S/N for a planet signal when speckle noise dominated.

    Args:
        Ip [flt]: planet signal in photons/sec
        Istar [flt]: star signal in photons/sec
        Ic [flt]: fractional contribution of intensity from residual dirraction from coronagraph
        Ias [flt]: contribution from atm speckles
        Iqs [flt]: fraction from quasistatic speckles
        tau_as [flt]: average lifetime of atm speckles
        tau_qs [flt]: average liftetim of qs speckles
        deltat [flt]: observation time in seconds
        RN [flt]: read noise
        Isky [flt]: sky intensity in photons/sec
        Idc [flt]: dark current in photons/sec
        texp [flt]: time for single exposure in sec (required only for RN term)

    Returns:
        flt: signal to noise ratio
    '''
    signal = Ip * deltat
    photon_noise = Ic + Ias + Iqs
    atm_speckles = Istar * ( tau_as * (Ias**2 + 2*(Ic*Ias + Ias*Iqs)) )
    qs_speckles = Istar * ( tau_qs * (Iqs**2 + 2*Ic*Iqs) )
    sigma_sq_h = Istar * deltat * (photon_noise + atm_speckles + qs_speckles) + signal
    if RN is not None:
        skyanddetector = Isky*deltat + Idc*deltat + (RN * deltat/texp)**2
        noise = np.sqrt(sigma_sq_h + skyanddetector)
    else:
        noise = np.sqrt(sigma_sq_h)
        
    return signal / noise


def GetSNR(planet_wavelength, planet_flux, 
           star_wavelength, star_flux,
           primary_mirror_diameter, 
           planet_radius, star_radius, 
           distance, sep_au, wfc,
           filters, observationtime,
           Ic = 1e-20,
           Iqs = 1e-20,
           tau_as = 0.02, # sec, from Fig 10 in Males+ 2021
           tau_qs = 0.05,
           RN = None, Isky = None, Idc = None, texp = None
          ):
    
    from myastrotools.reflectx_supplemental import GetPhotonsPerSec
    
    ####### Signal:
    # Planet signal:
    planet_signal = []
    for filt in filters:
        planet_signal.append(GetPhotonsPerSec(planet_wavelength, planet_flux, filt, distance, 
                                              planet_radius, primary_mirror_diameter).value)
    planet_signal = np.array(planet_signal)
    
    # Star signal:
    star_signal = []
    for filt in filters:
        star_signal.append(GetPhotonsPerSec(star_wavelength, star_flux, filt, distance, 
                                            star_radius, primary_mirror_diameter).value)
    star_signal = np.array(star_signal)
    
    ####### Noise:
    ### Get Ias:
    # get guidestar magnitude:
    from myastrotools.reflectx_supplemental import GetGuidestarMagForIasTable, GetIas, Get_LOD
    star_gsm = []
    star_mag = []
    for filt in filters:
        gsm, mag = GetGuidestarMagForIasTable(star_wavelength, star_flux, filt, distance, star_radius, 
                                                   primary_mirror_diameter)
        star_gsm.append(gsm)
        star_mag.append(mag)
    star_gsm = np.array(star_gsm)
    star_mag = np.array(star_mag)
    # Get Ias from lookup table:
    sep_mas = (sep_au/distance.value)*u.arcsec.to(u.mas)
    # LOD:
    lods = [Get_LOD(f.central_wavelength*f.wavelength_unit, primary_mirror_diameter) for f in filters]
    sep_lods = [sep_mas/lod for lod in lods]

    pa = 90 # deg
    # Get Ias:
    Ias = []
    for i in range(len(filters)):
        wavelength = filters[i].central_wavelength*filters[i].wavelength_unit
        Ias.append(GetIas(star_gsm[i], wfc, sep_lods[i], pa, wavelength))
    Ias = np.array(Ias)
    
    ###### SNR:
    from myastrotools.reflectx_supplemental import ComputePlanetSNR

    if type(observationtime) == np.ndarray:
        # For an array of times:
        all_snrs = []
        for i in range(len(filters)):
            snrs = []
            for t in observationtime:
                snrs.append(ComputePlanetSNR(planet_signal[i], star_signal[i], 
                                             Ic, Ias[i], Iqs, tau_as, tau_qs, t, 
                                   RN = None, Isky = None, Idc = None, texp = None))
            all_snrs.append(snrs)
        return all_snrs
    else:
        # for a single time:
        snrs = []
        for i in range(len(filters)):
            snrs.append(ComputePlanetSNR(planet_signal[i], star_signal[i], Ic, Ias[i], Iqs, tau_as, tau_qs, 
                                observationtime, 
                                RN = RN, Isky = Isky, Idc = Idc, texp = texp))
        return snrs


def MakeInteractiveSeparationContrastPlotOfNearbyRVPlanets(orbits, plotx, ploty, phases, 
                                                           saveplot = True, 
                                                           sepau = None,
                                                           sepmas = None,
                                                           filt = 'None', xaxis_label = '',
                                                           annotation_text = '', IWA = 2,
                                                           ytop = 2e-6, ybottom = 2e-10,
                                                           xright = 20, xleft = 0,
                                                           ncolors = 10, ticklocs = 'None', ticklabels = 'None',
                                                          output_file_name = 'RVPlanetContrastPlot'):


    rad = orbits['Re'].copy()
    spt = orbits['SpT Number'].copy()
    
    plotx, ploty = np.array(plotx),np.array(ploty)
    multiplier = 2
    datadf = pd.DataFrame(data={'plotx':plotx, 'ploty':ploty, 'color':spt, 'markersize':rad*multiplier,
                               'name':orbits['pl_name'], 'rad':rad, 'spt':spt, 'dist':orbits['sy_dist'],
                                'phases':phases, 'plotx_og':plotx, 'ploty_og':ploty, 'iwa': 2, 
                                'sepau':sepau, 'sepmas':sepmas
                               })
    datadict = datadf.to_dict(orient = 'list')

    from bokeh.plotting import figure, show, output_file, save
    from bokeh.io import output_notebook
    from bokeh.models import LinearColorMapper, ColumnDataSource, LinearInterpolator
    from bokeh.models import  Range1d, LabelSet, Label, ColorBar, FixedTicker, Span
    from bokeh.models import CustomJS, Slider
    from bokeh.layouts import column, row
    from bokeh.palettes import Magma256, Turbo256, brewer
    from bokeh.transform import linear_cmap
    #output_notebook()


    data=ColumnDataSource(data=datadict)


    tools = "hover, zoom_in, zoom_out, save, undo, redo, reset, pan"
    tooltips = [
        ('Planet', '@name'),
        #("(x,y)", "($x, $y)"),
        ('Cont', '@ploty'),
        ('Phase [deg]', '@phases{0}'),
        ('Sep [au]', '@sepau{0.00}'),
        ('Sep [mas]', '@sepmas{0.00}'),
        ('Rad [Rearth]','@rad{0.00}'),
        ('SpT','@spt{0.0}'),
        ('Dist [pc]','@dist{0.0}')
    ]

    p = figure(width=900, height=750, y_axis_type="log", tools=tools, 
               tooltips=tooltips, toolbar_location="above")

#     mapper = linear_cmap(field_name='color', 
#                          #palette=Magma256,
#                          palette=Turbo256[::-1],
#                          low=min(spt), high=max(spt),
#                         low_color=Turbo256[::-1][150], high_color=Turbo256[::-1][200])
    
    mapper = linear_cmap(field_name='phases', 
                         palette=brewer['PuOr'][ncolors],
                         #palette=Magma256,
                         #palette=Turbo256[::-1],
                         #low=min(phases), high=max(phases))
                         low=20, high=150)
    
    p.circle('plotx','ploty', source=data, fill_alpha=0.8, size='markersize', 
             line_color=mapper, color=mapper)

    
    color_bar = ColorBar(color_mapper=mapper['transform'], width=15, 
                         location=(0,0), title="Phase",
                        title_text_font_size = '20pt',
                         major_label_text_font_size = '15pt')
    
    if ticklocs == 'None':
        ticklocs = np.arange(0,180,20)
        ticklabels = {}
        [ticklabels.update({ticklocs[i]:str(np.round(ticklocs[i],decimals=-1)).replace('.0','')}) for i in range(len(ticklocs))]
    color_bar.ticker=FixedTicker(ticks=ticklocs)
    color_bar.major_label_overrides = ticklabels

    p.add_layout(color_bar, 'right')

    label = Label(
        text= annotation_text,
        x=50, y=20,
        x_units="screen", y_units="screen",text_font_size = '20pt'
    )
    p.add_layout(label)
    
    delt = np.log10(ytop) - np.log10(ybottom)
    
    x,y = 16, 10**(np.log10(ybottom) + (0.9*delt))
    p.circle(x,y, fill_alpha=0.6, size=11*multiplier,
             color='black')
    label1 = Label(x=x, y=y, text=r'\[ 11 R_\oplus\]',
                       x_offset=20, y_offset=-20,
                       text_font_size = '20pt')
    p.add_layout(label1)
    
    x,y = 16, 10**(np.log10(ybottom) + (0.85*delt))
    p.circle(x,y, fill_alpha=0.6, size=5*multiplier, 
             color='black')
    label2 = Label(x=x, y=y, text=r'$$5 R_\oplus$$',
                      x_offset=20, y_offset=-20,text_font_size = '20pt')
    p.add_layout(label2)
    x,y = 16, 10**(np.log10(ybottom) + (0.8*delt))
    p.circle(x,y, fill_alpha=0.6, size=1*multiplier, 
             color='black')
    label3 = Label(x=x, y=y, text=r'$$1 R_\oplus$$',
                      x_offset=20, y_offset=-20,text_font_size = '20pt')
    p.add_layout(label3)

    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = r'\[ \mathrm{Planet/Star\; Reflected\; Light\; Flux\; Ratio} \]'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.major_label_text_font_size = "15pt"
    p.xaxis.major_label_text_font_size = "15pt"
    
    iwa = Span(location=IWA,
                              dimension='height', line_color='grey',
                              line_dash='dashed', line_width=3)
#     iwa = Span(location=data['iwa'],
#                                dimension='height', line_color='grey',
#                                line_dash='dashed', line_width=3)
    p.add_layout(iwa)
    
    p.x_range=Range1d(xleft,xright)
    p.y_range=Range1d(ybottom,ytop)
    

    AgSlider = Slider(start=0.05, end=1.0, value=0.3, step=.01, title="Geometric Albedo")
    IWASlider = Slider(start=1, end=10, value=2, step=.5, title="IWA")
    LambdaSlider = Slider(start=400, end=2000, value=800, step=50, title="Wavelength [nm]")
    DSlider = Slider(start=2, end=34, value=25.4, step=0.5, title="Primary Mirror Diameter [m]")
#         
#        var newx = x.map(d => d * 6.3/newlod );
    sliders_callback_code = """
        var Ag = Ag.value;
        var Lambda = Lambda.value;
        var D = D.value;
        
        var lod = 6.3;
        var newlod = ((Lambda/1000) / D) * 1000
        
        var y = source.data['ploty_og'];
        var x = source.data['plotx_og'];
        var newy = y.map(m => m * Ag/0.3 );
        var newx = x.map(b => b * 800/Lambda );
        var newx = newx.map(d => d * D/25.4 );


        console.log(newy)
        console.log(newx)
        source.data['ploty'] = newy;
        source.data['plotx'] = newx;
        source.change.emit();
    """

    slider_args = dict(source=data, Ag=AgSlider, Lambda=LambdaSlider, D=DSlider)
    
    AgSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))
    LambdaSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))
    DSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))


    #show(row(p, column(AgSlider)))
    #show(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
    

    
    if saveplot:
        output_file(output_file_name+".html")
        save(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
    else:
        show(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
        #pass

    #return p

def MakeInteractiveSeparationContrastPlotOfNearbyRVPlanetsMagAOX(orbits, plotx, ploty, phases, 
                                                           saveplot = True, 
                                                           sepau = None,
                                                           sepmas = None,
                                                           filt = 'None', xaxis_label = '',
                                                           annotation_text = '', IWA = 2,
                                                           ytop = 2e-6, ybottom = 2e-10,
                                                           xright = 20, xleft = 0,
                                                           ncolors = 10, ticklocs = 'None', ticklabels = 'None',
                                                          output_file_name = 'RVPlanetContrastPlot'):


    rad = orbits['Re'].copy()
    spt = orbits['SpT Number'].copy()
    
    plotx, ploty = np.array(plotx),np.array(ploty)
    multiplier = 2
    datadf = pd.DataFrame(data={'plotx':plotx, 'ploty':ploty, 'color':spt, 'markersize':rad*multiplier,
                               'name':orbits['pl_name'], 'rad':rad, 'spt':spt, 'dist':orbits['sy_dist'],
                                'phases':phases, 'plotx_og':plotx, 'ploty_og':ploty, 'iwa': 2, 
                                'sepau':sepau, 'sepmas':sepmas
                               })
    datadict = datadf.to_dict(orient = 'list')

    from bokeh.plotting import figure, show, output_file, save
    from bokeh.io import output_notebook
    from bokeh.models import LinearColorMapper, ColumnDataSource, LinearInterpolator
    from bokeh.models import  Range1d, LabelSet, Label, ColorBar, FixedTicker, Span
    from bokeh.models import CustomJS, Slider
    from bokeh.layouts import column, row
    from bokeh.palettes import Magma256, Turbo256, brewer
    from bokeh.transform import linear_cmap
    #output_notebook()


    data=ColumnDataSource(data=datadict)


    tools = "hover, zoom_in, zoom_out, save, undo, redo, reset, pan"
    tooltips = [
        ('Planet', '@name'),
        #("(x,y)", "($x, $y)"),
        ('Cont', '@ploty'),
        ('Phase [deg]', '@phases{0}'),
        ('Sep [au]', '@sepau{0.00}'),
        ('Sep [mas]', '@sepmas{0.00}'),
        ('Rad [Rearth]','@rad{0.00}'),
        ('SpT','@spt{0.0}'),
        ('Dist [pc]','@dist{0.0}')
    ]

    p = figure(width=900, height=750, y_axis_type="log", tools=tools, 
               tooltips=tooltips, toolbar_location="above")

#     mapper = linear_cmap(field_name='color', 
#                          #palette=Magma256,
#                          palette=Turbo256[::-1],
#                          low=min(spt), high=max(spt),
#                         low_color=Turbo256[::-1][150], high_color=Turbo256[::-1][200])
    
    mapper = linear_cmap(field_name='phases', 
                         palette=brewer['PuOr'][ncolors],
                         #palette=Magma256,
                         #palette=Turbo256[::-1],
                         #low=min(phases), high=max(phases))
                         low=20, high=150)
    
    p.circle('plotx','ploty', source=data, fill_alpha=0.8, size='markersize', 
             line_color=mapper, color=mapper)

    
    color_bar = ColorBar(color_mapper=mapper['transform'], width=15, 
                         location=(0,0), title="Phase",
                        title_text_font_size = '20pt',
                         major_label_text_font_size = '15pt')
    
    if ticklocs == 'None':
        ticklocs = np.arange(0,180,20)
        ticklabels = {}
        [ticklabels.update({ticklocs[i]:str(np.round(ticklocs[i],decimals=-1)).replace('.0','')}) for i in range(len(ticklocs))]
    color_bar.ticker=FixedTicker(ticks=ticklocs)
    color_bar.major_label_overrides = ticklabels

    p.add_layout(color_bar, 'right')

    label = Label(
        text= annotation_text,
        x=50, y=20,
        x_units="screen", y_units="screen",text_font_size = '20pt'
    )
    p.add_layout(label)
    
    delt = np.log10(ytop) - np.log10(ybottom)
    
    x,y = 16, 10**(np.log10(ybottom) + (0.9*delt))
    p.circle(x,y, fill_alpha=0.6, size=11*multiplier,
             color='black')
    label1 = Label(x=x, y=y, text=r'\[ 11 R_\oplus\]',
                       x_offset=20, y_offset=-20,
                       text_font_size = '20pt')
    p.add_layout(label1)
    
    x,y = 16, 10**(np.log10(ybottom) + (0.85*delt))
    p.circle(x,y, fill_alpha=0.6, size=5*multiplier, 
             color='black')
    label2 = Label(x=x, y=y, text=r'$$5 R_\oplus$$',
                      x_offset=20, y_offset=-20,text_font_size = '20pt')
    p.add_layout(label2)
    x,y = 16, 10**(np.log10(ybottom) + (0.8*delt))
    p.circle(x,y, fill_alpha=0.6, size=1*multiplier, 
             color='black')
    label3 = Label(x=x, y=y, text=r'$$1 R_\oplus$$',
                      x_offset=20, y_offset=-20,text_font_size = '20pt')
    p.add_layout(label3)

    p.xaxis.axis_label = xaxis_label
    p.yaxis.axis_label = r'\[ \mathrm{Planet/Star\; Reflected\; Light\; Flux\; Ratio} \]'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.major_label_text_font_size = "15pt"
    p.xaxis.major_label_text_font_size = "15pt"
    
    iwa = Span(location=IWA,
                              dimension='height', line_color='grey',
                              line_dash='dashed', line_width=3)
#     iwa = Span(location=data['iwa'],
#                                dimension='height', line_color='grey',
#                                line_dash='dashed', line_width=3)
    p.add_layout(iwa)
    
    p.x_range=Range1d(xleft,xright)
    p.y_range=Range1d(ybottom,ytop)
    

    AgSlider = Slider(start=0.05, end=1.0, value=0.3, step=.01, title="Geometric Albedo")
    IWASlider = Slider(start=1, end=10, value=2, step=.5, title="IWA")
    LambdaSlider = Slider(start=400, end=2000, value=800, step=50, title="Wavelength [nm]")
    DSlider = Slider(start=2, end=34, value=6.5, step=0.5, title="Primary Mirror Diameter [m]")
#         
#        var newx = x.map(d => d * 6.3/newlod );
    sliders_callback_code = """
        var Ag = Ag.value;
        var Lambda = Lambda.value;
        var D = D.value;
        
        var lod = 6.3;
        var newlod = ((Lambda/1000) / D) * 1000
        
        var y = source.data['ploty_og'];
        var x = source.data['plotx_og'];
        var newy = y.map(m => m * Ag/0.3 );
        var newx = x.map(b => b * 800/Lambda );
        var newx = newx.map(d => d * D/6.5 );


        console.log(newy)
        console.log(newx)
        source.data['ploty'] = newy;
        source.data['plotx'] = newx;
        source.change.emit();
    """

    slider_args = dict(source=data, Ag=AgSlider, Lambda=LambdaSlider, D=DSlider)
    
    AgSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))
    LambdaSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))
    DSlider.js_on_change('value', CustomJS(args=slider_args,code=sliders_callback_code))


    #show(row(p, column(AgSlider)))
    #show(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
    

    
    if saveplot:
        output_file(output_file_name+".html")
        save(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
    else:
        show(column(p, row(AgSlider),row(LambdaSlider),row(DSlider)))
        #pass

    #return p

