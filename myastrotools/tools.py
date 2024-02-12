import numpy as np
import astropy.units as u
import astropy.constants as c
import os
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

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

def MonteCarloIt(thing, N = 10000):
    ''' 
    Generate a random sample of size = N from a 
    Gaussian centered at thing[0] with std thing[1]
    
    Args:
        thing (tuple, flt): tuple of (value,uncertainty).  Can be either astropy units object \
            or float
        N (int): number of samples
    Returns:
        array: N random samples from a Gaussian.

    Written by Logan Pearce, 2020
    '''
    try:
        out = np.random.normal(thing[0].value,thing[1].value,N)
    except:
        out = np.random.normal(thing[0],thing[1],N)

    return out

def CenteredDistanceMatrix(nx, ny = None):
    ''' Creates 2d array of the distance of each element from the center

    Parameters
    ----------
        n : flt
            x-dimension of 2d array
        ny : flt (optional)
            optional y-dimension of 2d array.  If not provided, array is square of dimension nxn
    
    Returns
    -------
        2d matrix of distance from center
    '''
    if ny:
        pass
    else:
        ny = nx
    center = ((nx-1)*0.5,(ny-1)*0.5)
    xx,yy = np.meshgrid(np.arange(nx)-center[0],np.arange(ny)-center[1])
    r=np.hypot(xx,yy)
    return r

def circle_mask(radius, xsize, ysize, xc, yc, radius_format = 'pixels', cval = 0):
    xx,yy = np.meshgrid(np.arange(xsize)-xc,np.arange(ysize)-yc)
    r=np.hypot(xx,yy)
    return np.where(r<radius)

############################################################################################################
############################################# Astrometry ###################################################
############################################################################################################

def parallax(d):
    """
    Returns parallax in arcsec given distances.
    Args:
        d (float): distance
    Return:
        parallax in arcsecs
    Written by: Logan Pearce, 2017
    """
    from astropy import units as u
    d = d.to(u.pc)
    x = (1*u.au)/(d)
    return x.to(u.arcsec, equivalencies=u.dimensionless_angles())

def physical_separation(d,theta):
    """
    Returns separation between two objects in the plane of the sky in AU given distance and parallax
    Distance and parallax must be astropy unit objects.
    Args:
        d (float): distance
        theta (float): parallax
    Return:
        separation in AU
    Written by: Logan Pearce, 2017
    """
    from astropy import units as u
    d = d.to(u.pc)
    theta = theta.to(u.arcsec)
    a = (d)*(theta)
    return a.to(u.au, equivalencies=u.dimensionless_angles())

def angular_separation(d,a):
    """
    Returns angular separation between two objects in the plane of the sky in arcsec given distance and 
    physical separation
    Distance and physcial separation must be astropy unit objects.
    Args:
        d (float): distance
        a (float): physical separation
    Return:
        separation in AU
    Written by: Logan Pearce, 2017
    """
    from astropy import units as u
    d = d.to(u.pc)
    a = a.to(u.au)
    theta = a / d
    return theta.to(u.arcsec, equivalencies=u.dimensionless_angles())

def to_si(mas,mas_yr,d):
    '''Convert from mas -> km and mas/yr -> km/s
        Input: 
         mas (array) [mas]: separation in mas
         mas_yr (array) [mas/yr]: velocity in mas/yr
         d (float) [pc]: distance to system in parsecs
        Returns:
         km (array) [km]: separation in km
         km_s (array) [km/s]: velocity in km/s
    '''
    import astropy.units as u
    km = ((mas*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d)*u.AU).to(u.km)
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    return km.value,km_s

def period(a_au,m):
    """ Given semi-major axis in AU and mass in solar masses, return period in years using Kepler's 3rd law"""
    import numpy as np
    return np.sqrt((np.absolute(a_au)**3)/np.absolute(m))

def circular_velocity(au,m):
    """ Given separation in AU and total system mass, return the velocity of a test particle on a circular orbit
        around a central body at that mass """
    import astropy.constants as c
    import astropy.units as u
    import numpy as np
    try:
        m = m.to(u.Msun)
        au = au.to(u.AU)
    except:
        m = m*u.Msun
        au = au*u.AU
    v = np.sqrt( c.G * m.to(u.kg) / (au.to(u.m)) )
    return v.to(u.km/u.s)

def v_escape(r,M):
    ''' Compute the escape velocity of an object of mass M at distance r.  M and r should be
        astropy unit objects
    '''
    try:
        r = r.to(u.au)
        M = M.to(u.Msun)
    except:
        r = r*u.au
        M = M*u.Msun
    return (np.sqrt(2*c.G*(M) / (r))).decompose()

def parallax_to_circularvel(plx,mass,theta):
    """ Given a parallax value+error, total system mass, and separation, compute the circular velocity
        for a test particle on a circular orbit at that separation.  Plx should be a tuple of
        (plx value, error) in mas.  Theta should be an astropy units object, either arcsec or mas, mass
        in solar masses.  Returns circular vel in km/s
    """
    from myastrotools.astrometry import circular_velocity, physical_separation
    from myastrotools.gaia_tools import distance
    import astropy.units as u
    dist = distance(*plx)[0]*u.pc
    sep = physical_separation(dist,theta)
    cv = circular_velocity(sep.value,mass)
    return cv

def seppa_to_radec(sep,pa):
    """Convert separation and position angle into delta(RA) and delta(Dec) in arcsec.  Sep (in angular units)
        and PA (degrees or radians) must be astropy unit objects.
    """
    import numpy as np
    import astropy.units as u
    try:
        pa = pa.to(u.deg)
        sep = sep.to(u.arcsec)
    except:
        print('Error: Sep/PA must be astropy unit objects')
        return ''
    RA = sep * np.sin(np.radians(pa))
    Dec = sep * np.cos(np.radians(pa))
    return RA.value, Dec.value

def radec_to_seppa(ra, dec):
    """Convert delta(RA) and delta(Dec) to separation in mas and position angle in deg.  RA and Dec
        must be astropy unit objects, such as u.arcsec.
    """
    import numpy as np
    import astropy.units as u
    try:
        ra, dec = ra.to(u.mas), dec.to(u.mas)
    except:
        print('Error: RA/Dec must be astropy unit objects')
        return ''
    sep = np.sqrt(ra**2+dec**2)
    pa = (np.arctan2(ra,dec).to(u.deg).value)%360.
    return sep.value, pa

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

def masyr_to_kms(mas_yr,plx):
    '''
    Convert from mas/yr -> km/s
     
    Args:
        mas_yr (array): velocity in mas/yr
        plx (tuple,float): parallax, tuple of (plx,plx error)
    Returns:
        array : velocity in km/s
    
    Written by Logan Pearce, 2019
    '''
    from orbittools.orbittools import distance
    d = distance(*plx)
    # convert mas to km:
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km)
    # convert yr to s:
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    
    return km_s

def turn_gaia_into_physical(ra, dec, plx, plx_error, pmra, pmdec):
    ''' Take Gaia archive units (deg, mas, mas/yr) and turn them into physical units (pc, AU, km/s)
    Args:
        ra (flt): RA in deg
        dec (flt): DEC in deg
        plx (tuple) [parallax, parallax_error] in mas
        pmra (flt): proper motion in RA in mas/yr
        pmdec (flt): proper motion in DEC in mas/yr
        
    Returns:
        flt: RA in AU
        flt: DEC in AU
        flt: distance in pc
        flt: pmra in km/s
        flt: pmdec in km/s
    '''
    import astropy.units as u
    from orbittools.orbittools import distance, masyr_to_kms
    # Get distance in pc:
    dist = distance(np.array([plx,plx_error]))[0]
    # Convert ra/dec from degrees to arcsec
    ra = (ra*u.deg).to(u.arcsec)
    dec = (dec*u.deg).to(u.arcsec)
    # theta * d = a:
    ra_au = ((ra * dist).value)*u.au
    dec_au = ((dec * dist).value)*u.au
    # proper motions:
    pmra_kms = masyr_to_kms(pmra,plx_tuple)    # km/s
    pmdec_kms = masyr_to_kms(pmdec,plx_tuple)
    
    return ra_au, dec_au, dist, pmra_kms, pmdec_kms

############################################################################################################
###################################### Coordinate Conversions ##############################################
############################################################################################################

def space_velocity(R,D,PR,sig_PR,PD,sig_PD,RV,sig_RV,plx,sig_plx):
    """
    Returns UVW space velocities given RA, Dec, PM, RV, and parallax
    Adapted from the matricies determined in Johnson & Soderblom, 1987.
    Args:
        R (float): right ascension in decimal degrees
        D (float): declination in decimal degrees
        plx, sig_plx (float): parallax in mas
        RV, sig_RV (float): radial velocity in km/s
        PR, sig_PR (float): RA proper motion and error in mas/yr (uncorrected for declination)
        PD, sig_PD (float): Dec proper motion and error in mas/yr
    Return:
        2D array of U, V, W space velocities (U defined as positive towards the galactic center, right handed system),
            and SD(U), SD(V), SD(W), in km/s, heliocentric
        1D array of galactic lat/long coordinates (b,l) (l=0 is towards galactic center, increasing to the east; b=0 
            is in galactic plane, +90o above galactic plane, -90o below)
        1D array of galactic cartesian Earth-centered XYZ spatial coords (X: l=0,b=0; Y: l=270,b=0; Z:b=+90)
    Written by: Logan Pearce, 2018
    """
    import numpy as np
    from numpy import sin,cos,tan
    # Unit conversions:
    r,d = np.radians(R),np.radians(D)
    pr = (PR/1000.)*cos(d)
    pd = PD/1000.
    para = plx/1000.
    
    sig_pr = (sig_PR/1000.)
    sig_pd = sig_PD/1000.
    sig_para = sig_plx/1000.
    
    k = 4.74057
    T = np.array([[-0.06699,-0.87276,-0.48354],
                  [0.49273,-0.45035,0.74458],
                  [-0.86760,-0.18837,0.46020]])
    A = np.array([[cos(r)*cos(d),-sin(r),-cos(r)*sin(d)],
                  [sin(r)*cos(d),cos(r),-sin(r)*sin(d)],
                  [sin(d),0.,cos(d)]])
    B = np.matmul(T,A)
    out1 = np.matmul(B, np.array([RV,(k*pr/para),(k*pd/para)]))
    
    C = np.square(B)
    varmat = np.array([sig_RV**2,
                       ((k/para)**2)*(sig_pr**2+(pr*sig_para/para)**2),
                      ((k/para)**2)*(sig_pd**2+(pd*sig_para/para)**2)])
    cross_matrix = np.array([B[0,1]*B[0,2],
                            B[1,1]*B[1,2],
                            B[2,1]*B[2,2]])
    cross_term = 2*pr*pd*(k**2)*(sig_para**2)/(para**4)*cross_matrix
    out2 = np.matmul(C,varmat)+cross_term
    out = np.array([out1,np.sqrt(out2)])
    
    # Computing b and l:
    blmatrix = np.matmul(T,np.array([cos(np.radians(D))*cos(np.radians(R)),
                                    cos(np.radians(D))*sin(np.radians(R)),
                                    sin(np.radians(D))]))
    b = np.arcsin(blmatrix[2])
    sinl = blmatrix[1]/cos(b)
    cosl = blmatrix[0]/cos(b)
    ll = np.arcsin(sinl)
    if sinl >= 0 and cosl >=0:
        l = np.degrees(ll)
    elif sinl >=0 and cosl <= 0:
        l = 90+np.degrees(ll)
    elif sinl <= 0 and cosl <=0:
        l = 270+np.degrees(ll)
    elif sinl <= 0 and cosl >=0:
        l = 360+np.degrees(ll)
    b = np.degrees(b)
    latlong = np.array([b,l])
    
    # Computing galactic cartesian coords:
    d = 1/(para/1000)
    X = d*cos(np.radians(l))*sin(np.radians(90.-b))
    Y = d*sin(np.radians(l))*sin(np.radians(90.-b))
    Z = d*cos(np.radians(90.-b))
    cartesian = np.array([X,Y,Z])
    
    return out, latlong, cartesian

def UVW_to_propermotions(UVW,R,D,plx):
    import numpy as np
    r,d = np.radians(R),np.radians(D)
    k = 4.74057
    T = np.array([[-0.06699,-0.87276,-0.48354],
                    [0.49273,-0.45035,0.74458],
                    [-0.86760,-0.18837,0.46020]])
    from numpy import sin,cos
    A = np.array([[cos(r)*cos(d),-sin(r),-cos(r)*sin(d)],
                [sin(r)*cos(d),cos(r),-sin(r)*sin(d)],
                [sin(d),0.,cos(d)]])
    B = np.matmul(T,A)
    Binv = np.linalg.inv(B)

    M = np.matmul(Binv,UVW)
    B_RV = M[0]
    B_pmra = (M[1] * plx / k)/cos(d)
    B_pmdec = M[2] * plx / k
    return B_RV,B_pmra,B_pmdec

def deproject_apparent_velocities(sourceid1,sourceid2, pmra = [], pmdec = [], rv = [], \
                                  catalog='gaiaedr3.gaia_source', return_old_vels = False):
    ''' For widely separated binaries for which projection effects might be significant, "deproject"\
        the apparent velocities of one of the stars ('B') by converting them into UVW space velocities, \
        then computing new pm/RV at the same ra/dec as the other star ('A').
        
    Args:
        sourceid1, sourceid2 (int): Gaia source ids of the two stars
        pmra, pmdec, RV (tuple): option to supply user-determined apparent velocities for star B
        catalog (str): Gaia catalog from which to draw astrometry.  Default = EDR3
        return_old_vels (bool): if True, also return the original velocities. Default = False
        
    Returns:
    '''
    from astroquery.gaia import Gaia
    from myastrotools.astrometry import space_velocity, UVW_to_propermotions, MonteCarloIt
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000
    
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(sourceid1))
    A = job.get_results()
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(sourceid2))
    B = job.get_results()
    
    plx = B['parallax'][0]
    sig_plx = B['parallax_error'][0]
    R,sig_R, D,sig_D = B['ra'][0],B['ra_error'][0]*mas_to_deg,B['dec'][0],B['dec_error'][0]*mas_to_deg
    if len(pmra) == 0:
        PR, sig_PR = B['pmra'][0],B['pmra_error'][0] #mas/yr
        PD, sig_PD = B['pmdec'][0],B['pmdec_error'][0] #mas/yr
        if B['dr2_radial_velocity'].mask[0]:
            print('No RV entry for this source, cannot complete deprojection')
            return
        else:
            RV, sig_RV = B['dr2_radial_velocity'][0],B['dr2_radial_velocity_error'][0] #km/s
    else:
        PR, sig_PR = pmra[0], pmra[1]
        PD, sig_PD = pmdec[0], pmdec[1]
        RV, sig_RV = rv[0], rv[1]
    
    # Compute space velocity of Star B:
    RR = MonteCarloIt([R,sig_R])
    DD = MonteCarloIt([D,sig_D])
    PP = MonteCarloIt([plx,sig_plx])
    PPRR = MonteCarloIt([PR,sig_PR])
    PPDD = MonteCarloIt([PD,sig_PD])
    RRVV = MonteCarloIt([RV,sig_RV])
    UVW, sig_UVW = np.zeros((3,len(RR))),np.zeros((3,len(RR)))
    for i in range(len(RR)):
        UVW[:,i], sig_UVW[:,i] = space_velocity(RR[i],DD[i],PPRR[i],sig_PR,PPDD[i],\
                                                sig_PD,RRVV[i],sig_RV,PP[i],sig_plx)[0]
    # Pull out A's info:  
    A_plx = A['parallax'][0]
    A_sig_plx = A['parallax_error'][0]
    A_R,A_sig_R, A_D,A_sig_D = A['ra'][0],A['ra_error'][0]*mas_to_deg,A['dec'][0],\
        A['dec_error'][0]*mas_to_deg
    A_RR = MonteCarloIt([A_R,A_sig_R])
    A_DD = MonteCarloIt([A_D,A_sig_D])
    # Do not change the parallax for B to A's parallax
    #A_PP = MonteCarloIt([A_plx,A_sig_plx])
    PMS = np.zeros((3,len(A_RR)))
    # Compute velocities for B at the location of star A:
    for i in range(len(A_RR)):
        PMS[:,i] = UVW_to_propermotions(UVW[:,i],A_RR[i],A_DD[i],PP[i])
        
    new_pmra = (np.mean(PMS[1]),np.std(PMS[1]))
    new_pmdec = (np.mean(PMS[2]),np.std(PMS[2]))
    new_rv = (np.mean(PMS[0]),np.std(PMS[0]))
    
    if return_old_vels:
        return new_pmra, new_pmdec, new_rv, (PR, sig_PR), (PD, sig_PD), (RV, sig_RV)
    
    return new_pmra, new_pmdec, new_rv

############################################################################################################
################## Functions for common proper motion arguements ###########################################
############################################################################################################

def are_projection_effects_relevant(sourceid1,sourceid2):
    from myastrotools.tools import physical_separation
    from myastrotools.tools import get_seppa, get_distance
    import astropy.units as u
    ''' Determine the separation between two Gaia objects in parsecs,and compare to \
        the findings of El-Badry et al. 2019 that project effects cause a significant \
        difference between apparent and true velocities if sep >~ 0.1 pc.

    Args:
        sourceid1,sourceid2 (int): Gaia EDR3 source ids
    
    Returns:
        str: string of comparison to El-Badry 2019 finding
        flt: separation in parsecs
    '''

    dist = get_distance(sourceid1)
    seppa = get_seppa(sourceid1,sourceid2)
    sep_au = physical_separation(dist[0]*u.pc,seppa[0]*u.arcsec)
    sep_pc = sep_au.to(u.pc)
    if sep_pc.value >= 0.1:
        out = 'yes'
    elif sep_pc.value > 0.05 and sep_pc.value < 0.1:
        out = 'maybe'
    else:
        out = 'probably not'
    print(out, ', Sep = ',sep_pc)
    return out, sep_pc

def probability_of_chance_alignment(sourceid1, sourceid2, 
                                    radius = 30, 
                                    plx_cut = 0.5,
                                    pm_cut = 1,
                                    adql_results_csv = [],
                                    catalog='gaiaedr3.gaia_source',
                                    deproject = True
                                   ):
    ''' Use the Gaia catalog to estimate the probability of finding a star with similar \
        parallax and proper motion as the central star within the separation of the candidate\
        binary companion, given the density of objects on the sky within a set radius.

    Args:
        sourceid1, sourceid2 (int): Gaia source ids of central star and candidate companion
        radius (flt): radius for cone search in degrees
        plx_cut (flt): boundaries for parallax cut will be target star's plx +/- cut
        pm_cut (flt): boundaries for pm cut will be target star's pmra/pmdec +/- cut
        adql_results_csv (str): ADQL searches performed through astroquery truncate at 2000\
                results. If there are more than 2000 results, perform the search using the ADQL \
                interface in the Gaia archive, donwload the results csv, and provide the path to \
                the file via this keyword.
        catalog (str): Gaia catalog to query. Default = EDR3

    '''
    from myastrotools.tools import deproject_apparent_velocities, update_progress
    from myastrotools.tools import get_seppa
    from astroquery.gaia import Gaia
    import astropy.units as u
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000

    # Get target star's info:
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(sourceid1))
    j = job.get_results()
    RA1, RA1err, Dec1, Dec1err = j['ra'],j['ra_error']*mas_to_deg,j['dec'],j['dec_error']*mas_to_deg
    pmRA1,pmRA1err,pmDec1,pmDec1err = j['pmra'],j['pmra_error'],j['pmdec'],j['pmdec_error']
    parallax, parallax_error = j['parallax'],j['parallax_error']
    
    if len(adql_results_csv) == 0:
        # Perform ADQL search of Gaia catalog of all objects
        # within radius within parallax cut:
        search_string = "SELECT DISTANCE( \
        POINT('ICRS', ra, dec), \
        POINT('ICRS', "+str(RA1.data[0])+", "+str(Dec1.data[0])+")) AS dist, * \
        FROM gaiaedr3.gaia_source \
        WHERE 1=CONTAINS( \
        POINT('ICRS', ra, dec), \
        CIRCLE('ICRS', "+str(RA1.data[0])+", "+str(Dec1.data[0])+", "+str(radius)+")) \
        AND parallax < "+str(parallax.data[0] + plx_cut)+"  AND parallax > "+str(parallax.data[0] - plx_cut)+" \
        ORDER BY dist ASC"
        try:
            job = Gaia.launch_job(search_string)
            g = job.get_results()
            print('N objects with similar plx:',len(g))
            if len(g) == 2000:
                print('Table truncated, use search string on ADQL interface and upload csv of results.')
                print(search_string)
                return
        except ValueError:
            print('No table found, try adjusting cuts or using ADQL interface.')
            print(search_string)
            return
    else:
        import pandas as pd
        g = pd.read_csv(adql_results_csv)
    
    # Compute probability of chance alignment within the separation of the companion, given the densty of
    # objects in the area within the similar parallax:
    size_of_area = np.pi*radius**2
    density_of_objects_with_same_plx = (len(g)) / size_of_area
    density_of_objects_with_same_plx, size_of_area # objects per square degree
    seppa = get_seppa(sourceid1,sourceid2)
    rho = seppa[0]*u.arcsec.to(u.deg)
    area_of_circle_at_comp_sep = np.pi*rho**2
    n_objects_within_circle = 2
    density_of_objects_in_circle = n_objects_within_circle/area_of_circle_at_comp_sep

    print('Probability of chance alignment of object that close given the density of objects in the nearby area the same plx:')
    print(density_of_objects_with_same_plx/(density_of_objects_with_same_plx+density_of_objects_in_circle))
    prob_same_plx = density_of_objects_with_same_plx/(density_of_objects_with_same_plx+\
                                                      density_of_objects_in_circle)
    print()
    
    if deproject:
        # project velocities of target star onto locations of all stars passing plx cuts:
        g['deproj_pmra'], g['deproj_pmdec'], g['deproj_rv'] = np.nan,np.nan,np.nan
        print("Deprojecting objects' velocities")
        for i in range(len(g)):
            new_pmra, new_pmdec, new_rv = deproject_apparent_velocities(g['source_id'][i],sourceid1)
            g['deproj_pmra'][i], g['deproj_pmdec'][i], g['deproj_rv'][i] = new_pmra[0], new_pmdec[0], new_rv[0]
            update_progress(i,len(g))
        ind = np.where((g['deproj_pmra'] > j['pmra'][0] - pm_cut) & (g['deproj_pmra'] < j['pmra'][0] + pm_cut))[0]
        g2 = g.loc[ind]
        g2 = g2.reset_index(drop=True)
        ind2 = np.where((g2['deproj_pmdec'] > j['pmdec'][0] - pm_cut) & \
                    (g2['deproj_pmdec'] < j['pmdec'][0] + pm_cut))[0]
        g3 = g2.loc[ind2]
    else:
        ind = np.where((g['pmra'] > j['pmra'][0] - pm_cut) & (g['pmra'] < j['pmra'][0] + pm_cut))[0]
        g2 = g.loc[ind]
        g2 = g2.reset_index(drop=True)
        ind2 = np.where((g2['pmdec'] > j['pmdec'][0] - pm_cut) & \
                    (g2['pmdec'] < j['pmdec'][0] + pm_cut))[0]
        g3 = g2.loc[ind2]
    n_objects_within_area_similar_plx_and_pm = len(g3)
    print()
    print('N objects with similar plx and pm:',len(g3))
    density_of_objects_with_similar_plx_and_pm = n_objects_within_area_similar_plx_and_pm / size_of_area
    print('Probability of an object that close given the density of objects in search region with the similar pm and similar parallax:')
    print(density_of_objects_with_similar_plx_and_pm/(density_of_objects_with_similar_plx_and_pm+\
                                                      density_of_objects_in_circle))
    prob_same_plx_pm = density_of_objects_with_similar_plx_and_pm/(density_of_objects_with_similar_plx_and_pm+\
                                                      density_of_objects_in_circle)
                                                      
    return prob_same_plx_pm, prob_same_plx

def get_ell_or_hypo(sourceid1,sourceid2,mass1,mass2, N = 10000):
    """Given Gaia source ids and masses, compute
        observables XYZ position and velocity, and determine the fraction of elliptical and hyperbolic
        orbits consistent with relative velocities.

        Args:
            sourceid1/sourceid2 (int): Gaia source ids for two object
            mass1/mass2 (arr): array of [mass,error] in solar masses.
            N (int): number of samples for Monte Carlo sampling. Default = 10000
        Returns:
            flt: fraction of elliptical orbits
            flt: fraction of hyperbolic orbits
            1xN arr: velocities from Monte Carlo sample
            1xN arr: escape velocities from Monte Carlo sample
            1xN arr: orbits from Monte Carlo sample, either 'e' or 'h'
            
        Written by Logan Pearce, 2021, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    from astroquery.gaia import Gaia
    from myastrotools.gaia_tools import edr3ToICRF, distance
    from myastrotools.astrometry import MonteCarloIt, keplersconstant
    from myastrotools.astrometry import turn_gaia_into_physical, masyr_to_kms
    
    catalog = 'gaiaedr3.gaia_source'
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(sourceid1))
    j = job.get_results()
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(sourceid2))
    k = job.get_results()
    
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000

    plx = j['parallax'][0]
    sig_plx = j['parallax_error'][0]
    R,sig_R, D,sig_D = j['ra'][0],j['ra_error'][0]*mas_to_deg,j['dec'][0],\
                j['dec_error'][0]*mas_to_deg
    PR, sig_PR = j['pmra'][0],j['pmra_error'][0] #mas/yr
    PD, sig_PD = j['pmdec'][0],j['pmdec_error'][0] #mas/yr
    RV, sig_RV = j['dr2_radial_velocity'][0],j['dr2_radial_velocity_error'][0] #km/s

    PLX1 = MonteCarloIt([plx,sig_plx], N = N)
    RR1 = MonteCarloIt([R,sig_R], N = N)
    DD1 = MonteCarloIt([D,sig_D], N = N)
    PP1 = MonteCarloIt([plx,sig_plx], N = N)
    PPRR1 = MonteCarloIt([PR,sig_PR], N = N)
    PPDD1 = MonteCarloIt([PD,sig_PD], N = N)
    RRVV1 = MonteCarloIt([RV,sig_RV], N = N)

    plx = k['parallax'][0]
    sig_plx = k['parallax_error'][0]
    R,sig_R, D,sig_D = k['ra'][0],k['ra_error'][0]*mas_to_deg,k['dec'][0],\
                k['dec_error'][0]*mas_to_deg
    PR, sig_PR = k['pmra'][0],k['pmra_error'][0] #mas/yr
    PD, sig_PD = k['pmdec'][0],k['pmdec_error'][0] #mas/yr
    RV, sig_RV = k['dr2_radial_velocity'][0],k['dr2_radial_velocity_error'][0] #km/s

    PLX2 = MonteCarloIt([plx,sig_plx], N = N)
    RR2 = MonteCarloIt([R,sig_R], N = N)
    DD2 = MonteCarloIt([D,sig_D], N = N)
    PP2 = MonteCarloIt([plx,sig_plx], N = N)
    PPRR2 = MonteCarloIt([PR,sig_PR], N = N)
    PPDD2 = MonteCarloIt([PD,sig_PD], N = N)
    RRVV2 = MonteCarloIt([RV,sig_RV], N = N)

    thing = mass1.copy()
    m1 = MonteCarloIt([thing[0],thing[1]], N = N)*u.Msun
    thing = mass2.copy()
    m2 = MonteCarloIt([thing[0],thing[1]], N = N)*u.Msun
    kep = keplersconstant(m1,m2)

    pmRACorrected2,pmDecCorrected2 = edr3ToICRF(k[0]['pmra'],k[0]['pmdec'],k[0]['ra'],k[0]['dec'],\
                                                k[0]["phot_g_mean_mag"])
    PPRR2_corr = MonteCarloIt([pmRACorrected2,sig_PR], N = N)
    PPDD2_corr = MonteCarloIt([pmDecCorrected2,sig_PD], N = N)
    
    velocity, escape_velocity, orbit_type = np.zeros(N),np.zeros(N),np.zeros(N, dtype=str)
    for i in range(RRVV2.shape[0]):
        ra = (RR2[i]*deg_to_mas - RR1[i]*deg_to_mas) * np.cos(np.radians(np.mean([DD1[i],DD2[i]])))
        ra = (ra*u.mas).to(u.arcsec)
        dec = ((DD2[i] - DD1[i])*u.deg).to(u.arcsec).value
        dist1 = distance(PLX1[i],sig_plx)
        dist2 = distance(PLX2[i],sig_plx)
        ra = (ra*np.mean([dist1[0],dist2[0]]))
        dec = (dec*np.mean([dist1[0],dist2[0]]))
        z = ((dist2[0] - dist1[0])*u.pc).to(u.AU)
        pos = np.array([ra.value,dec,z.value])

        pmra = masyr_to_kms(PPRR2[i],[PLX2[i],sig_plx]) - masyr_to_kms(PPRR1[i],[PLX1[i],sig_plx])
        pmdec = masyr_to_kms(PPDD2[i],[PLX2[i],sig_plx]) - masyr_to_kms(PPDD1[i],[PLX1[i],sig_plx])
        rv = -(RRVV2[i] - RRVV1[i])
        vel = np.array([pmra,pmdec,rv])

        v = np.linalg.norm(vel)
        r = np.linalg.norm(pos*u.AU)
        v_esc = np.sqrt(2*kep[i]/r).decompose().to(u.km/u.s)

        if v >= v_esc.value:
            orbit = 'hyperbolic'
        else: orbit = 'elliptical'

        pmra = masyr_to_kms(PPRR2_corr[i],[PLX2[i],sig_plx]) - masyr_to_kms(PPRR1[i],[PLX1[i],sig_plx])
        pmdec = masyr_to_kms(PPDD2_corr[i],[PLX2[i],sig_plx]) - masyr_to_kms(PPDD1[i],[PLX1[i],sig_plx])

        rv = -(RRVV2[i] - RRVV1[i])
        vel = np.array([pmra,pmdec,rv])

        v = np.linalg.norm(vel)
        r = np.linalg.norm(pos*u.AU)
        v_esc = np.sqrt(2*kep[i]/r).decompose().to(u.km/u.s)

        if v >= v_esc.value:
            orbit = 'hyperbolic'
        else: orbit = 'elliptical'
        velocity[i], escape_velocity[i], orbit_type[i] = v,v_esc.value,orbit
        
    hyper = np.where(orbit_type == 'hyperbolic')[0]
    ell = np.where(orbit_type == 'elliptical')[0]
    fraction_hyperbolic = hyper.shape[0] / orbit_type.shape[0]
    fraction_elliptical = ell.shape[0] / orbit_type.shape[0]
    return fraction_elliptical, fraction_hyperbolic, velocity, escape_velocity, orbit_type

############################################################################################################
###################################### Atmospheric corrections #############################################
############################################################################################################

def zenith_correction_factor(zo):
    fz_dict = {'0':0.,'10':0.,'20':0.,'30':0.,'35':0.,'40':2e-4,'45':6e-4,'50':12e-4,\
         '55':21e-4,'60':34e-4,'65':56e-4,'70':97e-4} #create a dictionary for the lookup table
    gz_dict = {'0':4e-4,'10':4e-4,'20':4e-4,'30':5e-4,'35':5e-4,'40':5e-4,'45':6e-4,'50':6e-4,\
         '55':7e-4,'60':8e-4,'65':10e-4,'70':13e-4}
    if zo >= 0 and zo <= 30:
        z = str(int(np.round(zo,decimals=-1))) #round to nearest 10s
        fz = fz_dict[z]
        gz = gz_dict[z]
        return fz,gz
    elif zo > 30 and zo <= 70:
        z = str(int(np.round(zo/5.0)*5.0)) #round to nearest 5s
        fz = fz_dict[z]
        gz = gz_dict[z]
        return fz,gz
    else:
        return 'Atmospheric correction not required, zenith greater than 70 deg'

def atm_corr(zo,p,t,lamb,hum):
    fz,gz = zenith_correction_factor(zo)
    zo = np.radians(zo)
    po = 101325. #Pa
    Ro = 60.236*np.tan(zo)-0.0675*(np.tan(zo)**3) # In arcseconds
    # Atm correction:
    F = (1.-(0.003592*(t-15.))-(5.5e-6*((t-7.5)**2)))*(1.+fz)
    G = (1.+(0.943e-5*(p-po))-((0.78e-10)*((p-po)**2)))*(1.+gz)
    R = Ro*(p/po)*(1.055216/(1.+0.00368084*t))*F*G
    # Chromatic effects:
    R = R*(0.98282+(0.005981/(lamb**2)))
    # Water vapor correction:
    f = hum/100. #convert percent to decimal
    # Calculate saturation vapor pressure:
    # Using Buck eqn (https://en.wikipedia.org/wiki/Vapour_pressure_of_water):
    Psat = 0.61121*np.exp((18.678-(t/234.5))*(t/(257.14+t))) # in kPa
    Psat = Psat*1000 #convert to Pa
    # Calculate water vapor partial pressure: (http://www.engineeringtoolbox.com/relative-humidity-air-d_687.html)
    Pw = f*Psat
    R = R*(1-0.152e-5*Pw - 0.55e-9*(Pw**2)) # R is in arcseconds
    zo = np.degrees(zo)
    z = zo+(R/3600.)  # Convert R to degrees
    return R/3600.,z


lamb_dict = {'z':1.0311,'Y':1.0180,'J':1.248,'H':1.633,'K':2.196,'Ks':2.146,'Kp':2.124,\
             'Lw':3.5197,'Lp':3.776,'Ms':4.670} #Dictionary of median wavelength for NIRC2 filters in micrometers

def ecliptic_to_equatorial(lon, lat):
    ''' Convert array from ecliptic to equatorial coordinates using astropy's SkyCoord object
        Inputs:
            lon, lat [deg] (array): ecliptic longitude (lambda) and ecliptic latitude (beta)
        Returns:
            newRA, newDec [deg] (array): array points in equatorial RA/Dec coordinates
    '''
    from astropy.coordinates import SkyCoord
    import numpy as np
    # Compute ecliptic motion to equatorial motion:
    newRA, newDec = np.zeros(len(lon)),np.zeros(len(lon))
    for i in range(len(lon)):
        obj2 = SkyCoord(lon = lon[i],\
                    lat = lat[i], \
                    frame='geocentrictrueecliptic', unit='deg') 
        obj2 = obj2.transform_to('icrs')
        newRA[i] = obj2.ra.deg
        newDec[i] = obj2.dec.deg
    return newRA,newDec


def cpm_plot(RA, RAerr, Dec, Decerr, pmRA, pmRAerr, pmDec, pmDecerr, parallax, parallaxerr, ref_date, \
                  obsdate, obs_RAs, obs_RAs_err, obs_Decs, obs_Decs_err, labels, \
                  ref_RA_offset = 0,
                  ref_Dec_offset = 0,
                  time_interval = [3,12],
                  n_times = 800,
                  plt_xlim=None,
                  plt_ylim=None,
                  xlabel = r'$\Delta$ RA [mas]',
                  ylabel = r'$\Delta$ Dec [mas]',
                  marker = ['^','o'],
                  marker_size = [100,100],
                  fontsize = 15,
                  tick_labelsize = 12,
                  labelsize = 13,
                  label_offset = [-15,0],
                  colors = ['purple','mediumblue','darkcyan','green','orange','red'],
                  alpha=0.6,
                  figsize = (8,8),
                  plt_style = 'default',
                  write_to_file = False,
                  output_name = 'cpm.pdf',
                  form = 'pdf'
                 ):
    ''' Test for common proper motion of a candidate companion by plotting the track the cc would have
        been observed on if it were a background object and not graviationally bound.
        Requires: astropy, numpy, matplotlib
        Inputs:
            RA/Dec + errors [deg] (flt): RA/Dec of host star
            pmRA, pmDec + errors [mas/yr] (flt): - proper motion of host star.  Use the negative of reported
                values because we treat the star as unmoving and will observe the apparent motion of the cc
            parallax + error [mas] (flt): parallax
            ref_date [decimal year] (astropy Time object): reference date
            obsdate [decimal year] (array): array of dates of observations
            obs_RAs, obs_Decs + errors [mas] (array): array of observed RA/Dec offsets of companion to host star
            labels (str array): strings to label plot points 
            ref_RA_offset, ref_Dec_offset [mas] (flt): 'zero point' reference RA/Dec offset for companion
                to host
            time_interval [yrs] (array):  Number of years [below,above] reference date to compute plot.  
                EX: time_interval = [3,12] plots the background star track for a period extending 3 years 
                before and 12 years after the reference date.
            n_times (int): number of time points to compute values for plot.  Default = 800
            plt_xlim, plt_ylim [mas] (array): axis limits [min,max].  Default = autoscale
            xlabel, ylabel (str): X and Y axis labels
            marker (array): markers to use for prediction points [0] and observed points [1]
            marker_size (array): size of markers for predicition points [0] and observed points [1]
            fontsize, tick_labelsize, labelsize (int): font sizes for axis labels, tick labels, and epoch labels
            label_offset (array): offset the epoch labels from the points in the x [0] and the y [1] directions
            colors (list): colors to use for epochs
            alpha (flt): opacity of observation points
            figsize (tuple): figure size
            plt_style (str): specify the name of the matplotlib stylesheet to use
            write_to_file (bool): write out plot to file.  Default = False
            output_name (str): name of output figure
            form (str): format of written out figure
        Returns:
            fig (matplotlib figure): plot of proper motion track (blue), predicted background star observation points
                (default: triangles, solid), actualy observation points (default: circles, 60% opacity).
            pred_dRA_total, pred_dDec_total (flt): predicted RA/Dec offsets if cc were a background object
    '''
    from astropy.time import Time
    from astropy.coordinates import SkyCoord, EarthLocation, AltAz, GeocentricTrueEcliptic
    from astropy import units as u
    import matplotlib.pyplot as plt
    from cycler import cycler
    import numpy as np
    from myastrotools.astrometry import ecliptic_to_equatorial
    
    plt.style.use(plt_style)
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000.
    ############### Compute track: ###################
    # Define a time span around reference date:
    delta_time = np.linspace(-time_interval[0], time_interval[1], n_times)*u.yr
    times = ref_date + delta_time
    
    # Compute change in RA/Dec during time interval due to proper motion only:
    dRA, dDec = (pmRA)*(delta_time.value), (pmDec)*(delta_time.value)
    
    # Compute motion in the ecliptic coords due to parallactic motion:
    # Make a sky coord object in RA/Dec:
    obj = SkyCoord(ra = RA, dec = Dec, frame='icrs', unit='deg'#, obstime = ref_date
                       ) 
    # Convert to ecliptic lon/lat:
    gteframe = GeocentricTrueEcliptic()
    obj_ecl = obj.transform_to(gteframe)
    
    # Angle array during a year:
    theta = (delta_time.value%1)*2*np.pi
    #Parallel to ecliptic:
    x = parallax*np.sin(theta)  
    #Perp to ecliptic
    y = parallax*np.sin(obj_ecl.lat.rad)*np.cos(theta)
    
    # Compute ecliptic motion to equatorial motion:
    print('Plotting... this part may take a minute.')
    new_RA, new_Dec = ecliptic_to_equatorial(obj_ecl.lon.deg+x*mas_to_deg, \
                                           obj_ecl.lat.deg+y*mas_to_deg)
    # Compute change in RA/Dec for each time point in mas:
    delta_RA, delta_Dec = (new_RA-RA)*deg_to_mas,(new_Dec-Dec)*deg_to_mas
    
    #Put it together:
    dRA_total = delta_RA + dRA + ref_RA_offset
    dDec_total = delta_Dec + dDec + ref_Dec_offset
    
    ############# Compute prediction: #############
    ### Where the object would have been observed were it a background object
    
    # Compute how far into each year the observation occured:
    pred_time_delta = (obsdate - np.floor(obsdate))
    pred_theta = (pred_time_delta)*2*np.pi
    
    # Compute ecliptic motion:
    pred_x = parallax*np.sin(pred_theta)  #Parallel to ecliptic
    pred_y = parallax*np.sin(obj_ecl.lat.rad)*np.cos(pred_theta)  #Perp to ecliptic
    
    # Convert to RA/Dec:
    pred_new_RA, pred_new_Dec = ecliptic_to_equatorial(obj_ecl.lon.deg+pred_x*mas_to_deg, \
                                           obj_ecl.lat.deg+pred_y*mas_to_deg)
    pred_delta_RA, pred_delta_Dec = (pred_new_RA-RA)*deg_to_mas,(pred_new_Dec-Dec)*deg_to_mas
    
    # Compute location due to proper motion:
    pred_dRA, pred_dDec = (pmRA)*(obsdate-ref_date.value), (pmDec)*(obsdate-ref_date.value)

    # Put it together:
    pred_dRA_total = -pred_delta_RA + pred_dRA + ref_RA_offset
    pred_dDec_total = -pred_delta_Dec + pred_dDec + ref_Dec_offset

    #################### Draw plot: #################
    plt.rcParams['ytick.labelsize'] = tick_labelsize
    plt.rcParams['xtick.labelsize'] = tick_labelsize
    custom_cycler = (cycler(color=colors))
    fig = plt.figure(figsize = figsize)
    #plt.plot(dRA_total,dDec_total, lw=3, color='lightgrey', alpha = 0.5, zorder = 0)
    plt.plot(dRA_total,dDec_total, zorder = 1)
    plt.gca().set_prop_cycle(custom_cycler)
    for i in range(len(pred_dRA)):
        plt.scatter(pred_dRA_total[i], pred_dDec_total[i], marker = marker[0], s=marker_size[0], zorder=2, 
                    edgecolors='black')
        plt.annotate(
            labels[i],
            xy=(pred_dRA_total[i], pred_dDec_total[i]), xytext=(label_offset[0], label_offset[1]),
            textcoords='offset points', ha='right', va='bottom', fontsize=labelsize, color=colors[i])
    for i in range(len(obs_RAs)):
        plt.scatter(obs_RAs[i], obs_Decs[i], edgecolors="black", marker = marker[1], s=marker_size[1], \
                        alpha=alpha, zorder = 10)
        plt.errorbar(obs_RAs[i], obs_Decs[i], xerr= obs_RAs_err[i], yerr=obs_Decs_err[i], ls='none',
                 elinewidth=1,capsize=0, ecolor='black',zorder=10)
    if plt_xlim != None:
        plt.xlim(plt_xlim[0],plt_xlim[1])
    if plt_ylim != None:
        plt.ylim(plt_ylim[0],plt_ylim[1])
    plt.xlabel(xlabel, fontsize = fontsize)
    plt.ylabel(ylabel, fontsize = fontsize)
    plt.gca().invert_xaxis()
    plt.grid(ls=':')
    plt.tight_layout()
    if write_to_file == True:
        plt.savefig(output_name, format=form)
    
    return fig, pred_dRA_total, pred_dDec_total

############################################################################################################
###################################### Atmospheric corrections #############################################
############################################################################################################

def zenith_correction_factor(zo):
    fz_dict = {'0':0.,'10':0.,'20':0.,'30':0.,'35':0.,'40':2e-4,'45':6e-4,'50':12e-4,\
         '55':21e-4,'60':34e-4,'65':56e-4,'70':97e-4} #create a dictionary for the lookup table
    gz_dict = {'0':4e-4,'10':4e-4,'20':4e-4,'30':5e-4,'35':5e-4,'40':5e-4,'45':6e-4,'50':6e-4,\
         '55':7e-4,'60':8e-4,'65':10e-4,'70':13e-4}
    if zo >= 0 and zo <= 30:
        z = str(int(np.round(zo,decimals=-1))) #round to nearest 10s
        fz = fz_dict[z]
        gz = gz_dict[z]
        return fz,gz
    elif zo > 30 and zo <= 70:
        z = str(int(np.round(zo/5.0)*5.0)) #round to nearest 5s
        fz = fz_dict[z]
        gz = gz_dict[z]
        return fz,gz
    else:
        return 'Atmospheric correction not required, zenith greater than 70 deg'

def atm_corr(zo,p,t,lamb,hum):
    fz,gz = zenith_correction_factor(zo)
    zo = np.radians(zo)
    po = 101325. #Pa
    Ro = 60.236*np.tan(zo)-0.0675*(np.tan(zo)**3) # In arcseconds
    # Atm correction:
    F = (1.-(0.003592*(t-15.))-(5.5e-6*((t-7.5)**2)))*(1.+fz)
    G = (1.+(0.943e-5*(p-po))-((0.78e-10)*((p-po)**2)))*(1.+gz)
    R = Ro*(p/po)*(1.055216/(1.+0.00368084*t))*F*G
    # Chromatic effects:
    R = R*(0.98282+(0.005981/(lamb**2)))
    # Water vapor correction:
    f = hum/100. #convert percent to decimal
    # Calculate saturation vapor pressure:
    # Using Buck eqn (https://en.wikipedia.org/wiki/Vapour_pressure_of_water):
    Psat = 0.61121*np.exp((18.678-(t/234.5))*(t/(257.14+t))) # in kPa
    Psat = Psat*1000 #convert to Pa
    # Calculate water vapor partial pressure: (http://www.engineeringtoolbox.com/relative-humidity-air-d_687.html)
    Pw = f*Psat
    R = R*(1-0.152e-5*Pw - 0.55e-9*(Pw**2)) # R is in arcseconds
    zo = np.degrees(zo)
    z = zo+(R/3600.)  # Convert R to degrees
    return R/3600.,z


lamb_dict = {'z':1.0311,'Y':1.0180,'J':1.248,'H':1.633,'K':2.196,'Ks':2.146,'Kp':2.124,\
             'Lw':3.5197,'Lp':3.776,'Ms':4.670} #Dictionary of median wavelength for NIRC2 filters in micrometers



############################################################################################################
############################################ Unit Conversions ##############################################
############################################################################################################

def FnuToFlamb(Fnu, lamb, returnvalueonly = False):
    '''From https://www.stsci.edu/~strolger/docs/UNITS.txt:
        [Y erg/cm^2/s/A]             = 2.99792458E-05 * [X1 Jy] / [X2 A]^2

        [Y photon/cm^2/s/A]          = 1.50918896E+03 * [X1 Jy] / [X2 A]'''
    import astropy.units as u
    Fnu = Fnu.to(u.Jy)
    lamb = lamb.to(u.AA)
    flamb = 2.99792458e-5 * Fnu / (lamb**2)
    if returnvalueonly:
        return flamb.value
    else:
        return flamb.value*u.erg/u.cm**2/u.s/u.AA

def FnuToFlambInPhotons(Fnu, lamb, returnvalueonly = False):
    '''From https://www.stsci.edu/~strolger/docs/UNITS.txt:
        [Y erg/cm^2/s/A]             = 2.99792458E-05 * [X1 Jy] / [X2 A]^2

        [Y photon/cm^2/s/A]          = 1.50918896E+03 * [X1 Jy] / [X2 A]'''
    import astropy.units as u
    Fnu = Fnu.to(u.Jy)
    lamb = lamb.to(u.AA)
    flamb = 1.50918896e3 * Fnu / (lamb)
    if returnvalueonly:
        return flamb.value
    else:
        return flamb.value*1/u.cm**2/u.s/u.AA



############################################################################################################
############################################ Astrophysics ##################################################
############################################################################################################
def PlanetMass2Radius(M):
    ''' Theoretical mass-radius relation for planets and brown dwarfs by Jared
        taken from 
        https://jaredmales.github.io/mxlib-doc/group__planets.html#ga4b350ecfdeaca1bedb897db770b09789
    '''
    try:
        M = M.to(u.Mearth)
        M = M.value
    except:
        pass
    
    if M < 4.1:
        R = M**(1/3)
        
    if M >= 4.1 and M < 15.84:
        R = 0.62 * M**(0.67)
        
    if M >= 15.84 and M < 3591.1:
        coeff = [14.0211, -44.8414, 53.6554, -25.3289, 5.4920, -0.4586]
        power = [0, 1, 2, 3, 4, 5]
        R = 0
        for i in range(6):
            R += coeff[i] * (np.log10(M)**power[i])
            
    if M >= 3591.1:
        R = 32.03 * M**(-1/8)
        
    return R

def PlanetRadius2Mass(planet_radius):
    from myastrotools.tools import PlanetMass2Radius
    Ms = np.logspace(-0.5,3,1000)
    Rs = []
    for M in Ms:
        Rs.append(PlanetMass2Radius(M))
    from scipy.interpolate import interp1d
    f = interp1d(Rs,Ms)
    return f(planet_radius)

def EkerMLR(M):
    ''' Emperical mass-luminosity relation for main sequence stars from Eker et al. 2018 Table 4 
    (https://academic.oup.com/mnras/article/479/4/5491/5056185)
    
    Args:
        M (flt): mass in solar masses
        
    Returns:
        flt: log(Luminosity) in solar lum
    '''
    
    if M > 0.179 and M <= 0.45:
        logL = 2.028*np.log10(M) - 0.976
    elif M > 0.45 and M <= 0.72:
        logL = 4.572*np.log10(M) - 0.102
    elif M > 0.72 and M <= 1.05:
        logL = 5.743*np.log10(M) - 0.007
    elif M > 1.05 and M <= 2.40:
        logL = 4.329*np.log10(M) + 0.010
    elif M > 2.40 and M <= 7:
        logL = 3.967*np.log10(M) + 0.093
    elif M > 7 and M <= 31:
        logL = 2.865*np.log10(M) + 1.105
    else:
        logL = np.nan
    return logL


def EkerMRR(M):
    ''' Emperical mass-radius relation for MS stars from Eker et al. 2018 Table 5 
    (https://academic.oup.com/mnras/article/479/4/5491/5056185)
    
    Args:
        M (flt): mass in solar masses
        
    Returns:
        flt: radius in Rsun
        flt: uncertainty on radius
    '''
    if M >= 0.179 and M <= 1.5:
        R = 0.438*(M**2) + 0.479*M + 0.075
        Rerr = 0.176
    elif M > 1.5 and M <= 31:
        logL = EkerMLR(M)
        logTeff, TeffErr = EkerMTR(M)
        Teff = 10**logTeff
        R = ((10**logL)**0.5) * ((5780/Teff)**(-2))
        Rerr = 0.787
    else:
        R = np.nan
        Rerr = np.nan
    return R, Rerr
        
        
def EkerMTR(M):
    ''' Emperical mass-teff relation for MS stars from Eker et al. 2018 Table 5 
    (https://academic.oup.com/mnras/article/479/4/5491/5056185)
    
    Args:
        M (flt): mass in solar masses
        
    Returns:
        flt: Teff in K
        flt: uncertainty on teff
    '''
    if M >= 0.179 and M <= 1.5:
        R, Rerr = EkerMRR(M)
        logL = EkerMLR(M)
        Teff = (((10**(logL))**(1/4)) * R**(-0.5)) * 5780
        logTeff = np.log10(Teff)
        logTeffErr = 0.025
        
    elif M > 1.5 and M <= 31:
        logTeff = -0.170*(np.log10(M)**2) + 0.88*np.log10(M) + 3.671
        logTeffErr = 0.042
    return logTeff, logTeffErr

def saha(rho_c,T):
    ''' Assuming the composition is entirely hydrogen, determine the ratio of n+/no,
        and the ionization fraction x = n+/(no + n+)
        Inputs:
            rho_c: central density in astropy units of kg/m^3
            T: central temperature in astropy units of K
        Returns:
            ratio of n+ ions to neutral ion density
            ionization fraction x
    '''
    import numpy as np
    import astropy.units as u
    import astropy.constants as c
    ne = rho/c.m_p
    up = 2
    uo = 1
    chi = (13.6*u.eV).to(u.J)
    one = 2 * up / uo
    two = (2*np.pi*c.m_e*c.k_B*T)**(3/2) / (c.h**3)
    three = np.exp(-chi/(c.k_B*T))
    s = (one*two*three*(1/ne))
    return s.decompose(), (s/(1+s)).decompose()


def runge_kutta_4(f1, f2, xn, y1n, y2n, h, n):
    '''4th order Runge Kutta numerical integrator for two coupled
       differential equations f1 and f2
    '''
    # First step:
    k1_1 = h * f1( xn, y1n, y2n )
    k1_2 = h * f2( xn, y1n, y2n, n )
    # Second step:
    k2_1 = h * f1( xn+0.5*h, y1n+0.5*k1_1, y2n+0.5*k1_2 )
    k2_2 = h * f2( xn+0.5*h, y1n+0.5*k1_1, y2n+0.5*k1_2, n )
    # Third step:
    k3_1 = h * f1( xn+0.5*h, y1n+0.5*k2_1, y2n+0.5*k2_2 )
    k3_2 = h * f2( xn+0.5*h, y1n+0.5*k2_1, y2n+0.5*k2_2, n )
    # Fourth step:
    k4_1 = h * f1( xn+h, y1n+k3_1, y2n+k3_2 )
    k4_2 = h * f2( xn+h, y1n+k3_1, y2n+k3_2, n )
    
    return y1n + (1./6.)*k1_1 + (1./3.)*k2_1 + (1./3.)*k3_1 + (1./6.)*k4_1,  \
           y2n + (1./6.)*k1_2 + (1./3.)*k2_2 + (1./3.)*k3_2 + (1./6.)*k4_2


def RK4(x_vector, f_vector, h):
    ''' Compute 4th order Runge-Kutta for a set of n-coupled ODEs
    Args:
        x_vector ((n+1)x1 arr): vector of iterating parameter (e.g. x or t) and n ODE parameters
        f_vector (nx1 arr): vector of ODE functions for each of n parameters
        h (flt): step size
    '''
    k0 = np.zeros(len(x_vector)-1)
    k1 = np.zeros(len(x_vector)-1)
    k2 = np.zeros(len(x_vector)-1)
    k3 = np.zeros(len(x_vector)-1)
    new_x_vector = np.zeros(len(x_vector))
    for i in range(len(x_vector)-1):
        k0[i] = h*f_vector[i](*x_vector)
    for i in range(len(x_vector)-1):
        k1[i] = h*f_vector[i](x_vector[0]+0.5*h, *x_vector[1:]+0.5*k0)
    for i in range(len(x_vector)-1):
        k2[i] = h*f_vector[i](x_vector[0]+0.5*h, *x_vector[1:]+0.5*k1)
    for i in range(len(x_vector)-1):
        k3[i] = h*f_vector[i](x_vector[0]+h, *x_vector[1:]+k1)
    new_x_vector[0] = x_vector[0] + h
    new_x_vector[1:] = x_vector[1:] + (1/6)*(k0+2*k1+2*k2+k3)
    return new_x_vector

# Functions for solving the Lane Emden equation for a star using Runge-Kutta
# 4th order coupled DE solver:

def solve_lane_emden(n,dx):
    ''' Solve the Lane-Emden 2nd order DE equation
        Inputs:
            polytropic index n
            iterator step size dx
        Returns:
            x (array): array of dimensionless radius values xi for which y1 and y2 were computed
            y1 (array): values of theta for each xi value
            y2 (array): dtheta/dxi values for each xi
    '''
    import numpy as np
    from myastrotools.astrophysics import runge_kutta_4
    # Establish coupled differential equations:
    def f1(x,y1,y2):
        dy1dx = y2
        return dy1dx
    def f2(x,y1,y2,n):
        dy2dx = -y1**n - (2/x)*y2
        return dy2dx
    
    # Set boundary conditions: 
    # at r = 0, theta = 1, dtheta/dxi = 0:
    theta0 = 1.
    dthetadxi0 = 0
    #xi0 = 0.
    xi0 = 0
    # at surface, theta = 0, xi = xi1
    theta1 = 0.
    
    # define steps in x:
    x0 = xi0
    xsteps = np.arange(x0,20+dx,dx)
    
    # initialize y1 and y2 arrays:
    y1_0 = theta0
    y2_0 = dthetadxi0
    
    y1,y2 = np.zeros(len(xsteps)),np.zeros(len(xsteps))
    y1[0] = np.array([y1_0])
    y2[0] = np.array([y2_0])

    i=0
    
    for i in range(len(xsteps)):
        try:
            y1[i+1] = runge_kutta_4(f1, f2, xsteps[i+1], y1[i], y2[i], dx, n)[0]
            y2[i+1] = runge_kutta_4(f1, f2, xsteps[i+1], y1[i], y2[i], dx, n)[1]
        except:
            pass
    return xsteps, y1, y2

def mass_correction_factor(n):
    ''' Compute the mass correction factor MCF = (-3/xi_1)(dTheta/dXi|xi_1) for a polytrope
        of index n.
        M* = 4/3pi R*^3 rho_c * MCF  where:
           rho_c = Central density
           M* = Mass of star
           R* = star radius
           MCF = correction from constant density case of entire star being at rho_c density.

        Input: n [int]: polytropic index
        Returns: 
            xi1 [flt]: dimensionless radius of star
            MCF [flt]: mass correction factor = rho_bar/rho_c (avg density / central density)
    '''
    from myastrotools.astrophysics import solve_lane_emden
    import numpy as np
    x, y1, y2 = solve_lane_emden(n, 0.001)
    index = np.where(y1>0)[0][-1]
    xi1 = x[index]
    dthetadxi_at_xi1 = y2[index]
    return xi1, (-3/xi1) * dthetadxi_at_xi1

def cauchy_formula(l, a = 272.6, b = 1.22):
    """ Given wavelength in microns, return the index of refraction of air 
    at that wavelength.
    
    Args:
        l (astropy unit object): wavelength of observation.  Must be astropy unit object such as microns
            or nm
        a, b (flt): constants specific to the material
        
    Returns:
        flt: index of refraction of air at that wavelength
    """
    l = l.to(u.um)
    return 1 + (a + b/(l**2))*10**(-6)

def dispersion(l,I):
    ''' Using the Cauchy formula, compute the difference in angle between where an object should (I) and where 
    it appears to be due to atmospheric refraction
    
    Args:
        l (astropy unit object): wavelength, must be astropy unit object
        I (flt): incident angle in degrees
        
    Returns:
        astropy unit object: dispersion in arcmin 
    '''
    from myastrotools.astrphysics import cauchy_formula
    n_test = cauchy_formula(l)
    R_test = np.degrees(np.arcsin(n1 * np.sin(np.radians(I))/n_test))
    d_theta = I - R_test
    return d_theta.to(u.arcmin)

def planck_wavelength(wavelength, T):
    ''' Given wavelength as an astropy unit object and T in K, return
    Blackbody spectral radiance in W sr^-1 m^-3
    
    Args:
        wavelength (astropy unit object): wavelength, must be astropy unit object
        T (flt): temperature in K
        
    Returns:
        flt: blackbody value at that wavelength in W sr^-1 m^-3
    '''
    import astropy.units as u
    import astropy.constants as c
    h = c.h.value #m^2 kg / s
    cc = c.c.value # m/s
    kb = c.k_B.value #m^2 kg s^-2 K^-1
    wavelength_meters = wavelength.to(u.m).value
    prefactor = (2*h*(cc**2)) / (wavelength_meters**5)
    denom = np.exp( (h*cc)/(wavelength_meters*kb*T) ) - 1
    return (prefactor * 1/denom)*u.W*u.sr/(u.m**3)

############################################################################################################
############################################ Gaia Tools ####################################################
############################################################################################################

def RUWE_u0(g_mag=0, bp_rp=0):
    '''Adapted from https://github.com/adrn/pyia/blob/master/pyia/ruwetools.py
    Return the normalization factor for the re-normalised unit weight error for a Gaia DR2 solution.
    Tables and techincal note on RUWE found here: https://www.cosmos.esa.int/web/gaia/dr2-known-issues
    '''
    import numpy as np
    import os
    from scipy.interpolate import RectBivariateSpline
    # The shape of the input files:  There are 111 colors for each of the
    # 1741 magnitudes
    ngmagbins = 1741
    ncolorbins = 111
    if g_mag != 0 and bp_rp != 0:
        # If you have both gmag and bprp data use this table:
        u0data = np.genfromtxt('/Users/loganpearce/anaconda3/lib/python3.7/site-packages/myastrotools/table_u0_g_col.txt', names=['g_mag', 'bp_rp', 'u0'], skip_header=1, delimiter=',')
        # Reshape so that each column is strictly increasing:
        gmagmesh = np.reshape(u0data['g_mag'], (ngmagbins, ncolorbins))
        bprpmesh = np.reshape(u0data['bp_rp'], (ngmagbins, ncolorbins))
        u0mesh = np.reshape(u0data['u0'], (ngmagbins, ncolorbins))
        gmag = gmagmesh[:, 0]
        bprp = bprpmesh[0, :]
        # Create the bivariate spline function:
        u0spline = RectBivariateSpline(gmag, bprp, u0mesh, kx=1, ky=1, s=0)
        # Compute the value at the given gmag and color:
        u0 = u0spline.ev(g_mag,bp_rp)
    else:
        print("Please call function as RUWE(g_mag = value, bp_rp = value)")
    return u0

def RUWE_u0_nocolor(g_mag):
    import numpy as np
    from scipy.interpolate import interp1d, RectBivariateSpline
    # If you only have gmag data use this table:
    u0data = np.genfromtxt('/Users/loganpearce/anaconda3/lib/python3.7/site-packages/myastrotools/table_u0_g.txt', names=['g_mag', 'u0'], skip_header=1, delimiter=',')
    # Create the interpolation function:
    u = interp1d(u0data['g_mag'], u0data['u0'], bounds_error=True)
    # Compute at the given gmag:
    u0 = u(g_mag)
    return u0

def get_RUWE(source_ids, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia DR2 source id to return the re-normalised unit weight error for the solution'''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    from myastrotools.gaia_tools import RUWE_u0,RUWE_u0_nocolor
    # If a list of source ids:
    try:
        for source_id in source_ids:
            job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id))
            j = job.get_results()
            if np.isnan(j['bp_rp']):
                u0 = RUWE_u0_nocolor(np.array(j['phot_g_mean_mag']))
            else:
                u0 = RUWE_u0(g_mag = j['phot_g_mean_mag'],bp_rp = j['bp_rp'])
                u = np.sqrt(np.array(j['astrometric_chi2_al'])/(np.array(j['astrometric_n_good_obs_al'])-5))
            RUWE = u/u0
            print('For',source_id,'RUWE=',RUWE[0])
    except:
        # If a sinlge source id:
        job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_ids))
        j = job.get_results()
        if np.isnan(j['bp_rp']):
            u0 = RUWE_u0_nocolor(np.array(j['phot_g_mean_mag']))
        else:
            u0 = RUWE_u0(g_mag = j['phot_g_mean_mag'],bp_rp = j['bp_rp'])
            u = np.sqrt(np.array(j['astrometric_chi2_al'])/(np.array(j['astrometric_n_good_obs_al'])-5))
        RUWE = u/u0
        print('For',source_ids,'RUWE=',RUWE[0])

def get_RUWE(source_ids, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia DR2 source id to return the re-normalised unit weight error for the solution'''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    from myastrotools.gaia_tools import RUWE_u0,RUWE_u0_nocolor
    # If a list of source ids:
    try:
        for source_id in source_ids:
            job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id))
            j = job.get_results()
            if np.isnan(j['bp_rp']):
                u0 = RUWE_u0_nocolor(np.array(j['phot_g_mean_mag']))
            else:
                u0 = RUWE_u0(g_mag = j['phot_g_mean_mag'],bp_rp = j['bp_rp'])
                u = np.sqrt(np.array(j['astrometric_chi2_al'])/(np.array(j['astrometric_n_good_obs_al'])-5))
            RUWE = u/u0
            print('For',source_id,'RUWE=',RUWE[0])
    except:
        # If a sinlge source id:
        job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_ids))
        j = job.get_results()
        if np.isnan(j['bp_rp']):
            u0 = RUWE_u0_nocolor(np.array(j['phot_g_mean_mag']))
        else:
            u0 = RUWE_u0(g_mag = j['phot_g_mean_mag'],bp_rp = j['bp_rp'])
            u = np.sqrt(np.array(j['astrometric_chi2_al'])/(np.array(j['astrometric_n_good_obs_al'])-5))
        RUWE = u/u0
        print('For',source_ids,'RUWE=',RUWE[0])

def get_distance(source_ids, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia DR2 source id to return the distance and error in parsecs'''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    from myastrotools.tools import distance
    try:
        d,e = np.array([]),np.array([])
        for source_id in source_ids:
            job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id))
            j = job.get_results()
            di,ei = distance(np.array(j['parallax']),np.array(j['parallax_error']))
            d = np.append(d,di)
            e = np.append(e,ei)
            print('For',source_id,'d=',[di,ei])
    except:
        job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_ids))
        j = job.get_results()
        d,e = distance(np.array(j['parallax']),np.array(j['parallax_error']))
    return d,e

def get_gmag(source_ids, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia source id to return the Gmag'''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    from myastrotools.tools import distance
    try:
        gmag = np.array([])
        for source_id in source_ids:
            job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id))
            j = job.get_results()
            gmag = np.append(gmag, j['phot_g_mean_mag'][0])
            print('For',source_id,'gmag=',j['phot_g_mean_mag'][0])
    except:
        job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_ids))
        j = job.get_results()
        gmag = j['phot_g_mean_mag'][0]
    return gmag

def get_separation(source_id1,source_id2, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia DR2 source ids to return the separation between two sources in arcsec'''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    from myastrotools.gaia_tools import to_polar
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000.
    
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id1))
    j = job.get_results()

    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id2))
    k = job.get_results()

    RAa, RAaerr = j[0]['ra'], j[0]['ra_error']*mas_to_deg
    RAb, RAberr = k[0]['ra'], k[0]['ra_error']*mas_to_deg
    Deca, Decaerr = j[0]['dec'], j[0]['dec_error']*mas_to_deg
    Decb, Decberr = k[0]['dec'], k[0]['dec_error']*mas_to_deg

    raa_array = np.random.normal(RAa, RAaerr, 10000)
    rab_array = np.random.normal(RAb, RAberr, 10000)
    deca_array = np.random.normal(Deca, Decaerr, 10000)
    decb_array = np.random.normal(Decb, Decberr, 10000)
    rho_array,pa_array = to_polar(raa_array,rab_array,deca_array,decb_array)
    rho, rhoerr  = np.mean(rho_array).value, np.std(rho_array).value
    return rho/1000,rhoerr/1000

def get_positionangle(source_id1,source_id2, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia DR2 source ids to return the position angle between two sources in degrees'''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    from myastrotools.gaia_tools import to_polar
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000.
    
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id1))
    j = job.get_results()

    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id2))
    k = job.get_results()

    RAa, RAaerr = j[0]['ra'], j[0]['ra_error']*mas_to_deg
    RAb, RAberr = k[0]['ra'], k[0]['ra_error']*mas_to_deg
    Deca, Decaerr = j[0]['dec'], j[0]['dec_error']*mas_to_deg
    Decb, Decberr = k[0]['dec'], k[0]['dec_error']*mas_to_deg

    raa_array = np.random.normal(RAa, RAaerr, 10000)
    rab_array = np.random.normal(RAb, RAberr, 10000)
    deca_array = np.random.normal(Deca, Decaerr, 10000)
    decb_array = np.random.normal(Decb, Decberr, 10000)
    rho_array,pa_array = to_polar(raa_array,rab_array,deca_array,decb_array)
    pa,paerr = np.mean(pa_array).value,np.std(pa_array).value
    return pa,paerr

def get_seppa(source_id1,source_id2, catalog = 'gaiaedr3.gaia_source'):
    '''Use Gaia DR2 source ids to return separation in arcsec and 
    the position angle between two sources in degrees
    '''
    import warnings
    warnings.filterwarnings("ignore")
    import numpy as np
    from astroquery.gaia import Gaia
    from myastrotools.tools import to_polar
    deg_to_mas = 3600000.
    mas_to_deg = 1./3600000.
    
    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id1))
    j = job.get_results()

    job = Gaia.launch_job("SELECT * FROM "+catalog+" WHERE source_id = "+str(source_id2))
    k = job.get_results()

    RAa, RAaerr = j[0]['ra'], j[0]['ra_error']*mas_to_deg
    RAb, RAberr = k[0]['ra'], k[0]['ra_error']*mas_to_deg
    Deca, Decaerr = j[0]['dec'], j[0]['dec_error']*mas_to_deg
    Decb, Decberr = k[0]['dec'], k[0]['dec_error']*mas_to_deg

    raa_array = np.random.normal(RAa, RAaerr, 10000)
    rab_array = np.random.normal(RAb, RAberr, 10000)
    deca_array = np.random.normal(Deca, Decaerr, 10000)
    decb_array = np.random.normal(Decb, Decberr, 10000)
    rho_array,pa_array = to_polar(raa_array,rab_array,deca_array,decb_array)
    pa,paerr = np.mean(pa_array).value,np.std(pa_array).value
    rho, rhoerr  = np.mean(rho_array).value, np.std(rho_array).value
    return rho/1000,rhoerr/1000, pa, paerr

def edr3ToICRF(pmra,pmdec,ra,dec,G):
    ''' Corrects for biases in proper motion. The function is from https://arxiv.org/pdf/2103.07432.pdf

    Args:
        pmra,pmdec (float): proper motion
        ra, dec (float): right ascension and declination
        G (float): G magnitude

    Written by Sam Christian, 2021
    '''
    if G>=13:
        return pmra , pmdec
    import numpy as np
    def sind(x):
        return np.sin(np.radians(x))
    def cosd(x):
        return np.cos(np.radians(x))
    table1="""
    0.0 9.0 9.0 9.5 9.5 10.0 10.0 10.5 10.5 11.0 11.0 11.5 11.5 11.75 11.75 12.0 12.0 12.25 12.25 12.5 12.5 12.75 12.75 13.0
    18.4 33.8 -11.3 14.0 30.7 -19.4 12.8 31.4 -11.8 13.6 35.7 -10.5 16.2 50.0 2.1 19.4 59.9 0.2 21.8 64.2 1.0 17.7 65.6 -1.9 21.3 74.8 2.1 25.7 73.6 1.0 27.3 76.6 0.5
    34.9 68.9 -2.9 """
    table1 = np.fromstring(table1,sep=" ").reshape((12,5)).T
    Gmin = table1[0]
    Gmax = table1[1]
    #pick the appropriate omegaXYZ for the source’s magnitude:
    omegaX = table1[2][(Gmin<=G)&(Gmax>G)][0]
    omegaY = table1[3][(Gmin<=G)&(Gmax>G)][0] 
    omegaZ = table1[4][(Gmin<=G)&(Gmax>G)][0]
    pmraCorr = -1*sind(dec)*cosd(ra)*omegaX -sind(dec)*sind(ra)*omegaY + cosd(dec)*omegaZ
    pmdecCorr = sind(ra)*omegaX -cosd(ra)*omegaY
    return pmra-pmraCorr/1000., pmdec-pmdecCorr/1000.

def get_BJ2021_distance(sourceid, get_rpgeo = True):
    from astroquery.vizier import Vizier
    if get_rpgeo:
        s = Vizier(catalog="I/352/gedr3dis",
             columns=['rgeo', 'rpgeo','b_rpgeo','B_rpgeo']).query_constraints(Source=sourceid)[0]
        return s['rpgeo'][0], s['b_rpgeo'][0], s['B_rpgeo'][0]
    else:
        s = Vizier(catalog="I/352/gedr3dis",
             columns=['rgeo', 'b_rgeo','B_rgeo']).query_constraints(Source=sourceid)[0]
        return s['rgeo'][0], s['b_rgeo'][0], s['B_rgeo'][0]


def GetPointsWithinARegion(xdata, ydata, points):
    ''' For a region defined by points, return the indicies of items from [xdata,ydata]
    that lie within that region
    
    Args:
        xdata, ydata (arr): x and y data 
        points (arr): array of points describing region in tuples of (x,y)
        
    Returns:
        indicies of points in dataframe that lie within the region.
    '''
    y = points[:,1]
    x = points[:,0]

    # find points that lie within region:
    stacked1 = np.stack((xdata,ydata),axis=1)
    from matplotlib import path
    p = path.Path(points)
    indicieswithinregion = p.contains_points(stacked1)
    return indicieswithinregion


############################################################################################################
#################################### PMa ####################################################
############################################################################################################

def gamma(Pbar):
    ''' Return the observing window smearing function gamma given Pbar where
    Pbar = Period / dt; dt = Gaia or Hip observing window.
    dt = 668 d (Gaia DR2); 1227 d (Hip2)
    '''
    import numpy as np
    
    return (Pbar/(np.sqrt(2)*np.pi)) * np.sqrt(1 - np.cos(2*np.pi / Pbar))
    
# Written by Kervella:
def zetafunc(P):
    from astropy.io import ascii
    zetval = ascii.read('../zeta-values.csv')
    zet = np.interp(np.array(P), np.array(zetval['P/T']), np.array(zetval['zeta']))
    zet_errplus = np.interp(np.array(P), np.array(zetval['P/T']), np.array(zetval['zeta_errplus']))
    zet_errminus = np.interp(np.array(P), np.array(zetval['P/T']), np.array(zetval['zeta_errminus']))
    return np.array(zet), np.array(zet_errplus), np.array(zet_errminus)

def mBr_function(starmass, dvel_norm, dvel_norm_err, r,\
                 dtGaia=(1037.93*u.day),dtHG=((2016.0-1991.25)*u.year)):
    # dtHG = 2016.0 - 1991.25
    # dtGaia = (Time('2017-05-28T08:44:00',format='isot')-Time('2014-07-25T10:30:00', format='isot'))
    #        = 1037.93 days
    from myastrotools.tools import zetafunc, gamma
    Pr = np.sqrt(r**3 * (4*np.pi**2)/(c.G*starmass))
    barP = (Pr.to(u.year).value)/(dtGaia.to(u.year).value)
    zetval, zetval_errplus, zetval_errminus = zetafunc(Pr.to(u.year).value/dtHG)
    mBr = (np.sqrt(r/u.au) / gamma(barP) / zetval\
        * dvel_norm/0.87\
        * np.sqrt(1*u.au * starmass/c.G)).to(u.Mjup)
    rel_Verr = dvel_norm_err / dvel_norm

    mBrmin = mBr * (1-np.sqrt((.12/.87)**2 + rel_Verr**2 + (zetval_errminus/zetval)**2))
    mBrmax = mBr * (1+np.sqrt((.32/.87)**2 + rel_Verr**2 + (zetval_errplus/zetval)**2))

    return r, mBr, mBrmin, mBrmax


def MakeKervellaPMaPlot(HIP, distance):
    from myastrotools.tools import mBr_function
    radius = 1
    result = v.query_object("HIP "+str(HIP), radius=radius*u.arcsec)
    dVt = result[0]['dVt']
    m1 = result[0]['M1']

    minAU = 0.5
    maxAU = 200
    nval = 500
    r = np.geomspace(minAU,maxAU,nval)*u.au

    r, mBr, mBrmin, mBrmax = mBr_function(m1, dVt, 3.06, r,
                                          dtGaia=(1037.93*u.day), dtHG=((2016.0-1991.25)*u.year))
    r_as = r.value/distance
    
    fig, ax1 = plt.subplots()
    ax1.plot(r_as,mBr)
    ax1.fill_between(r_as,mBrmin.to(u.Mjup).value,mBrmax.to(u.Mjup).value,
                    alpha = 0.2)

    ax2 = ax1.twiny()
    ax2.plot(r,mBr, alpha=0)

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_xscale('log')

    ax1.grid(ls=':')
    ax1.set_xlabel(r'Separation (")')
    ax2.set_xlabel(r'Orbital Radius (AU)')
    ax1.set_ylabel(r'm$_2$ (M$_{\mathrm{Jup}}$)')
    #ax1.set_ylim(top=1e3)
    plt.tight_layout()
    
    return fig, mBr

############################################################################################################
########################################### Orbit Tools ####################################################
############################################################################################################

def hyperbolic_anomaly(H,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return H - (e*np.sin(H)) - M

def hyperbolic_solve(f, M0, e, h):
    ''' Newton-Raphson solver for hyperbolic anomaly
    from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2019
    '''
    import numpy as np
    H0 = M0
    lastH = H0
    nextH = lastH + 10* h 
    number=0
    while (abs(lastH - nextH) > h) and number < 1001: 
        new = f(nextH,e,M0) 
        lastH = nextH
        nextH = lastH - new / (1.-e*np.cos(lastH)) 
        number=number+1
        if number >= 100:
            nextH = float('NaN')
    return nextH

def hyperbolic_sma(vinf,M):
    """ Given hyperbolic excess velocity, compute semi-major axis for
        hyperbolic orbit around object of mass M.  vinf and M must
        be astropy unit objects
    """
    import astropy.units as u
    import astropy.constants as c
    import numpy as np
    vinf = vinf.to(u.m/u.s)
    mu = c.G*(M.to(u.kg))
    return -mu/(vinf**2)

def hyperbolic_ecc(vinf,M,R):
    """ Given hyperbolic excess velocity, compute eccentricity for
        hyperbolic orbit around object of mass M to periastron distance R.  
        vinf, R and M must be astropy unit objects
    """
    import astropy.units as u
    import astropy.constants as c
    import numpy as np
    a = sma(vinf,M)
    e = 1 - R/a
    return e

def hyperbolic_compute_psi(vinf,M,R):
    """ Given hyperbolic excess velocity, compute maximum deflection angle for
        hyperbolic orbit around object of mass M to periastron distance R.  
        vinf, R and M must be astropy unit objects.
    """
    import astropy.units as u
    import astropy.constants as c
    import numpy as np
    e = ecc(vinf,M,R)
    psi = 2*np.arcsin(1/e)
    return psi

def impact_parameter(vinf,M,R):
    """ Given hyperbolic excess velocity, compute impact parameter required for
        hyperbolic orbit around object of mass M to periastron distance R.  
        vinf, R and M must be astropy unit objects.
    """
    import astropy.units as u
    import astropy.constants as c
    import numpy as np
    a = sma(vinf,M)
    e = ecc(vinf,M,R)
    return -a*np.sqrt(e**2-1)

def ComputeChi2(array, measurements):
    chi = 0
    for i in range(len(array)):
        chi += ( (array[i][0] - measurements[i]) / array[i][1] ) ** 2
    return chi

def t0_to_tau(t0, period, ref_epoch = 2015.5):
    """
    Convert epoch of periastron passage (t0) to orbit fraction (tau)
    Args:
        t0 (float or np.array): value to t0 to convert, decimal years
        ref_epoch (float or np.array): reference epoch that tau is defined from. Default 2015.5 for DR2
        period (float or np.array): period (in years) that tau is defined by
    Returns:
        tau (float or np.array): corresponding taus
    """
    tau = (ref_epoch - t0)/period
    tau %= 1

    return tau

def tau_to_t0(tau, period, ref_epoch = 2015.5, after_date = None):
    """
    Convert tau (epoch of periastron in fractional orbital period after ref epoch) to
    T0 in decimal years.  Will return as the first periastorn passage before the reference date of 2015.5.
    Args:
        tau (float or np.array): value of tau to convert
        ref_epoch (float or np.array): date that tau is defined relative to.  2015.5 for Gaia DR2
        period (float or np.array): period (in years) that tau is noralized with
        after_date (float): T0 will be the first periastron after this date. If None, use ref_epoch.
    Returns:
        t0 (float or np.array): corresponding T0 of the taus in decimal years.
    """

    t0 = ref_epoch - (tau * period)

    if after_date is not None:
        num_periods = (after_date - t0)/period
        num_periods = int(np.ceil(num_periods))
        
        t0 += num_periods * period

    return t0

def period(sma,mass):
    """ Given semi-major axis in AU and mass in solar masses, return the period in years of an orbit using 
        Kepler's third law.
        Written by Logan Pearce, 2019
    """
    import numpy as np
    import astropy.units as u
    # If astropy units are given, return astropy unit object
    try:
        sma = sma.to(u.au)
        mass = mass.to(u.Msun)
        period = np.sqrt(((sma)**3)/mass).value*(u.yr)
    # else return just a value.
    except:
        period = np.sqrt(((sma)**3)/mass)
    return period

def semimajoraxis(period,mass):
    """ Given period in years and mass in solar masses, return the semi-major axis in au of an orbit using 
        Kepler's third law.
        Written by Logan Pearce, 2019
    """
    import numpy as np
    import astropy.units as u
    # If astropy units are given, return astropy unit object
    try:
        period = period.to(u.yr)
        mass = mass.to(u.Msun)
        sma = ((mass * period**2) ** (1/3)).value*u.au
    # else return just a value.
    except:
        sma = (mass * period**2) ** (1/3)
    return sma

def distance(parallax,parallax_error):
    '''Computes distance from Gaia parallaxes using the Bayesian method of Bailer-Jones 2015.
    Input: parallax [mas], parallax error [mas]
    Output: distance [pc], 1-sigma uncertainty in distance [pc]
    '''
    import numpy as np
    # Compute most probable distance:
    L=1350 #parsecs
    # Convert to arcsec:
    parallax, parallax_error = parallax/1000., parallax_error/1000.
    # establish the coefficients of the mode-finding polynomial:
    coeff = np.array([(1./L),(-2),((parallax)/((parallax_error)**2)),-(1./((parallax_error)**2))])
    # use numpy to find the roots:
    g = np.roots(coeff)
    # Find the number of real roots:
    reals = np.isreal(g)
    realsum = np.sum(reals)
    # If there is one real root, that root is the  mode:
    if realsum == 1:
        gd = np.real(g[np.where(reals)[0]])
    # If all roots are real:
    elif realsum == 3:
        if parallax >= 0:
            # Take the smallest root:
            gd = np.min(g)
        elif parallax < 0:
            # Take the positive root (there should be only one):
            gd = g[np.where(g>0)[0]]
    
    # Compute error on distance from FWHM of probability distribution:
    from scipy.optimize import brentq
    rmax = 1e6
    rmode = gd[0]
    M = (rmode**2*np.exp(-rmode/L)/parallax_error)*np.exp((-1./(2*(parallax_error)**2))*(parallax-(1./rmode))**2)
    lo = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), 0.001, rmode)
    hi = brentq(lambda x: 2*np.log(x)-(x/L)-(((parallax-(1./x))**2)/(2*parallax_error**2)) \
               +np.log(2)-np.log(M)-np.log(parallax_error), rmode, rmax)
    fwhm = hi-lo
    # Compute 1-sigma from FWHM:
    sigma = fwhm/2.355
            
    return gd[0],sigma

def to_polar(RAa,RAb,Deca,Decb):
    ''' Converts RA/Dec [deg] of two binary components into separation and position angle of B relative 
        to A [mas, deg]
    '''
    import numpy as np
    import astropy.units as u
    dRA = (RAb - RAa) * np.cos(np.radians(np.mean([Deca,Decb])))
    dRA = (dRA*u.deg).to(u.mas)
    dDec = (Decb - Deca)
    dDec = (dDec*u.deg).to(u.mas)
    r = np.sqrt( (dRA ** 2) + (dDec ** 2) )
    p = (np.degrees( np.arctan2(dDec.value,-dRA.value) ) + 270.) % 360.
    p = p*u.deg
    return r, p

def as_to_km2(arcsec,dist):
    '''
    Convert from as -> km using the distance rather than parallax.  Does not
    compute errors on separation.

    Args:
        arcsec (array, flt): sep in arcsec
        dist (array, flt): distance in parsecs
    Returns:
        array : separation in km

    Written by Logan Pearce, 2019
    '''
    # convert to arcsec, and multiply by distance in pc:
    km = (arcsec)*dist # AU
    km = km * 149598073 # km/AU

    return km

def circular_velocity(au,m):
    """ Given separation in AU and total system mass, return the velocity of a test particle on a circular orbit
        around a central body at that mass """
    import astropy.constants as c
    import astropy.units as u
    import numpy as np
    m = m*u.Msun
    au = au*u.AU
    v = np.sqrt( c.G * m.to(u.kg) / (au.to(u.m)) )
    return v.to(u.km/u.s)

def masyr_to_kms(mas_yr,plx):
    '''
    Convert from mas/yr -> km/s
     
    Args:
        mas_yr (array): velocity in mas/yr
        plx (tuple,float): parallax, tuple of (plx,plx error)
    Returns:
        array : velocity in km/s
    
    Written by Logan Pearce, 2019
    '''
    from myastrotools.tools import distance
    d = distance(*plx)
    # convert mas to km:
    km_s = ((mas_yr*u.mas.to(u.arcsec)*d[0])*u.AU).to(u.km)
    # convert yr to s:
    km_s = (km_s.value)*(u.km/u.yr).to(u.km/u.s)
    
    return km_s

def turn_gaia_into_physical(ra, dec, plx, plx_error, pmra, pmdec):
    ''' Take Gaia archive units (deg, mas, mas/yr) and turn them into physical units (pc, AU, km/s)
    Args:
        ra (flt): RA in deg
        dec (flt): DEC in deg
        plx (tuple) [parallax, parallax_error] in mas
        pmra (flt): proper motion in RA in mas/yr
        pmdec (flt): proper motion in DEC in mas/yr
        
    Returns:
        flt: RA in AU
        flt: DEC in AU
        flt: distance in pc
        flt: pmra in km/s
        flt: pmdec in km/s
    '''
    import astropy.units as u
    from orbittools.orbittools import distance, masyr_to_kms
    # Get distance in pc:
    dist = distance(np.array([plx,plx_error]))[0]
    # Convert ra/dec from degrees to arcsec
    ra = (ra*u.deg).to(u.arcsec)
    dec = (dec*u.deg).to(u.arcsec)
    # theta * d = a:
    ra_au = ((ra * dist).value)*u.au
    dec_au = ((dec * dist).value)*u.au
    # proper motions:
    pmra_kms = masyr_to_kms(pmra,plx_tuple)    # km/s
    pmdec_kms = masyr_to_kms(pmdec,plx_tuple)
    
    return ra_au, dec_au, dist, pmra_kms, pmdec_kms

def v_escape(r,M):
    ''' Compute the escape velocity of an object of mass M at distance r.  M and r should be
        astropy unit objects
    '''
    try:
        r = r.to(u.au)
        M = M.to(u.Msun)
    except:
        r = r*u.au
        M = M*u.Msun
    return (np.sqrt(2*c.G*(M) / (r))).decompose()
    
def parallax_to_circularvel(plx,mass,theta):
    """ Given a parallax value+error, total system mass, and separation, compute the circular velocity
        for a test particle on a circular orbit at that separation.  Plx should be a tuple of
        (plx value, error) in mas.  Theta should be an astropy units object, either arcsec or mas, mass
        in solar masses.  Returns circular vel in km/s
    """
    from myastrotools.astrometry import circular_velocity, physical_separation
    from myastrotools.gaia_tools import distance
    import astropy.units as u
    dist = distance(*plx)[0]*u.pc
    sep = physical_separation(dist,theta)
    cv = circular_velocity(sep.value,mass)
    return cv

def tisserand(a,e,i):
    """ Return Tisserand parameter for given orbital parameters relative to
           Jupiter
        Inputs:
            a : flt
                semi-major axis in units of Jupiter sma
            e : flt
                eccentricity
            i : flt
                inclination in radians relative to J-S orbit plane
        Returns:
            T : flt
                Tisserand parameter define wrt Jupiter's orbit
    """
    return 1/(a) + 2*np.sqrt(a*(1-e**2))*np.cos(i)

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

def M(T,T0,obsdate = 2016):
    """ Given period, ref date, and observation date,
            return the mean anomaly

       Parameters:
       -----------
       T : flt
           period
       T0 : flt
           time of periastron passage
       obsdate : flt
           observation date.  Default = 2015.5 (Gai DR2 ref date)
       Returns:
       -----------
       mean anomaly [radians]
    """
    return (2*np.pi/T)*(obsdate-T0)

def NielsenPrior(Nsamples):
    e = np.linspace(1e-3,0.95,Nsamples)
    P = (2.1 - (2.2*e))
    P = P/np.sum(P)
    ecc = np.random.choice(e, size = Nsamples, p = P)
    return ecc

def draw_orbits(number, EccNielsenPrior = False, draw_lon = False):
    ''' Semi-major axis is fixed at 100 au and long. of asc. node is fixed at 0 deg.
    Written by Logan Pearce, 2019
    '''
    import astropy.units as u
    import numpy as np
    sma = 100.*u.au
    sma = np.array(np.linspace(sma,sma,number))
    # Eccentricity:
    if EccNielsenPrior:
        from myastrotools.tools import NielsenPrior
        ecc = NielsenPrior(number)
    else:
        ecc = np.random.uniform(0.0,1.0,number)
    # Inclination in radians:
    cosi = np.random.uniform(-1.0,1.0,number)  #Draws sin(i) from a uniform distribution.  Inclination
    # is computed as the arccos of cos(i):
    inc = np.degrees(np.arccos(cosi))
    # Argument of periastron in degrees:
    argp = np.random.uniform(0.0,360.0,number)
    # Long of nodes:
    if draw_lon:
        lon = np.random.uniform(0.0,360.0,number)
    else:
        lon = np.degrees(0.0)
        lon = np.array([lon]*number)
    # orbit fraction (fraction of orbit completed at observation date since reference date)
    orbit_fraction = np.random.uniform(0.0,1.0,number)
    return sma, ecc, inc, argp, lon, orbit_fraction

def eccentricity_anomaly(E,e,M):
    '''Eccentric anomaly function'''
    import numpy as np
    return E - (e*np.sin(E)) - M

def solve(f, M0, e, h):
    ''' Newton-Raphson solver for eccentricity anomaly
    from https://stackoverflow.com/questions/20659456/python-implementing-a-numerical-equation-solver-newton-raphson
    Inputs: 
        f (function): function to solve (transcendental ecc. anomaly function)
        M0 (float): mean anomaly
        e (float): eccentricity
        h (float): termination criteria for solver
    Returns: nextE (float): converged solution for eccentric anomaly
        Written by Logan Pearce, 2019
    '''
    import numpy as np
    from myastrotools.tools import eccentricity_anomaly
    if M0 / (1.-e) - np.sqrt( ( (6.*(1-e)) / e ) ) <= 0:
        E0 = M0 / (1.-e)
    else:
        E0 = (6. * M0 / e) ** (1./3.)
    lastE = E0
    nextE = lastE + 10* h 
    number=0
    while (abs(lastE - nextE) > h) and number < 1001: 
        new = f(nextE,e,M0) 
        lastE = nextE
        nextE = lastE - new / (1.-e*np.cos(lastE)) 
        number=number+1
        if number >= 1000:
            nextE = float('NaN')
    return nextE

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

###################################################################
# a set of functions for working with position, velocity, acceleration,
# and orbital elements for Keplerian orbits.

def calc_XYZ(a,T,to,e,i,w,O,date, solvefunc = solve):
    ''' Compute projected on-sky position only of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point. 
        Inputs:
            a [as]: semi-major axis
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
        Returns: X, Y, and Z coordinates [as] where +X is in the reference direction (north) and +Y is east, and +Z
            is towards observer
    '''
    import numpy as np
    from lofti_gaia.loftitools import solve
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solvefunc(eccentricity_anomaly, M,e, 0.001)
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    # orbit plane radius in as:
    r = (a*(1.-e**2))/(1.+(e*cos(f)))
    X = r * ( cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i) )
    Y = r * ( sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i) )
    Z = r * sin(w+f)*sin(i)
    return X,Y,Z

def calc_velocities(a,T,to,e,i,w,O,date,dist, solvefunc = solve):
    ''' Compute 3-d velocity of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  Uses my eqns derived from Seager 
        Exoplanets Ch2.
        Inputs:
            a [as]: semi-major axis
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            m_tot [Msol]: total system mass
        Returns: X dot, Y dot, Z dot three dimensional velocities [km/s]
    '''
    import numpy as np
    import astropy.units as u
    from orbittools.orbittools import as_to_km2, solve
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    
    # convert to km:
    a_km = as_to_km2(a,dist)
    #a_km = a_km[0]
    
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solvefunc(eccentricity_anomaly, M,e, 0.001)
    #E = solve(eccentricity_anomaly, M,e, 0.001)
    r1 = a*(1.-e*cos(E))
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    
    # Compute velocities:
    rdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * e*sin(f)
    rfdot = ( (n*a_km) / (np.sqrt(1-e**2)) ) * (1 + e*cos(f))
    Xdot = rdot * (cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
           rfdot * (-cos(O)*sin(w+f) - sin(O)*cos(w+f)*cos(i))
    Ydot = rdot * (sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
           rfdot * (-sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zdot = ((n*a_km) / (np.sqrt(1-e**2))) * sin(i) * (cos(w+f) + e*cos(w))
    
    Xdot = Xdot*(u.km/u.yr).to((u.km/u.s))
    Ydot = Ydot*(u.km/u.yr).to((u.km/u.s))
    Zdot = Zdot*(u.km/u.yr).to((u.km/u.s))
    return Xdot,Ydot,Zdot

def calc_accel(a,T,to,e,i,w,O,date,dist, solvefunc = solve):
    ''' Compute 3-d acceleration of a single object on a Keplerian orbit given a 
        set of orbital elements at a single observation point.  
        Inputs:
            a [as]: semi-major axis in as
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
        Returns: X ddot, Y ddot, Z ddot three dimensional accelerations [m/s/yr]
    '''
    import numpy as np
    from numpy import tan, arctan, sqrt, cos, sin, arccos
    import astropy.units as u
    from lofti_gaia.loftitools import to_si, solve
    # convert to km:
    a_mas = a*u.arcsec.to(u.mas)
    try:
        a_mas = a_mas.value
    except:
        pass
    a_km = to_si(a_mas,0.,dist)[0]
    # Compute true anomaly:
    n = (2*np.pi)/T
    M = n*(date-to)
    try:
        nextE = [solvefunc(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(M,e)]
        E = np.array(nextE)
    except:
        E = solvefunc(eccentricity_anomaly, M,e, 0.001)
    # r and f:
    f1 = sqrt(1.+e)*sin(E/2.)
    f2 = sqrt(1.-e)*cos(E/2.)
    f = 2.*np.arctan2(f1,f2)
    r = (a_km*(1-e**2))/(1+e*cos(f))
    # Time derivatives of r, f, and E:
    Edot = n/(1-e*cos(E))
    rdot = e*sin(f)*((n*a_km)/(sqrt(1-e**2)))
    fdot = ((n*(1+e*cos(f)))/(1-e**2))*((sin(f))/sin(E))
    # Second time derivatives:
    Eddot = ((-n*e*sin(f))/(1-e**2))*fdot
    rddot = a_km*e*cos(E)*(Edot**2) + a_km*e*sin(E)*Eddot
    fddot = Eddot*(sin(f)/sin(E)) - (Edot**2)*(e*sin(f)/(1-e*cos(E)))
    # Positional accelerations:
    Xddot = (rddot - r*fdot**2)*(cos(O)*cos(w+f) - sin(O)*sin(w+f)*cos(i)) + \
            (-2*rdot*fdot - r*fddot)*(cos(O)*sin(w+f) + sin(O)*cos(w+f)*cos(i))
    Yddot = (rddot - r*fdot**2)*(sin(O)*cos(w+f) + cos(O)*sin(w+f)*cos(i)) + \
            (2*rdot*fdot + r*fddot)*(sin(O)*sin(w+f) + cos(O)*cos(w+f)*cos(i))
    Zddot = sin(i)*((rddot - r*(fdot**2))*sin(w+f) + ((2*rdot*fdot + r*fddot)*cos(w+f)))
    return Xddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), Yddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr)), \
                    Zddot*(u.km/u.yr/u.yr).to((u.m/u.s/u.yr))


def ThieleInnesToCampbell(A,B,F,G,Aerr = None, Berr = None, Ferr = None, Gerr = None):
    ''' Convert TI elements to Campbell orbital elements
    '''
    uu = (A**2 + B**2 + F**2 + G**2) / 2
    v = A*G - B*F
    sma = np.sqrt( uu + np.sqrt((uu+v)*(uu-v)) )

    wplusO = np.arctan2( (B-F),(A+G) ) % 2*np.pi
    wminusO = np.arctan2( (B+F),(G-A) ) % 2*np.pi
    #assert np.sign(np.sin(wplusO)) == np.sign((B-F))
    #assert np.sign(np.sin(wminusO)) == np.sign((-B-F))

    argp = np.degrees( ((wplusO + wminusO) / 2) )
    lan = np.degrees( ((wplusO - wminusO) / 2) )

    d1 = np.abs( (A+G)*np.cos(wminusO) )
    d2 = np.abs( (F-B)*np.sin(wminusO) )
    if d1 >= d2:
        inc = 2*np.arctan( np.sqrt( np.abs((A-G)*np.cos(wplusO))/d1 ) )
    else:
        inc = 2*np.arctan( np.sqrt( np.abs((B+F)*np.sin(wplusO))/d2 ) )
    
    if Aerr:
        # uncertainties
        tA = A + (A*u - G*v)/np.sqrt(uu**2 - v**2)
        tB = B + (B*u + F*v)/np.sqrt(uu**2 - v**2)
        tF = F + (F*u + B*v)/np.sqrt(uu**2 - v**2)
        tG = G + (G*u - A*v)/np.sqrt(uu**2 - v**2)
    
        smaerr = (2*sma)**(-1) * ((tA*Aerr)**2 + (tB*Berr)**2 + (tF*Ferr)**2 + (tG*Gerr)**2 
                                  
                                )
        
    return sma, np.degrees(inc), argp, lan


def GetGaiaOrbitalElementsForNSS(GaiaDR3_sourceid):
    ''' For solutions in Gaia DR3 Non-single star catalog (NSS) with solution type "Orbital",
    query the NSS catalog, retrieve Thiele Innes elements, anf convert them to Campbell elements.

    Args:
        GaiaDR3_sourceid (str): Gaia DR3 source id

    Returns:
        dict: Campbell elements
    '''
    from astroquery.vizier import Vizier
    from myastrotools.tools import ThieleInnesToCampbell
    # Gaia NSS catalog number:
    cat = 'I/357'
    result = Vizier.query_object("Gaia DR3 "+GaiaDR3_sourceid, catalog=cat)
    r = result[0]
    A,B,F,G = r['ATI'][0],r['BTI'][0],r['FTI'][0],r['GTI'][0]
    ecc = r['ecc'][0]
    T0 = 2016.0 + (r['Tperi'][0]*u.d.to(u.yr)) # # days since 2016.0
    P = r['Per'][0]*u.d.to(u.yr) # years
    sma, inc, argp, lan = ThieleInnesToCampbell(A,B,F,G,Aerr = None, Berr = None, Ferr = None, Gerr = None)
    elements = {"sma [arcsec]":sma,
                'ecc':ecc,
                'inc [deg]':inc,
                'argp [deg]': argp,
                'lan [deg]': lan,
                'T0 [yr]': T0,
                'P [yr]': P
               }
    return elements

###################################################################
# OFTI specific functions:

def scale_and_rotate(X,Y):
    ''' Generates a new semi-major axis, period, epoch of peri passage, and long of peri for each orbit
        given the X,Y plane of the sky coordinates for the orbit at the date of the reference epoch
    '''
    import numpy as np
    r_model = np.sqrt((X**2)+(Y**2))
    rho_rand = np.random.normal(rho/1000.,rhoerr/1000.) #This generates a gaussian random to 
    #scale to that takes observational uncertainty into account.  #convert to arcsec
    #rho_rand = rho/1000. 
    a2 = a*(rho_rand/r_model)  #<- scaling the semi-major axis
    #New period:
    a2_au=a2*dist #convert to AU for period calc:
    T2 = np.sqrt((np.absolute(a2_au)**3)/np.absolute(m1))
    #New epoch of periastron passage
    to2 = d-(const*T2)

    # Rotate:
    # Rotate PA:
    PA_model = (np.degrees(np.arctan2(X,-Y))+270)%360 #corrects for difference in zero-point
    #between arctan function and ra/dec projection
    PA_rand = np.random.normal(pa,paerr) #Generates a random PA within 1 sigma of observation
    #PA_rand = pa
    #New omega value:
    O2=[]
    for PA_i in PA_model:
        if PA_i < 0:
            O2.append((PA_rand-PA_i) + 360.)
        else:
            O2.append(PA_rand-PA_i)
    # ^ This step corrects for the fact that the arctan gives the angle from the +x axis being zero,
    #while for RA/Dec the zero angle is +y axis.  

    #Recompute model with new rotation:
    O2 = np.array(O2)
    O2 = np.radians(O2)
    return a2,T2,to2,O2

def calc_OFTI(a,T,const,to,e,i,w,O,d,m1,dist,rho,pa, solvefunc = solve):
    '''Perform OFTI steps to determine position/velocity/acceleration predictions given
       orbital elements.
        Inputs:
            a [as]: semi-major axis
            T [yrs]: period
            to [yrs]: epoch of periastron passage (in same time structure as dates)
            e: eccentricity
            i [rad]: inclination
            w [rad]: argument of periastron
            O [rad]: longitude of nodes
            date [yrs]: observation date
            dist [pc]: distance to system in pc
            rho [mas] (tuple, flt): separation and error
            pa [deg] (tuple, flt): position angle and error
        Returns: 
            X, Y, Z positions in plane of the sky [mas],
            X dot, Y dot, Z dot three dimensional velocities [km/s]
            X ddot, Y ddot, Z ddot 3d accelerations in [m/s/yr]
    '''
    import numpy as np
    import astropy.units as u
    
    # Calculate predicted positions at observation date:
    X1,Y1,Z1 = calc_XYZ(a,T,to,e,i,w,O,d)
    # scale and rotate:
    a2,T2,to2,O2 = scale_and_rotate(X1,Y1,rho,pa,a,const,m1,dist,d, solvefunc = solvefunc)
    # recompute predicted position:
    X2,Y2,Z2 = calc_XYZ(a2,T2,to2,e,i,w,O2,d)
    # convert units:
    X2,Y2,Z2 = (X2*u.arcsec).to(u.mas).value, (Y2*u.arcsec).to(u.mas).value, (Z2*u.arcsec).to(u.mas).value
    # Compute velocities at observation date:
    Xdot,Ydot,Zdot = calc_velocities(a2,T2,to2,e,i,w,O2,d,dist, solvefunc = solvefunc)
    # Compute accelerations at observation date:
    Xddot,Yddot,Zddot = calc_accel(a2,T2,to2,e,i,w,O2,d,dist, solvefunc = solvefunc)
    # Convert to degrees:
    i,w,O2 = np.degrees(i),np.degrees(w),np.degrees(O2)
    return X2,Y2,Z2,Xdot,Ydot,Zdot,Xddot,Yddot,Zddot,a2,T2,to2,e,i,w,O2
    

###################################################################
# a different set of functions for working with position, velocity, acceleration,
# and orbital elements for Keplerian orbits.

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

def keplerian_to_cartesian(sma,ecc,inc,argp,lon,meananom,kep, solvefunc = solve):
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
                        +x = -RA, +y = +Dec, +z = towards observer
            vel (3xN arr) [km/s]: velocity in xyz plane.
            acc (3xN arr) [km/s/yr]: acceleration in xyz plane.
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    
    # Compute mean motion and eccentric anomaly:
    meanmotion = np.sqrt(kep / sma**3).to(1/u.s)
    try:
        E = solve(eccentricity_anomaly, meananom, ecc, 0.001)
    except:
        nextE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(meananom, ecc)]
        E = np.array(nextE)

    # Compute position:
    try:
        pos = np.zeros((3,len(sma)))
    # In the plane of the orbit:
        pos[0,:], pos[1,:] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
    except:
        pos = np.zeros(3)
        pos[0], pos[1] = (sma*(np.cos(E) - ecc)).value, (sma*np.sqrt(1-ecc**2)*np.sin(E)).value
        
    # Rotate to plane of the sky:
    pos = rotate_z(pos, np.radians(argp))
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
        futureE = [solve(eccentricity_anomaly, varM,vare, 0.001) for varM,vare in zip(futuremeananom.value, ecc)]
        futureE = np.array(futureE)
    except:
        futureE = solve(eccentricity_anomaly, futuremeananom.value, ecc, 0.001)
    # Compute new velocity at future time:
    futurevel[0], futurevel[1] = (( -meanmotion * sma * np.sin(futureE) ) / ( 1- ecc * np.cos(futureE) )).to(u.km/u.s).value, \
                (( meanmotion * sma * np.sqrt(1 - ecc**2) *np.cos(futureE) ) / ( 1 - ecc * np.cos(futureE) )).to(u.km/u.s).value
    futurevel = rotate_z(futurevel, np.radians(argp))
    futurevel = rotate_x(futurevel, np.radians(inc))
    futurevel = rotate_z(futurevel, np.radians(lon))
    acc = (futurevel-vel)/deltat.value
    
    return np.transpose(pos)*u.au, np.transpose(vel)*(u.km/u.s), np.transpose(acc)*(u.km/u.s/u.yr)

def cartesian_to_keplerian(pos, vel, kep):
    """Given observables XYZ position and velocity, compute orbital elements.  Position must be in
       au and velocity in km/s.  Returns astropy unit objects for all orbital elements.
        Inputs:
            pos (3xN arr) [au]: position in xyz coords in au, with 
                        x = pos[0], y = pos[1], z = pos[2] for each of N orbits
                        +x = +Dec, +y = +RA, +z = towards observer
                        Must be astropy unit array e.g: [1,2,3]*u.AU, ~NOT~ [1*u.AU,2*u.AU,3*u,AU]
            vel (3xN arr) [km/s]: velocity in xyz plane.  Also astropy unit array
            kep (flt) [m^3/s^2] : kepler's constant.  From output of orbittools.keplersconstant. Must be
                        astropy unit object.
        Returns:
            sma (1xN arr flt) [au]: semi-major axis in au, must be an astropy units object
            ecc (1xN arr flt) [unitless]: eccentricity
            inc (1xN arr flt) [deg]: inclination
            argp (1xN arr flt) [deg]: argument of periastron
            lon (1xN arr flt) [deg]: longitude of ascending node
            meananom (1xN arr flt) [radians]: mean anomaly 
        Written by Logan Pearce, 2019, inspired by Sarah Blunt
    """
    import numpy as np
    import astropy.units as u
    # rvector x vvector:
    rcrossv = np.cross(pos, vel)
    # specific angular momentum:
    h = np.sqrt(rcrossv[0]**2 + rcrossv[1]**2 + rcrossv[2]**2)
    # normal vector:
    n = rcrossv / h
    
    # inclination:
    inc = np.arccos(n[2])
    
    # longitude of ascending node:
    lon = np.arctan2(n[0],-n[1])
    
    # semi-major axis:
    r = np.sqrt(pos[0]**2 + pos[1]**2 + pos[2]**2)
    v = np.sqrt(vel[0]**2 + vel[1]**2 + vel[2]**2)
    sma = 1/((2./r) - ((v)**2/kep))
    
    # ecc and f:
    rdotv = pos[0]*vel[0] + pos[1]*vel[1] + pos[2]*vel[2]
    rdot = rdotv/r
    parameter = h**2 / kep
    ecosf = parameter/r - 1
    esinf = (h)*rdot / (kep.to(u.m**3/u.s**2))
    ecc = np.sqrt(ecosf**2 + esinf**2)
    f = np.arctan2(esinf,ecosf)
    f = f.value%(2.*np.pi)
    
    # E and M:
    E = 2. * np.arctan( np.sqrt( (1 - ecc.value)/ (1 + ecc.value) ) * ( np.tan(f/2.) ) )
    M = E - ecc * np.sin(E)
    
    # argument of periastron:
    rcosu = pos[0] * np.cos(lon) + pos[1] * np.sin(lon)
    rsinu = (-pos[0] * np.sin(lon) + pos[1] * np.cos(lon)) / np.cos(inc)
    uangle = np.arctan2(rsinu,rcosu)
    argp = uangle.value - f
    
    return sma.to(u.au), ecc, np.degrees(inc), (np.degrees(argp)%360.)*u.deg, (np.degrees(lon.value)%360.)*u.deg, M

def kepler_advancer(ro, vo, t, k, to = 0):
    ''' Initial value problem solver.  Given an initial position and
        velocity vector (in 2 or 3 dimensions; in any reference frame
        [plane of the sky, plane of the orbit]) at an initial time to,
        compute the position and velocity vector at a later time t in 
        that same frame.

        Written by Logan A. Pearce, 2020
        
        Parameters:
       -----------
       ro : flt, arr
           initial position vector at time = to; astropy unit object of physical not angular distance
       vo : flt, arr
           initial velocity vector at time = to; astropy unit object
       t : flt
           future time at which to compute new r,v vectors; 
           astropy unit object
       k : flt
           "Kepler's constant", k = G*(m1+m2); astropy unit object
       to : flt
           initial time for initial values.  Default = 0; 
           astropy unit object
       
       Returns:
       --------
       new_r : flt, arr
           new position vector at time t in m
       new_v : flt, arr
           new velocity vector at time t in m/s
    '''
    from myastrotools.tools import danby_solve, eccentricity_anomaly
    import numpy as np
    # Convert everything to mks:
    ro = ro.to(u.m).value
    vo = vo.to(u.m/u.s).value
    k = k.to((u.m**3)/(u.s**2)).value
    t = t.to(u.s).value
    if to != 0:
        to = to.to(u.s).value
    # Compute magnitude of position vector:
    r = np.linalg.norm(ro)
    # Compute v^2:
    v2 = np.linalg.norm(vo)**2
    # Compute ang mom h^2:
    h2 = np.linalg.norm(np.cross(ro,vo))**2
    # find a [m] from vis-viva:
    a = (2/r - v2/k)**(-1)
    # mean motion:
    n = np.sqrt(k/(a**3))
    # ecc:
    e = np.sqrt( 1 - h2/(k*a) )
    # Eo:
    E0 = np.arccos(1/e*(1-r/a))
    # M0:
    M0 = E0 - e*np.sin(E0)
    # M(t = t):
    M = M0 + n*(t-to)
    # E:
    E = danby_solve(eccentricity_anomaly, M, e, 1.e-9)
    # delta E
    deltaE = E-E0
    cosDE = np.cos(deltaE)
    sinDE = np.sin(deltaE)
    # f, g:
    f = 1 + (a/r)*(cosDE - 1)
    g = t - to + (1/n)*(sinDE - deltaE)
    # new r:
    new_r = f*ro + g*vo
    # fdot, gdot:
    fprime = 1 - (cosDE * e * np.cos(E)) + (sinDE * e * np.sin(E))
    fdot = -n*a*sinDE / (r*fprime)
    gdot = 1 + (cosDE - 1) / fprime
    # new v:
    new_v = fdot*ro + gdot*vo
    
    return new_r*u.m, new_v*(u.m/u.s)

def kepler_advancer2(ro, vo, t, k, to = 0):
    ''' Initial value problem solver using Wisdom-Holman
        numerically well-defined expressions

        Written by Logan A. Pearce, 2020
        
        Parameters:
       -----------
       ro : flt, arr
           initial position vector at time = to; astropy unit object
       vo : flt, arr
           initial velocity vector at time = to; astropy unit object
       t : flt
           future time at which to compute new r,v vectors; 
           astropy unit object
       k : flt
           "Kepler's constant", k = G*(m1+m2); astropy unit object
       to : flt
           initial time for initial values.  Default = 0; 
           astropy unit object
       
       Returns:
       --------
       new_r : flt, arr
           new position vector at time t in m
       new_v : flt, arr
           new velocity vector at time t in m/s
    '''
    from myastrotools.tools import danby_solve, eccentricity_anomaly
    import numpy as np
    # Convert everything to mks:
    ro = ro.to(u.m).value
    vo = vo.to(u.m/u.s).value
    k = k.to((u.m**3)/(u.s**2)).value
    t = t.to(u.s).value
    if to != 0:
        to = to.to(u.s).value
    # Compute magnitude of position vector:
    r = np.linalg.norm(ro)
    # Compute v^2:
    v2 = np.linalg.norm(vo)**2
    # Compute ang mom h^2:
    h2 = np.linalg.norm(np.cross(ro,vo))**2
    # find a [m] from vis-viva:
    a = (2/r - v2/k)**(-1)
    # mean motion:
    n = np.sqrt(k/(a**3))
    # ecc:
    e = np.sqrt( 1 - h2/(k*a) )
    # Eo:
    E0 = np.arccos(1/e*(1-r/a))
    # M0:
    M0 = E0 - e*np.sin(E0)
    # M(t = t):
    M = M0 + n*(t-to)
    # E:
    E = danby_solve(eccentricity_anomaly, M, e, 1.e-9)
    # Delta E:
    deltaE = E - E0
    # s2:
    s2 = np.sin(deltaE/2)
    sinE = np.sin(E)
    # c2:
    c2 = np.cos(deltaE/2)
    cosE = np.cos(E)
    # s:
    s = 2*s2*c2
    # c:
    c = c2*c2 - s2*s2
    # f prime:
    fprime = 1 - c*e*cosE + s*e*sinE
    # f, g:
    f = 1 - 2*s2*s2*a/r
    g = 2*s2*(s2*e*sinE + c2*r/a)*(1./n)
    # new r:
    new_r = f*ro + g*vo
    # fdot, gdot:
    fdot = -(n*a*s) / (r*fprime)
    gdot = 1 - (2*s2*s2 / fprime)
    # new v:
    new_v = fdot*ro + gdot*vo
    
    return new_r*u.m, new_v*(u.m/u.s)

############################################################################################################
#################################### Orbit Fraction ########################################################
############################################################################################################
################################################################
# a set of functions for estimating how much of an orbit you
# need to observe to get a good handle on the orbit's velocity,
# acceleration, and 3rd derivative.

def orbit_fraction(sep, seperr, snr = 5):
    """ What fraction of an orbital period do you need to observe to characterize
        the velocity, acceleration, and 3rd derivatives to a given SNR?  That
        is, v/sigma_v = 5, etc.  This is a rough estimate derived from assuming 
        a circular face-on orbit.

      Parameters:
       -----------
       sep, seperr : flt
           observed separation and error in separation, any unit
       snr : flt
           desired signal to noise ratio.  Default = 5
        
       Returns:
       --------
        of_for_vel, of_for_acc, of_for_jerk : flt
            orbit fraction required to be observed to achieve desired snr 
            for understanding the velocity, acceleration, and jerk. 
            In decimal fraction.  ex: 0.01 = 1% of orbit
    """
    of_for_vel = (snr/5)*seperr/sep
    of_for_acc = (snr/5)*np.sqrt((9*seperr)/(5*sep))
    of_for_jerk = (snr/5)*(seperr/sep)**(1/3)
    return of_for_vel, of_for_acc, of_for_jerk

def orbit_fraction_observing_time(sep, seperr, period, snr = 5):
    """ Given a fractional postional uncertainty and a given orbital 
        period, what timespace do your observations need to cover to 
        achieve a desired SNR on velocity, acceleration, and jerk?
        Inputs:
            sep, seperr : observed separation and error in separation, any unit
            snr : desired signal to noise ratio.  Default = 5
            period : orbital period in years
        Returns:
            time needed to observe vel, acc, jerk to desired SNR in years.
    """
    from orbittools.orbittools import orbit_fraction
    v,a,j = orbit_fraction(sep, seperr, snr=snr)
    return v*period,a*period,j*period

def orbit_fraction_postional_uncertainty(time, period, sep = None, snr = 5):
    """ Given a certain observing timespan, what measurement precision
        is needed to obtain the desired SNR on vel, acc, and jerk?
        The orbit fraction is the ratio of observed time span to the period,
        and is also defined by the scale-free positional uncertainty given
        in the orbit_fraction() function.
        Inputs:
            time : observed time span in years
            period : orbital period in years
            sep : separation (optional)
            snr : desired signal-to-noise on measurements.  Currently set at 5.
        Returns:
            if separation is given, returns the required astrometric precision for
                snr = 5 for vel, acc, jerk in same units as sep input.
            if no separation if given, the scale-free positional uncertainty is 
                returned.
    """
    of = time/period
    v_scalefree_uncert = of
    a_scalefree_uncert = (of**2)*(5/9)
    j_scalefree_uncert = of**3
    if sep:
        return v_scalefree_uncert*sep, a_scalefree_uncert*sep, j_scalefree_uncert*sep
    else:
        return v_scalefree_uncert, a_scalefree_uncert, j_scalefree_uncert

############################################################################################################
#################################### Observing Planning ####################################################
############################################################################################################

def PlotObserving(SimbadName,DateString,LocationName,UTCOffset,
           plt_style = 'default',
           savefig = False,
           filename = 'observing_plot.png',
           form = 'png',
           dpi = 300,
           figsize=(7, 6),
           cmaps = ['Blues','Oranges','Purples','Reds','Greens'],
           radec = None
                       ):
    '''Args:
        SimbadName: [str or list of strings]: Simbad resolvable name of object(s)
        DateString [str]: A string of date. Ex: '2022-12-10'
        LocationName [str]: Name of Earth Location recognized by astropy.  Ex: 'Las Campanas Observatory'
        UTCOffset [int or flt]: offset of location from UTC
        plt_style [str]: specify the matplotlib plot style to use
        savefig [bool]: set to True to save the figure to file
        filename [str]: output file name is saving figure
        dpi [int]: dpi for fig
        figsize [tuple]: figure size
        cmaps [list]: colormaps to use for plotting motion of objects
    '''
    from astropy.coordinates import get_sun, get_moon
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    plt.style.use(plt_style)

    if SimbadName != None:
        SimbadName = SimbadName.split(',')
        nobs = len(SimbadName)
        #objects = SkyCoord.from_name(SimbadName)
        oblist = []
        for i in range(nobs):
            ob = SkyCoord.from_name(SimbadName[i])
            oblist.append(ob)
    else:
        radec = radec.split(',')
        nobs = len(radec)
        oblist = []
        for i in range(nobs):
            ob = SkyCoord(radec[i])
            oblist.append(ob)

    utc_offset = UTCOffset*u.hour
    location = EarthLocation.of_site(LocationName)

    TimeString = DateString + ' 00:00:00'

    midtime = Time(TimeString,scale='utc')+utc_offset
    # Establish times:
    midnight = midtime
    delta_midnight = np.linspace(-12, 12, 1000)*u.hour
    times = midnight + delta_midnight
    # Establish new AltAz frame at given location:
    altazframe = AltAz(obstime=times, location=location)
    # Sun, Moon, and all inputed object's into AltAz frame:
    sunaltazs = get_sun(times).transform_to(altazframe)
    moonaltazs = get_moon(times).transform_to(altazframe) 
    
    # Make the plot:
    fig = plt.figure(figsize=figsize)
    # Plot sun and moon:
    plt.plot(delta_midnight, sunaltazs.alt, color='orange', label='Sun')
    plt.plot(delta_midnight, moonaltazs.alt, color='darkgrey', label='Moon')
    
    # Plot object(s):
    for i,objects in enumerate(oblist):
        obaltazs = objects.transform_to(altazframe)
        plt.scatter(delta_midnight, obaltazs.alt, c=obaltazs.az, cmap=cmaps[i],s=8,lw=0)
    
    plt.fill_between(delta_midnight.value, 0, 90, sunaltazs.alt.value < -0, color='0.5', zorder=0)
    plt.fill_between(delta_midnight.value, 0, 90, sunaltazs.alt.value < -18, color='k', zorder=0)
    plt.ylim(0,90)
    plt.xticks(np.arange(13)*2 -12)
    leg = plt.legend(loc='upper left')
    plt.colorbar().set_label('Azimuth [deg]')
    plt.xlabel('Hours from Local Midnight')  
    plt.ylabel('Altitude [deg]')
    #plt.annotate(midtime.datetime, xy=(0.14,0.15),xycoords = 'figure fraction',fontsize=8)
    plt.grid(ls=':')
    plt.show()
    plt.close(fig)
    if savefig == True:
        plt.savefig(filename, format=form, dpi=dpi)
    return fig

def plot_airmass(ob,midtime,loc,names,
           plt_style = 'default',
           savefig = False,
           filename = 'observing_plot.png',
           form = 'png',
           dpi = 300,
           figsize=(7, 6),
           colors = ['b','orange','purple','red','green']
                       ):
    # Args:
    #  ob [object or array]: a single SkyCood object or a list of SkyCoord objects 
    #  midtime [object]: An Astropy Time object corresponding to midnight on the night of the observing in UTC
    #  loc [object]: Astropy EarthLocation object for location of observing
    #  name [string or list of strings]: String of the name of the object, or list of names
    from astropy.coordinates import get_sun, get_moon
    from astropy.coordinates import EarthLocation, AltAz
    from astropy.time import Time
    import numpy as np
    import astropy.units as u
    import matplotlib.pyplot as plt
    plt.style.use(plt_style)

    # Establish times:
    midnight = midtime
    delta_midnight = np.linspace(-12, 12, 1000)*u.hour
    times = midnight + delta_midnight
    # Establish new AltAz frame at given location:
    altazframe = AltAz(obstime=times, location=loc)
    # Sun, Moon, and all inputed object's into AltAz frame:
    sunaltazs = get_sun(times).transform_to(altazframe)
    moonaltazs = get_moon(times).transform_to(altazframe) 
    
    # Make the plot:
    fig = plt.figure()
    # Plot sun and moon:
    #plt.plot(delta_midnight, sunaltazs.alt, color='orange', label='Sun')
    #plt.plot(delta_midnight, moonaltazs.alt, color='darkgrey', label='Moon')
    # Plot object(s):
    try:
        l=len(ob)
    except:
        l=1
    if l==1:
        obaltazs = ob.transform_to(altazframe)
        plt.plot(delta_midnight, obaltazs.secz, label=names)
    else:
        for i in range(l):
            obaltazs = ob[i].transform_to(altazframe)
            plt.plot(delta_midnight, obaltazs.secz,label=names[i])
    
    plt.fill_between(delta_midnight, 0, 90, sunaltazs.alt < -0*u.deg, color='0.5', zorder=0)
    plt.fill_between(delta_midnight, 0, 90, sunaltazs.alt < -18*u.deg, color='k', zorder=0)
    plt.ylim(0,4)
    plt.xticks(np.arange(13)*2 -12)
    leg = plt.legend(loc='upper left')
    plt.xlabel('Hours from Local Midnight')  
    plt.ylabel('Airmass')
    plt.annotate(midtime.datetime, xy=(0.14,0.15),xycoords = 'figure fraction',fontsize=8)
    plt.grid(ls=':')
    plt.show()
    plt.close(fig)
    if savefig == True:
        plt.savefig(filename, format=form, dpi=dpi)
    return fig


def GetCurrentSiderealTime(LocationName):
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from datetime import datetime

    location = EarthLocation.of_site(LocationName)
    t = Time(datetime.utcnow(), scale='utc', location=location)
    return t.sidereal_time('apparent') 


def GetSiderealTime(TimeString,LocationName):
    from astropy.coordinates import EarthLocation
    from astropy.time import Time
    from datetime import datetime

    location = EarthLocation.of_site('Las Campanas Observatory')
    t = Time(TimeString, scale='utc', location=location)
    return t.sidereal_time('apparent')


############################################################################################################
######################## Exposure time estimates ###########################################################
############################################################################################################

def snr_for_MagAOX(t_exp, t_obs, filterband, mask = 'lyot stop', lyot_throughput = 0.6382, 
                    star_peak_to_halo_contrast = 1e-3, star_mag = 11,
                    companion_mag = 18, RON = 3.78, coronagraph = False, NRM = False, NRM_attentuation_factor = None):
    filters_F0 = {'Halpha':3.8e9,'ip':1.1e10,'rp':2.9e10,'zp':6.3e9,'gp':1.9e10,'CH4':0.875}
    filters_lambda0 = {'Halpha':0.656,'ip':0.762,'rp':0.615,'zp':0.908,'gp':0.525,'CH4':0.875} #central wavelength in um
    # focal plane mask throughout for lyotlg:
    fpm_throughput = {'Halpha':10**(-4.75),'gp':10**(-6),'ip':10**(-4.56),'rp':10**(-5),'zp':10**(-3.67)}
    if mask == 'fpm':
        throughput = fpm_throughput[filterband]
    elif mask == 'lyot stop':
        throughput = lyot_throughput
    # photon noise:
    F0 = filters_F0[filterband]
    star_peak_flux = (10**(-star_mag/2.5)) * F0 #photons per sec
    star_peak_flux = np.pi/4*star_peak_flux * throughput #photons. per sec per lod^2
    platescale = 0.006 #arcsec/pix
    lod_per_pix = 0.2063 * filters_lambda0[filterband] / 6.5 #arcsec
    pixel_side_in_lod = platescale / lod_per_pix # pixel side size in lod
    pixel_area_in_lod = pixel_side_in_lod**2
    star_peak_flux = star_peak_flux * pixel_area_in_lod # photons per pixel per second
    photon_noise_background = star_peak_flux * star_peak_to_halo_contrast
    if coronagraph:
        star_peak_flux = star_peak_flux * 1e-2
    if NRM:
        star_peak_flux = star_peak_flux * NRM_attentuation_factor
    
    # signal:
    if mask == 'lyot stop':
        comp_throughout = lyot_throughput
    elif mask == 'fpm':
        comp_throughout = 1
    companion_peak_flux = 10**(-companion_mag/2.5) * F0 #photons per sec
    companion_peak_flux = np.pi/4*companion_peak_flux * comp_throughout #photons per sec per lod^2
    companion_peak_flux = companion_peak_flux * pixel_area_in_lod # photons per pixel per second
    if NRM:
        companion_peak_flux = companion_peak_flux * NRM_attentuation_factor
    
    # strehl:
    # strehl = e^(-sigma^2); strehl = 0.46 at z' measured on sky, -> sigma^2 = 0.88 at 900nm, and 
    # sigma scales as lambda =>
    strehl = np.exp(-((0.88/0.908)*filters_lambda0[filterband])**2)
    N_exp = t_obs/t_exp
    
    # snr:
    signal_term = companion_peak_flux * strehl * t_obs
    noise_term = photon_noise_background * t_obs
    readout_term = (RON**2) * (N_exp)
    
    num = signal_term
    denom = np.sqrt( signal_term + noise_term + readout_term )
    SNR = num / denom
    return SNR, N_exp, star_peak_flux, companion_peak_flux

    # VIS-X exp time:
def snr_for_VISX(t_obs, filterband, star_peak_to_halo_contrast = 1e-3, star_mag = 11,
       companion_mag = 18):
    filters_F0 = {'Halpha':3.8e9,'ip':1.1e10,'rp':2.9e10,'zp':6.3e9,'gp':1.9e10,'CH4':0.875}
    filters_lambda0 = {'Halpha':0.656,'ip':0.762,'rp':0.615,'zp':0.908,'gp':0.525,'CH4':0.875} #central wavelength in um
        
    throughput = 0.1
    # photon noise:
    F0 = filters_F0[filterband]
    star_peak_flux = (10**(-star_mag/2.5)) * F0 #photons per sec
    star_peak_flux = np.pi/4*star_peak_flux * throughput #photons. per sec per lod^2
    platescale = 0.006 #arcsec/pix
    lod_per_pix = 0.2063 * filters_lambda0[filterband] / 6.5 #arcsec
    pixel_side_in_lod = platescale / lod_per_pix # pixel side size in lod
    pixel_area_in_lod = pixel_side_in_lod**2
    star_peak_flux = star_peak_flux * pixel_area_in_lod # photons per pixel per second
    photon_noise_background = star_peak_flux * star_peak_to_halo_contrast

    comp_throughout = throughput
    companion_peak_flux = 10**(-companion_mag/2.5) * F0 #photons per sec
    companion_peak_flux = np.pi/4*companion_peak_flux * comp_throughout #photons per sec per lod^2
    companion_peak_flux = companion_peak_flux * pixel_area_in_lod # photons per pixel per second
    
    # strehl:
    # strehl = e^(-sigma^2); strehl = 0.46 at z' measured on sky, -> sigma^2 = 0.88 at 900nm, and 
    # sigma scales as lambda =>
    strehl = np.exp(-((0.88/0.908)*filters_lambda0[filterband])**2)
    
    # snr:
    signal_term = companion_peak_flux * strehl * t_obs
    noise_term = photon_noise_background * t_obs
    
    num = signal_term
    denom = np.sqrt( signal_term + noise_term)
    SNR = num / denom
    return SNR

def plot_ADI_skyrotation(object_name, location, time, utc_offset):
    from astropy.coordinates import EarthLocation, AltAz, SkyCoord
    import matplotlib.pyplot as plt
    location = EarthLocation.of_site(location)
    ob = SkyCoord.from_name(object_name)
    midnight = time
    delta_midnight = np.linspace(-12, 12, 1000)*u.hour
    times = midnight + delta_midnight
    # Establish new AltAz frame at given location:
    altazframe = AltAz(obstime=times, location=location)
    obaltazs = ob.transform_to(altazframe)
    ZenithDistance = np.radians(90 - obaltazs.alt.value)
    az = np.radians(obaltazs.az.value)

    phi = 0.2506*np.cos(az)*np.cos(np.radians(location.lat.value)) / np.sin(ZenithDistance)
    phi = np.abs(phi*(u.deg/u.min))

    #%matplotlib notebook
    fig = plt.figure()
    plt.style.use('magrathea')
    plt.plot(delta_midnight,phi)
    plt.grid(ls=':')
    plt.xlabel('Hours since midnight')
    plt.ylabel('Field rotation rate [deg/min]')
    plt.tight_layout()
    return fig

############################################################################################################
############################################## Photometry ##################################################
############################################################################################################

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

def arcsec_to_lod(arcsec, diameter, lamb):
    lod = Get_LOD(lamb, diameter)
    return (arcsec.to(u.mas)/lod).value

def lod_to_arcsec(Nlod, diameter, lamb):
    lod_in_mas = Get_LOD(lamb, diameter)*u.mas
    return (Nlod * lod_in_mas).to(u.arcsec)

def lod_to_au(Nlod, distance, diameter, lamb):
    ''' Convert a distance in lamda over D to AU
        Inputs:
            lod [arcsec]: # of lambda over D to convert
            distance [pc]: distance to system in parsecs
            lamb [um]: filter central wavelength in microns
        Returns:
        
    '''
    import astropy.units as u
    distance = distance.to(u.pc)
    # 1 lambda/D in arcsec:
    sep_in_arcsec = lod_to_arcsec(Nlod, diameter, lamb)
    return (sep_in_arcsec * distance.value)*u.AU
    
    
def au_to_lod(au, distance, diameter, lamb):
    ''' Convert a physucal distance in AU to lambda over D
    '''
    distance = distance.to(u.pc)
    au = au.to(u.au)
    sep_in_arcsec = au.value/distance.value
    return arcsec_to_lod(sep_in_arcsec*u.arcsec, diameter, lamb)


def pixel_seppa_clio(x1,y1,x2,y2,imhdr=None):
    sep = np.sqrt((x1-x2)**2 + (y1-y2)**2)
    sep = sep*15.9
    sep = sep/1000
    if imhdr:
        pa = (np.degrees(np.arctan2((y2-y1),(x2-x1)))+270)%360
        NORTH_CLIO = -1.80
        derot = imhdr['ROTOFF'] - 180. + NORTH_CLIO
        pa = (pa + derot)
        return sep,pa%360.
    else:
        return sep


def Get_MagaoX_FWHM(filt):
    ''' Return lambda/D in pixels and mas for a given MagAO-X filter
    '''
    filters = {'z':0.908,'i':0.764,'r':0.613,'g':0.527}
    lamb_cent = filters[filt]
    lod = 0.206*lamb_cent/6.5
    lod = lod*u.arcsec.to(u.mas)
    pixscale = 6 #mas/pixel
    fwhm = lod/pixscale
    return fwhm, lod

# Magnitudes
def app_abs(m,d):
    # Convert apparent mag (m) into absolute mag given distance (d)
    from numpy import log10
    M = m-5.0*(log10(d)-1)
    return M

def app_abs_p(m,p):
    # Convert apparent mag (m) into absolute mag given parallax (p) in arcsec
    from numpy import log10
    M = m+5.0*(log10(p)+1)
    return M

def abs_app(M,d):
    # Convert absolute mag (M) into apparent mag given distance (d_
    from numpy import log10
    m = M+5.0*(log10(d)-1)
    return m

# Photometry and SNR
def findmax(data):  
    from numpy import unravel_index, argmax
    m = argmax(data)
    c = unravel_index(m, data.shape)
    return c

def dist_circle(xcen,ycen,x,y):
    '''Determine the distance in pixels of pixels from a given central pixel'''
    import numpy as np
    dx = np.abs(x-xcen)
    dy = np.abs(y-ycen)
    dist = np.sqrt(dx**2+dy**2)
    return dist

def aperture_annulus(image,sx,sy,r,r_in,r_out):
    """
    Returns a list of indicies which fall within the specified circular aperture and
    annulus around the source
    Args:
        image (2d float array): Image array extracted from fits file
        sx,sy (float): x and y pixel locations for the center of the source
        r (float): radius for aperture
        r_in,r_out (float): inner and outer radii for annulus
        
    Return:
        aperture_indicies (np.array): list of indicies within the aperture, in x,y
        annulus_indicies (np.array): list of indicies within the annulus, in x,y
    Written by: Logan Pearce, 2018
    """
    import warnings
    import numpy as np
    warnings.filterwarnings('ignore')
    # Measure the distances of all pixels in the image to the center of the source:
    xarray = np.int_(np.linspace(0,image.shape[1],image.shape[1]))
    yarray = np.int_(np.linspace(0,image.shape[0],image.shape[0]))
    index = np.linspace(0,xarray.shape[0]-1,xarray.shape[0], dtype=int)
    distances = np.zeros((xarray.shape[0],yarray.shape[0]))
    for xi,i1 in zip(xarray,index):
        for yi,i2 in zip(yarray,index):
            distances[i1,i2] = dist_circle(np.int_(sx),np.int_(sy),xi,yi)
    distances = np.int_(distances)
    # Make an array of indicies which fall within a annulus of specified inner and outer radius:
    annulus_indicies = np.where((distances>=np.int_(r_in))&(distances<=np.int_(r_out)))
    aperture_indicies = np.where(distances<=r)
    return aperture_indicies,annulus_indicies

def signal_noise_ratio(image,sx,sy,r,r_in,r_out):
    """
    Returns signal to noise ratio, signal value, noise value
    Args:
        image (2d float array): Image array extracted from fits file
        sx,sy (float): x and y pixel locations for the center of the source
        r (float): radius for aperture
        r_in,r_out (float): inner and outer radii for annulus
    Return:
        snr (float): signal-to-noise ratio
        signal (float): sum of all pixel values within the specified aperture minus sky background
        noise (float): noise in the signal calculated as std deviation of pixels in the sky annulus times sqrt of the area of the
        signal aperture
        poisson_noise (float): snr determined from the poisson noise in the source signal.  Poisson noise = sqrt(counts) [because variance
        of a poisson distribution is the parameter itself].  Poisson snr = Signal/sqrt(signal) = sqrt(signal)
    Written by: Logan Pearce, 2018
    """
    import warnings
    import numpy as np
    warnings.filterwarnings('ignore')
    ap_an = aperture_annulus(image,sx,sy,r,r_in,r_out)
    ap,skyan = ap_an[0],ap_an[1]
    apsum = np.sum(image[ap[1],ap[0]])
    skysum = np.sum(image[skyan[1],skyan[0]])
    skyarea = np.shape(skyan)[1]
    averagesky = skysum/skyarea
    signal = (apsum - np.shape(ap)[1]*averagesky)
    poisson_noise = np.sqrt(signal)
    noise = np.std(image[skyan[1],skyan[0]])
    noise = noise*np.sqrt(np.shape(ap)[1])
    snr = signal/noise
    return snr,signal,noise,poisson_noise

def snr_astropy(image,sx,sy,r,r_in,r_out):
    """
    Returns signal to noise ratio, signal value, noise value using Astropy's Photutils module
    Args:
        image (2d float array): Image array extracted from fits file
        sx,sy (float): x and y pixel locations for the center of the source
        r (float): radius for aperture
        r_in,r_out (float): inner and outer radii for annulus
    Return:
        snr (float): signal-to-noise ratio
        signal (float): sum of all pixel values within the specified aperture minus sky background
        noise (float): noise in the signal calculated as std deviation of pixels in the sky annulus times sqrt of the area of the
        signal aperture
        poisson_noise (float): snr determined from the poisson noise in the source signal.  Poisson noise = sqrt(counts) [because variance
        of a poisson distribution is the parameter itself].  Poisson snr = Signal/sqrt(signal) = sqrt(signal)
    Written by: Logan Pearce, 2018
    """
    import warnings
    warnings.filterwarnings('ignore')
    from photutils import CircularAperture, CircularAnnulus
    import numpy as np
    positions = (sx,sy)
    ap = CircularAperture(positions,r=r)
    skyan = CircularAnnulus(positions,r_in=r_in,r_out=r_out)
    apsum = ap.do_photometry(image)[0]
    skysum = skyan.do_photometry(image)[0]
    averagesky = skysum/skyan.area()
    signal = (apsum - ap.area()*averagesky)[0]
    n = ap.area()
    ap_an = aperture_annulus(image,sx,sy,r,r_in,r_out)
    skyan = ap_an[1]
    poisson_noise = np.sqrt(signal)
    noise = np.std(image[skyan[1],skyan[0]])
    noise = noise*np.sqrt(n)
    snr = signal/noise
    return snr,signal,noise,poisson_noise

def GetFofLambdaNaught(wavelength,flux,filt):
    '''Args:
        wavelength [arr]: wavelength array in um
        flux [arr]: flux array in ergs cm^-1 s^-1 cm^-2
        filt [myastrotools filter object]: filter
    '''
    from scipy.interpolate import interp1d
    f = interp1d(wavelength,flux)
    F0 = f(filt.central_wavelength*filt.wavelength_unit.to(u.um))

def GetPhotonsPerSec(wavelength, flux, filt, distance, radius, primary_mirror_diameter,
                    return_ergs_flux_times_filter = False, Omega = None):
    ''' Given a spectrum with wavelengths in um and flux in ergs cm^-1 s^-1 cm^-2, convolve 
    with a filter transmission curve and return photon flux in photons/sec
    
    Args:
        wavelength [arr]: wavelength array in um
        flux [arr]: flux array in ergs cm^-1 s^-1 cm^-2
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


############################################################################################################
####################################### Radial Profile ##################################################
############################################################################################################

def radial_data(data,annulus_width=1,working_mask=None,x=None,y=None,rmax=None):
    """
    From : https://www.astrobetter.com/wiki/python_radial_profiles
    By Ian J Crossfield
    r = radial_data(data,annulus_width,working_mask,x,y)
    
    A function to reduce an image to a radial cross-section.
    
    INPUT:
    ------
    data   - whatever data you are radially averaging.  Data is
            binned into a series of annuli of width 'annulus_width'
            pixels.
    annulus_width - width of each annulus.  Default is 1.
    working_mask - array of same size as 'data', with zeros at
                      whichever 'data' points you don't want included
                      in the radial data computations.
      x,y - coordinate system in which the data exists (used to set
             the center of the data).  By default, these are set to
             integer meshgrids
      rmax -- maximum radial value over which to compute statistics
    
     OUTPUT:
     -------
      r - a data structure containing the following
                   statistics, computed across each annulus:
          .r      - the radial coordinate used (outer edge of annulus)
          .mean   - mean of the data in the annulus
          .std    - standard deviation of the data in the annulus
          .median - median value in the annulus
          .max    - maximum value in the annulus
          .min    - minimum value in the annulus
          .numel  - number of elements in the annulus
    """
    
# 2010-03-10 19:22 IJC: Ported to python from Matlab
# 2005/12/19 Added 'working_region' option (IJC)
# 2005/12/15 Switched order of outputs (IJC)
# 2005/12/12 IJC: Removed decifact, changed name, wrote comments.
# 2005/11/04 by Ian Crossfield at the Jet Propulsion Laboratory
 
    import numpy as ny

    class radialDat:
        """Empty object container.
        """
        def __init__(self): 
            self.mean = None
            self.std = None
            self.median = None
            self.numel = None
            self.max = None
            self.min = None
            self.r = None

    #---------------------
    # Set up input parameters
    #---------------------
    data = ny.array(data)
    
    #if working_mask==None:
    if working_mask is None:
        working_mask = ny.ones(data.shape,bool)
    
    npix, npiy = data.shape
    if x==None or y==None:
        x1 = ny.arange(-npix/2.,npix/2.)
        y1 = ny.arange(-npiy/2.,npiy/2.)
        x,y = ny.meshgrid(y1,x1)

    r = abs(x+1j*y)

    if rmax==None:
        rmax = r[working_mask].max()

    #---------------------
    # Prepare the data container
    #---------------------
    dr = ny.abs([x[0,0] - x[0,1]]) * annulus_width
    radial = ny.arange(rmax/dr)*dr + dr/2.
    nrad = len(radial)
    radialdata = radialDat()
    radialdata.mean = ny.zeros(nrad)
    radialdata.std = ny.zeros(nrad)
    radialdata.median = ny.zeros(nrad)
    radialdata.numel = ny.zeros(nrad)
    radialdata.max = ny.zeros(nrad)
    radialdata.min = ny.zeros(nrad)
    radialdata.r = radial
    
    #---------------------
    # Loop through the bins
    #---------------------
    for irad in range(nrad): #= 1:numel(radial)
      minrad = irad*dr
      maxrad = minrad + dr
      thisindex = (r>=minrad) * (r<maxrad) * working_mask
      if not thisindex.ravel().any():
        radialdata.mean[irad] = ny.nan
        radialdata.std[irad]  = ny.nan
        radialdata.median[irad] = ny.nan
        radialdata.numel[irad] = ny.nan
        radialdata.max[irad] = ny.nan
        radialdata.min[irad] = ny.nan
      else:
        radialdata.mean[irad] = data[thisindex].mean()
        radialdata.std[irad]  = data[thisindex].std()
        radialdata.median[irad] = ny.nanmedian(data[thisindex])
        radialdata.numel[irad] = data[thisindex].size
        radialdata.max[irad] = data[thisindex].max()
        radialdata.min[irad] = data[thisindex].min()
    
    #---------------------
    # Return with data
    #---------------------
    
    return radialdata

def CenteredDistanceMatrix(n, ny = None, returnmesh = False):
    ''' Creates 2d array of the distance of each element from the center

    Parameters
    ----------
        n : flt
            x-dimension of 2d array
        ny : flt (optional)
            optional y-dimension of 2d array.  If not provided, array is square of dimension nxn
    
    Returns
    -------
        2d matrix of distance from center
    '''
    nx = n
    if ny:
        pass
    else:
        ny = nx
    center = ((nx-1)*0.5,(ny-1)*0.5)
    xx,yy = np.meshgrid(np.arange(nx)-center[0],np.arange(ny)-center[1])
    r=np.hypot(xx,yy)
    if returnmesh:
        return r, xx, yy
    return r 

def MakeRadialSubtractionMask(shape,r0,r1,phi0,phi1):
    ''' Make a mask that cuts out a region of radius r0<r<r1 and angle phi0<phi<phi1
    Args:
        shape (arr): shape of mask in pixels
        r0 (flt): inner radius of mask in pixels
        r1 (flt): outer radius of mask in pixels
        phi0 (flt): small angle of cutout region on degrees
        phi1 (flt): outer angle in degrees
        
    Returns:
        2d array of specified shape with masked regions of value 0 and
        unmasked regions valued 1
    '''
    r, xx, yy = CenteredDistanceMatrix(shape[0],ny=shape[1], returnmesh = True)
    mask = np.zeros(shape)
    mask[np.where((r > r0) & (r < r1))] = 1
    angle = np.arctan2(yy,xx)
    angle = (np.degrees(angle) + 270)%360
    mask[np.where((angle > phi0) & (angle < phi1))] = 0
    return mask

def SubtractRadProfile(im, boolmask):
    from myastrotools.tools import CenteredDistanceMatrix, radial_data
    from scipy.interpolate import interp1d
    r = CenteredDistanceMatrix(im.shape[0], returnmesh = False)
    radial_profile = radial_data(im,annulus_width=1,working_mask=boolmask,x=None,y=None,rmax=None)
    x = np.arange(np.max(r))
    rflat = np.int_(r.flat)
    f = interp1d(x, radial_profile.median)
    p = f(np.int_(r.flat)).reshape(r.shape)
    imsub = im - p
    return imsub
    
############################################################################################################
####################################### Calspec Models #####################################################
############################################################################################################

# Calspec models:
class Model(object):
    def __init__(self,StarName,SpT,V,BV,Name,Model,STIS,ra,dec,Vr,pmra,pmdec,SimbadName,AltSimbadName):
        ''' An obect containing all the relevant attributes for a Calspec spectral model
        https://www.stsci.edu/hst/instrumentation/reference-data-for-calibration-and-tools/
        astronomical-catalogs/calspec
        
        Attributes:
            StarName (str): Star Name in Calspec table
            SpT (str): SpT from Calspec table
            V (str): V mag  from Calspec table
            BV (str): B-V from Calspec table
            Name (str): Name from Calspec table
            Model (str): Model from Calspec table
            STIS (str): STIS from Calspec table
            ra (flt): RA from Calspec table in degreses
            dec (flt): DEC from Calspec table in degreses
            Vr (flt): Radial velocity from Calspec table
            pmra (flt): proper motion from Calspec table
            pmdec (flt): proper motion from Calspec table
            SimbadName (str): Simbad Name from Calspec table
            AltSimbadName (str):Alt Simbad Name from Calspec table
            wavelength (arr): model wavelengths in Ang
            wavelength_unit (str): unit for wavelength array
            flux (arr): model flux in erg s-1 cm-2 Ang-1
            flux_unit (str): Flux unit erg s-1 cm-2 Ang-1
            teff (flt): effective temperature from model header
            logg (flt): Log g from model header
            logz (flt): Log redshift from model header
            EBminusV (flt): E(B-V) extinction from model header
        '''
        self.StarName = StarName
        self.SpT = SpT
        self.V = V
        self.BV = BV
        self.Name = Name
        self.Model = Model
        self.STIS = STIS
        self.ra = ra
        self.dec = dec
        self.Vr = Vr
        self.pmra = pmra
        self.pmdec = pmdec
        self.SimbadName = SimbadName
        self.AltSimbadName = AltSimbadName
        model = Name.strip(' ')+Model.strip(' ')+'.fits'
        file = os.path.join(os.path.dirname(__file__), 'model_spectra/'+model)
        r = fits.open(file)
        data = r[1].data
        self.wavelength = data['WAVELENGTH']
        self.flux = data['FLUX']
        try:
            self.continuum = data['CONTINUUM']
        except:
            pass
        self.wavelength_unit = 'Ang'
        self.flux_unit = 'erg s-1 cm-2 Ang-1'
        try:
            t = r[0].header['TEFFGRAV'].split(',')
            self.teff = t[0]
            self.logg = t[1]
            self.logz = t[2]
            self.EBminusV = t[3]
        except:
            pass

def LoadCalspecModel(Name):
    ''' Load a Calspec Model object.
    Args:
        Name (str): The same name for the desired star given in the Calspec star list table.
    
    Returns:
        Model object
    '''
    import os
    import pandas as pd
    file = os.path.join(os.path.dirname(__file__), 'model_spectra/calspec_model_info.csv')
    jj = pd.read_csv(file)
    for i,n in enumerate(jj['Name']):
        if Name in n.strip(' '):
            index = i
    j = jj.loc[index]
    m = Model(j['Star name'].strip(' '), j['SpT'].strip(' '), j['V'].strip(' '), \
          j['B-V'].strip(' '), j['Name'].strip(' '), j['Model'].strip(' '), \
          j['STIS '].strip(' '), j['RA deg'], j['DEC deg'], \
          j['Vr'], j['PMRA'], j['PMDEC'], \
          j['Simbad Name'], j['Alt. Simbad Name'])
    return m
    
    
def LoadCalspecModels():
    ''' Load all the models into a dictionary.
    Returns:
        dictionary of model Name and Model object for all calspec models.
    '''
    from myastrotools.tools import update_progress
    import pandas as pd
    import os
    file = os.path.join(os.path.dirname(__file__), 'model_spectra/calspec_model_info.csv')
    jj = pd.read_csv(file)
    dictionary = {}
    for i,n in enumerate(jj['Name']):
        try:
            j = jj.loc[i]
            m = Model(j['Star name'].strip(' '), j['SpT'].strip(' '), j['V'].strip(' '), \
                j['B-V'].strip(' '), j['Name'].strip(' '), j['Model'].strip(' '), \
                j['STIS '].strip(' '), j['RA deg'], j['DEC deg'], \
                j['Vr'], j['PMRA'], j['PMDEC'], \
                j['Simbad Name'], j['Alt. Simbad Name'])
            Name = j['Name']
            dictionary.update({Name.strip(' '):m})
        except:
            pass
    return dictionary

def GetCalspecInfo():
    import pandas as pd
    import os
    file = os.path.join(os.path.dirname(__file__), 'model_spectra/calspec_model_info.csv')
    j = pd.read_csv(file)
    return j

def LoadCalspecSpT(SpT):
    file = os.path.join(os.path.dirname(__file__), 'model_spectra/calspec_model_info.csv')
    jj = pd.read_csv(file)
    dictionary = {}
    for i,n in enumerate(jj['SpT']):
        if SpT in n:
            try:
                j = jj.loc[i]
                m = Model(j['Star name'].strip(' '), j['SpT'].strip(' '), j['V'].strip(' '), \
                    j['B-V'].strip(' '), j['Name'].strip(' '), j['Model'].strip(' '), \
                    j['STIS '].strip(' '), j['RA deg'], j['DEC deg'], \
                    j['Vr'], j['PMRA'], j['PMDEC'], \
                    j['Simbad Name'], j['Alt. Simbad Name'])
                Name = j['Name']
                dictionary.update({Name.strip(' '):m})
            except:
                pass
    return dictionary

def LoadElBadryCatalog():
    ''' Load the catalog of Common Proper Motion objects in Gaia EDR3 from 
    El-Badry et al. 2021 (https://zenodo.org/record/4435257#.YfmZ0_XMKCw) into 
    a pandas DataFrame.
    '''
    import os
    import pandas as pd
    from astropy.io import fits
    file = os.path.join(os.path.dirname(__file__), 'El-Badry-all_columns_catalog.fits')
    cat = fits.getdata(file)
    c = pd.DataFrame(cat)
    return c




############################################################################################################
####################################### ADI/RDI/SDI Tools ##################################################
############################################################################################################


def GetStarLocation(im, threshold = 5000, lamb_cent = 0.908):
    from photutils import DAOStarFinder
    lod = 0.206*lamb_cent/6.5
    lod = lod*u.arcsec.to(u.mas)
    pixscale = 6 #mas/pixel
    fwhm = lod/pixscale
    #fwhm = 10

    daofind = DAOStarFinder(fwhm=fwhm, threshold=threshold) 
    sources = daofind(im)
    if len(sources) > 1:
        print('more than one source found, maybe change threshold?')
        return 
    if len(sources) != 1:
        print('no sources found')
        return 
    
    return (sources[0]['xcentroid'],sources[0]['ycentroid'])

def FlipImage(im):
    return im[:,::-1]

def ShiftImage(im, dx = None, dy = None, xc = None, yc = None, center = None,
              interp = 'bicubic', bordermode = 'constant', cval = 0, scale = 1):
    """Move an image by [dx,dy] pixels.  Uses OpenCV image processing package
    Written by Logan A. Pearce, 2022

    Dependencies: OpenCV

    Parameters:
    -----------
    im : 2d array
       2d image array
    dx, dy : None or flt
        amount of shift in x and y directions.  If None, center and xc,yc must be entered
    center : None or tuple
       (x,y) subpixel location for center of image.  If center=None,
       computes the center pixel of the image.
    xc, yc: None or flt
        if not entering a dx,dy shift, xc, yc are the location of the star in the image from which
        the shift will be computed.
    interp : str
        Interpolation mode for OpenCV.  Either nearest, bilinear, bicubic, or lanczos4.
        Default = bicubic
    bordermode : str
        How should OpenCV handle the extrapolation at the edges.  Either constant, edge, 
        symmetric, reflect, or wrap.  Default = constant
    cval : int or np.nan
        If bordermode = constant, fill edges with this value.  Default = 0
    scale : int or flt
        scale parameter for OpenCV.  Scale = 1 does not scale the image.  Default = 1

    Returns:
    --------
    imrot : 2d arr 
       rotated image with north up east left
    """
    import cv2
    
    if interp == 'bicubic':
        intp = cv2.INTER_CUBIC
    elif interp == 'lanczos4':
        intp = cv2.INTER_LANCZOS4
    elif interp == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interp == 'nearest':
        intp = cv2.INTER_NEAREST
    else:
        raise ValueError('Interpolation mode: please enter nearest, bilinear, bicubic, or lanczos4')
        
    if bordermode == 'constant':
        bm = cv2.BORDER_CONSTANT 
    elif bordermode == 'edge':
        bm = cv2.BORDER_REPLICATE 
    elif bordermode == 'symmetric':
        bm = cv2.BORDER_REFLECT
    elif bordermode == 'reflect':
        bm = cv2.BORDER_REFLECT_101
    elif bordermode == 'wrap':
        bm = cv2.BORDER_WRAP
    else:
        raise ValueError('Border mode: please enter constant, edge, symmetric, reflect, or wrap')
        
        
    if not dx:
        center = (0.5*((im.shape[1])-1),0.5*((im.shape[0])-1))
        dx,dy = xc-center[0],yc-center[1]
    num_rows, num_cols = im.shape[:2]
    translation_matrix = np.float32([ [1,0,dx], [0,1,dy] ])   
    imshift = cv2.warpAffine(im, translation_matrix, (num_cols, num_rows), 
                             flags=intp, borderMode=bm, borderValue=cval)  
    return imshift

def RotateImage(image, parang, rotationoffset, center = None, interp = 'bicubic', 
                bordermode = 'constant', cval = 0, scale = 1):
    """Rotate image to north up east left.  Uses OpenCV image processing package
       Written by Logan A. Pearce, 2022
       
       Dependencies: OpenCV

       Parameters:
       -----------
       image : 2d array
           2d image array
       parang : flt
           paralactic angle from header of image to be rotated
       center : None or tuple
           (x,y) subpixel location for center of rotation.  If center=None,
           computes the center pixel of the image.
       interp : str
            Interpolation mode for OpenCV.  Either nearest, bilinear, bicubic, or lanczos4.
            Default = bicubic
       bordermode : str
            How should OpenCV handle the extrapolation at the edges.  Either constant, edge, 
            symmetric, reflect, or wrap.  Default = constant
       cval : int or np.nan
            If bordermode = constant, fill edges with this value.  Default = 0
       scale : int or flt
            scale parameter for OpenCV.  Scale = 1 does not scale the image.  Default = 1
           
       Returns:
       --------
       imrot : 2d arr 
           rotated image with north up east left
    """
    import cv2
    
    if interp == 'bicubic':
        intp = cv2.INTER_CUBIC
    elif interp == 'lanczos4':
        intp = cv2.INTER_LANCZOS4
    elif interp == 'bilinear':
        intp = cv2.INTER_LINEAR
    elif interp == 'nearest':
        intp = cv2.INTER_NEAREST
    else:
        raise ValueError('Interpolation mode: please enter nearest, bilinear, bicubic, or lanczos4')
        
    if bordermode == 'constant':
        bm = cv2.BORDER_CONSTANT 
    elif bordermode == 'edge':
        bm = cv2.BORDER_REPLICATE 
    elif bordermode == 'symmetric':
        bm = cv2.BORDER_REFLECT
    elif bordermode == 'reflect':
        bm = cv2.BORDER_REFLECT_101
    elif bordermode == 'wrap':
        bm = cv2.BORDER_WRAP
    else:
        raise ValueError('Border mode: please enter constant, edge, symmetric, reflect, or wrap')
        
    derot = parang + rotationoffset
    y, x = image.shape
    if not center:
        center = (0.5*((image.shape[1])-1),0.5*((image.shape[0])-1))
    M = cv2.getRotationMatrix2D(center, derot, scale)
    imrot = cv2.warpAffine(image, M, (x, y),flags=intp, borderMode=bm, borderValue=cval)

    return imrot


def MakeImageStack(image_list, center_wavelength, center_image = True,
                   flip_image = False, rotate_image = False, rotationoffset = None):
    ''' Make a stack of images in a data set for differential imaging processes.
    
    Args:
        image_list (list): list of image files to incorporate into stack
        center_wavelength (flt): central wavelength of filter band of images
        center_image (bool): if true, find the center of the star in the image and shift
            if to the center of the frame. Default = True
        flip_image (bool): if true, flip the image horizontally. Default = False
        rotate_image (bool): if true, get the paralactic angle and rotate to north up east left. 
            Default = False.
        rotationoffset (flt): offset to correct paralactic angle in header to true north. Required if
            rotate_image = True.
        
    '''
    
    from myastrotools.tools import update_progress
    im = fits.getdata(image_list[0])
    stack = np.zeros((len(image_list),*im.shape))
    count = 0
    skipped = []
    skipped_index = []

    parang_dict = {}

    import warnings
    warnings.filterwarnings('ignore')

    for i in range(0,len(image_list)):
        im = fits.getdata(image_list[i])  

        try:
            im = fits.getdata(image_list[i])
            imhdr = fits.getheader(image_list[i])
            if center_image:
                from myastrotools.tools import GetStarLocation, ShiftImage
                xc,yc = 0.5*(im.shape[0]-1),0.5*(im.shape[1]-1)
                starloc = GetStarLocation(im, lamb_cent=center_wavelength)
                dx,dy = xc-starloc[0],yc-starloc[1]
                im = ShiftImage(im, dx = dx, dy = dy)
            if flip_image:
                from myastrotools.tools import FlipImage
                im = FlipImage(im)
            if rotate_image:
                from myastrotools.tools import RotateImage
                parang = {image_list[i]:imhdr['PARANG']}
                im = RotateImage(im, parang, rotationoffset, center = None, interp = 'bicubic', 
                            bordermode = 'constant', cval = 0, scale = 1)
            # put into the final cube:
            stack[count,:,:] = im

            parang_dict.update({image_list[i]:imhdr['PARANG']})

            count += 1

        except:
            #print('couldnt find source, skipping')
            skipped.append(image_list[i])
            skipped_index.append(i)

        update_progress(i,len(image_list)-1)

    stack = stack[:count,:,:]
    
    return stack, parang_dict, skipped, skipped_index



def KLIPSubtractScienceImage(scienceimage, K_klip, KLIPBasis, immean):
    shape=scienceimage.shape
    p = shape[0]*shape[1]
    
    # Reshape science target into 1xp array:
    T_reshape = np.reshape(scienceimage,(p))
    # Subtract mean from science image:
    T_meansub = T_reshape - immean[None,:]
    # Make K_klip number of copies of science image
    # to use fast vectorized math:
    T_meansub = np.tile(T_meansub, (np.max(K_klip), 1))
    
    # Soummer 2.2.4
    # Project science target onto KL Basis:
    projection_sci_onto_basis = np.dot(T_meansub,KLIPBasis.T)
    # This produces a (K_klip,K_klip) sized array of identical
    # rows of the projected science target.  We only need one row:
    projection_sci_onto_basis = projection_sci_onto_basis[0]
    # This fancy math let's you use fewer modes to subtract:
    lower_triangular = np.tril(np.ones([np.max(K_klip), np.max(K_klip)]))
    projection_sci_onto_basis_tril = projection_sci_onto_basis * lower_triangular
    # Create the final psf estimator by multiplying by the basis modes:
    Ihat = np.dot(projection_sci_onto_basis_tril[K_klip-1,:], KLIPBasis)
    
    # Soummer 2.2.5
    # Truncate the science image to the different number of requested modes to use:
    outputimage = T_meansub[:np.size(K_klip),:]
    # Subtract estimated psf from science image:
    outputimage = outputimage - Ihat
    # Reshape to 
    outputimage = np.reshape(outputimage, (np.size(K_klip),*shape))
    
    return outputimage


def MakePlanet(template, C, TC):
    ''' Make a simulated planet psf with desired contrast using template psf

    Parameters:
    -----------
    template : 2d image
        sample PSF for making simulated planet
    C : flt
        desired contrast in magnitudes
    TC : flt
        known contrast of template psf relative to science target
    Returns:
    --------
    2d arr
        scaled simulated planet psf with desired contrast to science target
    '''
    # Amount of magnitudes to scale template by to achieve desired
    # contrast with science target:
    D = C - TC
    # Convert to flux:
    scalefactor = 10**(-D/2.5)
    # Scale template pixel values:
    Pflux = template*scalefactor
    return Pflux


def InjectPlanet(image, parang, rotationoffset, template, sep, pa, contrast, TC, xc, yc, 
                 sepformat = 'lambda/D', 
                 pixscale = 6,
                 wavelength = 'none',
                 inject_negative_signal = False
                ):
    ''' Using a template psf, place a fake planet at the desired sep, pa, and
        contrast from the central object.  PA is measured relative to true north
        (rather than up in image)

    Parameters:
    -----------
    image : 2d array
        science image
    parang : flt
        paralactic angle from science image header
    template : 2d array
        template psf with known contrast to central object
    sep : flt or fltarr
        separation of planet placement in either arcsec, mas, pixels, or lambda/D
    pa : flt or fltarr
        position angle of planet relative to north in DEG
    contrast : flt or fltarr
        desired contrast of planet with central object
    TC : flt
        template contrast, known contrast of template psf relative to science target
    xc, yc : flt
        x,y pixel position of central object
    sepformat : str
        format of inputted desired separation. Either 'arcsec', 'mas', pixels', or 'lambda/D'.
        Default = 'pixels'
    pixscale : flt
        pixelscale in mas/pixel.  Default = 15.9 mas/pix, pixscale for CLIO narrow camera
    wavelength : flt
        central wavelength of filter in microns, needed if sepformat = 'lambda/D'
    box : int
        size of template box.  Template will be box*2 x box*2
    
    Returns:
    --------
    2d arr
        image with fake planet with desired parameters. 
    '''
    #from myastrotools.tools import MakePlanet
    # sep input into pixels
    if sepformat == 'arcsec':
        pixscale = pixscale/1000 # convert to arcsec/pix
        sep = sep / pixscale
    if sepformat == 'mas':
        sep = sep / pixscale
    if sepformat == 'lambda/D':
        #from myastrotools.tools import lod_to_pixels
        if wavelength == 'none':
            raise ValueError('wavelength input needed if sepformat = lambda/D')
        sep = lod_to_pixels(sep, wavelength)
    # pa input - rotate from angle relative to north to angle relative to image up:
    #    do the opposite of what you do to derotate images
    
    derot = parang + rotationoffset
    pa = (pa + derot)
        
    # Get cartesian location of planet:
    xx = sep*np.sin(np.radians((pa)))
    yy = sep*np.cos(np.radians((pa)))
    xs = np.int_(np.floor(xc-xx))
    ys = np.int_(np.floor(yc+yy))
    # Make planet from template at desired contrast
    from myastrotools.tools import MakePlanet
    Planet = MakePlanet(template, contrast, TC)
    # Make copy of image:
    synth = image.copy()
    # Get shape of template:
    boxy, boxx = np.int_(Planet.shape[0]/2),np.int_(Planet.shape[1]/2)
    x,y = xs,ys
    ymin, ymax = y-boxy, y+boxy
    xmin, xmax = x-boxx, x+boxx
    # Correct for sources near image edge:
    delta = 0
    if ymin < 0:
        delta = ymin
        ymin = 0
        Planet = Planet[(0-delta):,:]
    if ymax > image.shape[0]:
        delta = ymax - image.shape[0]
        ymax = image.shape[0]
        Planet = Planet[:(2*boxy-delta) , :]
    if xmin < 0:
        delta = xmin
        xmin = 0
        Planet = Planet[:,(0-delta):]
    if xmax > image.shape[1]:
        delta = xmax - image.shape[1]
        xmax = image.shape[1]
        Planet = Planet[:,:(2*boxx-delta)]
    if inject_negative_signal:
        Planet = Planet * (-1)
    # account for integer pixel positions:
    if synth[ymin:ymax,xmin:xmax].shape != Planet.shape:
        try:
            synth[ymin:ymax+1,xmin:xmax] = synth[ymin:ymax+1,xmin:xmax] + (Planet)
        except:
            try:
                synth[ymin:ymax,xmin:xmax+1] = synth[ymin:ymax,xmin:xmax+1] + (Planet)
            except:
                synth[ymin:ymax+1,xmin:xmax+1] = synth[ymin:ymax+1,xmin:xmax+1] + (Planet)
    else:
        synth[ymin:ymax,xmin:xmax] = synth[ymin:ymax,xmin:xmax] + (Planet)
    return synth

def GetMag(image, x, y, radius = None, returnflux = False, returntable = False):
    ''' Compute instrument magnitudes of one object.

    Parameters:
    -----------
    image : 2d array
        science image
    x,y : flt
        x and y pixel location of center of star
    radius : flt
        pixel radius for aperture.  Default = 3.89, approx 1/2 L/D for 
        CLIO 3.9um 
    r_in, r_out : flt
        inside and outside radius for background annulus.  Default = 10,12
    returnflux : bool
        if true, return the instrument mag and the raw flux value.
    returntable : bool
        if true, return the entire photometry table.
    Returns:
    --------
    flt
        instrument magnitudes of source
    flt
        signal to noise ratio
    '''
    from photutils import CircularAperture, aperture_photometry
    # Position of star:
    positions = [(x,y)]
    # Get sum of all pixel values within radius of center:
    aperture = CircularAperture(positions, r=radius)
    # Do photometry on star:
    phot_table = aperture_photometry(image, aperture)
    m =(-2.5)*np.log10(phot_table['aperture_sum'][0])
    if returnflux:
        return m, phot_table['aperture_sum'][0]
    if returntable:
        phot_table['Mag'] = m
        return m, phot_table
    return m

def GetTemplateContrast(image1,image2,pos1,pos2,**kwargs):
    ''' Return contrast of component 2 relative to 1 in magnitudes

    Parameters:
    ------------
    image1 : 2d array
        science image
    image2 : 2d array
        image of other object
    pos1 : arr
        x and y pixel location of center of star1 in order [x1,y1]
    pos2 : arr
        x and y pixel location of center of star2 in order [x2,y2]
    kwargs : 
        args to pass to mag function
    Returns:
    --------
    flt
        contrast in magnitudes of B component relative to A component
        
    '''
    from myastrotools.tools import GetMag
    mag1 = GetMag(image1,pos1[0],pos1[1], **kwargs)
    mag2 = GetMag(image2,pos2[0],pos2[1], **kwargs)
    return mag2 - mag1


############################################################################################################
############################ Modeling IFU Spectral Obs #####################################################
############################################################################################################

def ConvolveLSF(wavelengths, flux, R):
    ''' Convolve a spectrum S with given resolution R with Gaussian 
    kernel in velocity space to simulate the spectrograph LSF
    
    Args:
        wavelengths (1xN arr): spectrum model wavelengths
        flux (1xN arr): spectrum model flux
        R (int): spectrograph resolution
        
    Returns:
        arr: model convolved with Gaussian kernel
    
    '''
    # FWHM of Gaussian kernel in velocity space is c/R
    import numpy as np
    import astropy.constants as c
    fwhm = c.c/R
    
    from astropy.modeling.models import Gaussian1D
    # create Gaussian model:
    weights = Gaussian1D(mean = 0, stddev = fwhm/2.355, amplitude = 1)
    # create range of velocities to project Gaussian onto:
    v = np.linspace(-2*fwhm, 2*fwhm, 1000)
    
    # create lookup spline:
    from scipy.interpolate import UnivariateSpline
    S_spline = UnivariateSpline(wavelengths, flux)
    
    # create empty container:
    integrand = np.zeros((len(flux),len(v)))
    # for each velocity:
    for i in range(len(v)):
        vel = v[i]
        # get the Doppler shift: delta_lambda = delta_v/c * lambda
        shift = vel / c.c.to(u.m/u.s)
        # shift wavelengths by that amount:
        shifted_wavelengths = wavelengths - shift*wavelengths
        # get the new signal at the Doppler shifted wavelength:
        newS = S_spline(shifted_wavelengths)
        # apply weight to shifted spectrum from the Gaussian value at that vel:
        weightedS = weights(vel) * newS
        # add to empty container:
        integrand[:,i] = weightedS
        
    # sum all weighted spectra along wavelength axis:
    flux_convolved = np.sum(integrand, axis = 1)
    
    return flux_convolved

def SpectralSampling(R, wavelengths, flux, n_lambda = 100000):
    ''' Bin spectrum down due to instrument resolving power.
    
    Args:
        R (int): spectrograph resolution
        wavelengths (1xN arr): spectrum model wavelengths
        flux (1xN arr): spectrum model flux
        n_lambda (int): number of wavelengths for resampling spectra onto larger grid
        
    Returns:
        arr: model binned down
        
    '''
    import numpy as np
    #### Get bin boundaries:
    # start at the lowest model wavelength:
    BinMins = np.array([np.min(wavelengths)])
    # delta_lambda = lambda / R:
    dl = BinMins/R
    # bin upper boundry = lower + dl:
    BinMaxes = BinMins+dl
    # get the middle of the bin:
    BinMids = BinMins+(0.5*dl)
    
    # for each bin:
    while np.max(BinMids) < np.max(wavelengths):
        # add new bin lower boundry to list
        BinMins = np.append(BinMins,BinMaxes[-1])
        # compute dl
        dl = BinMins[-1]/R
        # add new bin max and mid to arrays:
        BinMaxes = np.append(BinMaxes, BinMins[-1]+dl)
        BinMids = np.append(BinMids, BinMins[-1]+(0.5*dl))
        # repeat until we reach the end of the model
        
    # resample onto finer grid:
    from scipy.interpolate import interp1d
    f = interp1d(wavelengths,flux)
    wnew = np.linspace(np.min(wavelengths),np.max(wavelengths), n_lambda)
    fnew = f(wnew)
    
    ### compute spectrum:
    observed_mean = np.array([])
    observed_median = np.array([])
    # for each bin:
    for i in range(len(BinMids)):
        # find the model points that fall inside the bin:
        inds = np.where( (wnew < BinMaxes[i]) & (wnew >= BinMins[i]) )
        # compute the mean/median of those points:
        observed_mean = np.append(observed_mean, np.mean(fnew[inds]))
        observed_median = np.append(observed_median, np.median(fnew[inds]))
        
    return observed_mean, BinMids

def SimulateObservedSpectrum(wavelengths, flux, R, n_lambda = 1000):
    # Convolve with LSF:
    flux_convolved = ConvolveLSF(wavelengths, flux, R)
    # Bin down:
    ObservedSpectrum, BinMids = SpectralSampling(R, wavelengths, flux_convolved, n_lambda = 100000)
    
    return ObservedSpectrum, BinMids


def MultiplyFilter(filt_lamb, filt_trans, spec_lamb, spec_flux):
    ''' Multiply a spectral model by a filter profile
    
    Args:
        filt_lamb, filt_trans (1xN arr): filter wavelength and transmission. Transmission must be
            in decimal not percent.  Filter wavelength must be same units as spectral model
        spec_lamb, spec_flux (1xN arr): spectrum wavelength and flux
        
    Returns:
        spectral model wavelengths
        spectral model times filter transmission
    '''
    # Interpolate filter onto spectral model wavelengths:
    from scipy.interpolate import interp1d
    f = interp1d(filt_lamb,filt_trans,fill_value="extrapolate")

    ind = np.where((spec_lamb > np.min(filt_lamb)) &
                   (spec_lamb < np.max(filt_lamb))
                  )[0]

    # resample filter onto model wavelengths:
    filt_lamb_resample = f(spec_lamb[ind])
    spec_flux_times_filter = spec_flux[ind] * filt_lamb_resample
    
    return spec_lamb[ind], spec_flux_times_filter

def GetContrast(line_filt_lamb, line_filt_trans, cont_filt_lamb, cont_filt_trans, spec_lamb, spec_flux):
    ''' Get the contrast between a narrow line filter and a narrow continuum filter.
    
    Args:
        line_filt_lamb, line_filt_trans (1xN arr): line filter wavelength and transmission. Transmission must be
            in decimal not percent.  Filter wavelength must be same units as spectral model
        cont_filt_lamb, cont_filt_trans (1xN arr): continuum filter wavelength and transmission. Transmission 
            must be in decimal not percent.  Filter wavelength must be same units as spectral model
        spec_lamb, spec_flux (1xN arr): spectrum wavelength and flux
        
    Returns:
        spectral model wavelengths
        spectral model times filter transmission
    
    '''
    spec_times_linefilter = MultiplyFilter(line_filt_lamb, line_filt_trans, spec_lamb, spec_flux)
    spec_times_contfilter = MultiplyFilter(cont_filt_lamb, cont_filt_trans, spec_lamb, spec_flux)
    
    linefiltersum = np.trapz(spec_times_linefilter[1], spec_times_linefilter[0])
    contfiltersum = np.trapz(spec_times_contfilter[1], spec_times_contfilter[0])
    
    flux_contrast =  linefiltersum / contfiltersum
    
    logflux_contrast = np.log10(linefiltersum) - np.log10(contfiltersum)
                    
    mag_contrast = -2.5*np.log10(linefiltersum) - (-2.5*np.log10(contfiltersum))
    
    return spec_times_linefilter, spec_times_contfilter, flux_contrast, logflux_contrast, mag_contrast


############################################################################################################
############################################## Stats #######################################################
############################################################################################################

def ComputeChi2(array, measurements):
    chi = 0
    for i in range(len(array)):
        chi += ( (array[i][0] - measurements[i]) / array[i][1] ) ** 2
    return chi

def add_in_quad(thing):
    """
    Add elements of an array of things in quadrature
    """
    out = 0
    for t in thing:
        out += t**2
    return np.sqrt(out)

def freedman_diaconis(array):
    '''Compute the optimal number of bins for a 1-d histogram using the Freedman-Diaconis rule of thumb
       Bin width = 2IQR/cuberoot(N)
       Inputs:
           array (arr): flattened array of data
        Returns:
           bin_width (flt): width of bin in optimal binning
           n (int): number of bins
    '''
    import numpy as np
    # Get number of observations:
    N = np.shape(array)[0]
    # Get interquartile range:
    iqr = np.diff(np.quantile(array, q=[.25, .75]))
    bin_width = (2.*iqr)/(N**(1./3.))
    n = int(((np.max(array) - np.min(array)) / bin_width)+1)
    return bin_width, n

def mode(array):
    import numpy as np
    from myastrotools.stats import freedman_diaconis
    n, bins = np.histogram(array, freedman_diaconis(array)[1], density=True)
    max_bin = np.max(n)
    bin_inner_edge = np.where(n==max_bin)[0]
    bin_outer_edge = np.where(n==max_bin)[0]+1
    # value in the middle of the highest bin:
    mode=(bins[bin_outer_edge] - bins[bin_inner_edge])/2 + bins[bin_inner_edge]
    return mode[0]

def ci(array,interval):
    """
    Returns a confidence interval for any distribution, regardless of shape.
    Args:
        distribution array (1d float array): array of samples comprising a pdf
        interval (float): decimal representing desired confidence interval. example: 0.95 = 95% CI
    Return:
        median value of distribution, upper and lower ci bounds, and plus/minus values 
        (ex: 4.5 ^{+1.1} _{-0.92})
        and mode computed as the middle of the largest histogram bin.
    Written by: Logan Pearce, 2018
    """
    import numpy as np
    from myastrotools.stats import mode
    sorts = np.sort(array)
    nsamples = sorts.shape[0]
    # Median
    m = np.int_(0.5*nsamples)
    median = sorts[m]
    # Mode
    mode = mode(array)
    # Lower 
    bounds = (1-interval)/2.
    b = np.int_(bounds*nsamples)
    bottom = sorts[b]
    # Upper 
    t = np.int_((1-bounds)*nsamples)
    top = sorts[t]

    minus = median - bottom
    plus = top-median
    return median,bottom,top,minus,plus,mode

def min_credible_interval(x, alpha):
    """
    Returns a minimum credible interval for Bayesian posterior
    Args:
        x (1d float array): array of posterior values
        alpha (float): decimal representing desired probability of Type 1 error. 
             example: For 68.3% Min CI, enter (1-0.683)
    Return:
        HDI min and max
    From: https://github.com/aloctavodia/Doing_bayesian_data_analysis/blob/master/hpd.py

    Example on using this to give a min credible interval for given desired percentage:
        sorted = np.sort(data_array)
        frac=0.683
        print calc_min_interval(sorted,(1-frac))
    """
    import numpy as np
    np.sort(x)
    n = len(x)
    cred_mass = 1.0-alpha

    interval_idx_inc = int(np.floor(cred_mass*n))
    n_intervals = n - interval_idx_inc
    interval_width = x[interval_idx_inc:] - x[:n_intervals]

    if len(interval_width) == 0:
        raise ValueError('Too few elements for interval calculation')

    min_idx = np.argmin(interval_width)
    hdi_min = x[min_idx]
    hdi_max = x[min_idx+interval_idx_inc]
    return hdi_min, hdi_max



    
    
    
    
    
############################################################################################################
######################################## Filter Curves #####################################################
############################################################################################################

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

def GetEffectiveWidth(wavelength, transmission):
    import numpy as np
    area_under_curve = np.trapz(transmission, x = wavelength)
    return np.abs(area_under_curve)

import astropy.units as u

class Clio39Filter(object):
    def __init__(self):
        self.wavelength = np.array([3.8,3.801,3.802,3.803,3.804,3.805,3.806,3.807,3.808,3.809,3.81,3.811,3.812,3.813,3.814,3.815,3.816,3.817,3.818,3.819,3.82,3.821,3.822,3.823,3.824,3.825,3.826,3.827,3.828,3.829,3.83,3.831,3.832,3.833,3.834,3.835,3.836,3.837,3.838,3.839,3.84,3.841,3.842,3.843,3.844,3.845,3.846,3.847,3.848,3.849,3.85,3.851,3.852,3.853,3.854,3.855,3.856,3.857,3.858,3.859,3.86,3.861,3.862,3.863,3.864,3.865,3.866,3.867,3.868,3.869,3.87,3.871,3.872,3.873,3.874,3.875,3.876,3.877,3.878,3.879,3.88,3.881,3.882,3.883,3.884,3.885,3.886,3.887,3.888,3.889,3.89,3.891,3.892,3.893,3.894,3.895,3.896,3.897,3.898,3.899,3.9,3.901,3.902,3.903,3.904,3.905,3.906,3.907,3.908,3.909,3.91,3.911,3.912,3.913,3.914,3.915,3.916,3.917,3.918,3.919,3.92,3.921,3.922,3.923,3.924,3.925,3.926,3.927,3.928,3.929,3.93,3.931,3.932,3.933,3.934,3.935,3.936,3.937,3.938,3.939,3.94,3.941,3.942,3.943,3.944,3.945,3.946,3.947,3.948,3.949,3.95,3.951,3.952,3.953,3.954,3.955,3.956,3.957,3.958,3.959,3.96,3.961,3.962,3.963,3.964,3.965,3.966,3.967,3.968,3.969,3.97,3.971,3.972,3.973,3.974,3.975,3.976,3.977,3.978,3.979,3.98,3.981,3.982,3.983,3.984,3.985,3.986,3.987,3.988,3.989,3.99,3.991,3.992,3.993,3.994,3.995,3.996,3.997,3.998,3.999,4.0,4.001,4.002,4.003,4.004,4.005,4.006,4.007,4.008,4.009,4.01,4.011,4.012,4.013,4.014,4.015,4.016,4.017,4.018,4.019,4.02,4.021,4.022,4.023,4.024,4.025,4.026,4.027,4.028,4.029,4.03,4.031,4.032,4.033,4.034,4.035,4.036,4.037,4.038,4.039,4.04,4.041,4.042,4.043,4.044,4.045,4.046,4.047,4.048,4.049,4.05,4.051,4.052,4.053,4.054,4.055,4.056,4.057,4.058,4.059,4.06,4.061,4.062,4.063,4.064,4.065,4.066,4.067,4.068,4.069,4.07,4.071,4.072,4.073,4.074,4.075,4.076,4.077,4.078,4.079,4.08,4.081,4.082,4.083,4.084,4.085,4.086,4.087,4.088,4.089,4.09,4.091,4.092,4.093,4.094,4.095,4.096,4.097,4.098,4.099,4.1])
        self.transmission = np.array([0.0,8.22702e-06,1.64531e-05,2.46813e-05,0.000646081,0.000654147,0.000662212,0.0006712269999999999,0.000679292,0.000235512,8.22724e-05,9.04994e-05,9.872610000000001e-05,0.00010695399999999999,0.000195835,0.000454023,0.000470631,0.00047917099999999997,0.000519503,0.0008019189999999999,0.000713579,0.000511359,0.000520374,0.0006902960000000001,0.0007629239999999999,0.000512797,0.000521501,0.00077122,0.00133603,0.00165862,0.0017002,0.00201495,0.00268447,0.00272556,0.00268463,0.00303206,0.00314492,0.00285436,0.00239462,0.00253191,0.00265444,0.00282351,0.0037518,0.00464701,0.00534916,0.00572856,0.00636667,0.00700383,0.00736708,0.00789163,0.00888968,0.00970545,0.0103953,0.0114655,0.012556200000000002,0.013650899999999999,0.015140100000000002,0.0169369,0.019019499999999998,0.020223500000000002,0.0223117,0.024303799999999997,0.0257921,0.027270599999999996,0.029271600000000002,0.031843,0.0345922,0.0372642,0.0414913,0.045260800000000004,0.0500158,0.054199699999999996,0.0592182,0.06850410000000001,0.0738993,0.08013510000000001,0.08658060000000001,0.0954304,0.104423,0.112953,0.123981,0.13560899999999998,0.147335,0.16078499999999998,0.172817,0.18717,0.20251,0.214896,0.236692,0.25433,0.271436,0.293093,0.315684,0.334582,0.359048,0.38026,0.40215700000000004,0.42323900000000003,0.447892,0.474234,0.49449099999999996,0.518241,0.5450550000000001,0.561844,0.586448,0.604304,0.6242,0.644068,0.661459,0.6769609999999999,0.6949930000000001,0.708025,0.719588,0.7275560000000001,0.7398899999999999,0.746206,0.752445,0.758537,0.7638229999999999,0.770683,0.774933,0.777572,0.779382,0.780067,0.780704,0.781276,0.781863,0.782323,0.7831060000000001,0.7841060000000001,0.784922,0.784855,0.7844,0.784665,0.7852020000000001,0.785455,0.7862060000000001,0.787694,0.789553,0.7907,0.791441,0.792149,0.793451,0.794596,0.796053,0.797966,0.8008350000000001,0.803714,0.805778,0.807779,0.810257,0.812563,0.814775,0.817325,0.820321,0.823808,0.8262940000000001,0.828434,0.8296690000000001,0.8303229999999999,0.830526,0.830175,0.82968,0.82921,0.828924,0.828306,0.826967,0.824751,0.820923,0.815624,0.810322,0.7991199999999999,0.790427,0.781856,0.769528,0.754951,0.7408520000000001,0.725751,0.707878,0.691087,0.6650560000000001,0.6423300000000001,0.618088,0.586663,0.560526,0.529471,0.5044029999999999,0.47103599999999995,0.445984,0.41374099999999997,0.385875,0.35694899999999996,0.33704,0.303775,0.281878,0.250161,0.23024,0.20685100000000003,0.194609,0.180334,0.160201,0.14721900000000002,0.134182,0.120948,0.10951199999999998,0.0978404,0.0903572,0.0787825,0.0712705,0.0661916,0.061524699999999995,0.057704700000000005,0.053661099999999996,0.0486983,0.042027499999999995,0.0375035,0.033137,0.030534,0.028166900000000005,0.0263335,0.0244746,0.0229411,0.021452900000000004,0.0197446,0.0175711,0.0159415,0.013932,0.012362200000000002,0.010595700000000001,0.0100413,0.0100433,0.00961809,0.00928778,0.0091184,0.00878849,0.00759862,0.00618652,0.00551741,0.00518687,0.00454858,0.00408884,0.00392063,0.003929,0.00359767,0.00275093,0.00220987,0.00184719,0.00195147,0.00208894,0.00230815,0.00270364,0.00290479,0.00260686,0.00196902,0.00139671,0.000887986,0.000509545,0.000323581,0.000461292,0.00103388,0.00147839,0.00150226,0.00136571,0.00124439,0.000994955,0.00043786,0.0,0.0,0.0,2.68403e-06,4.3632200000000006e-05,8.36878e-05,9.23999e-05,3.31469e-06,0.0,0.0,0.0,0.0,0.0,0.0,2.91175e-05,0.00017450200000000002,0.00020695099999999998,0.00021517599999999997,0.000223402,0.000175155,4.62039e-05,0.0,0.0,0.0,1.45232e-05,0.000159946,0.000305351,0.000450761,0.000483197,0.0004914230000000001,0.00040285900000000004,0.0,0.0,0.0,0.0])
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

class NIRC2JFilter(object):
    def __init__(self):
        import os
        #file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_j.csv')
        file = '/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/filter_curves/nirc2_j.csv'
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = (np.array(f['Transmission']) / np.max(np.array(f['Transmission'])))/ 100
        self.wavelength_unit = u.um
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
        self.label = 'J'

class NIRC2HFilter(object):
    def __init__(self):
        import os
        #file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_h.csv')
        file = '/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/filter_curves/nirc2_h.csv'
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = (np.array(f['Transmission'])  / np.max(np.array(f['Transmission']))) / 100
        self.wavelength_unit = u.um
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
        self.label = 'H'

class NIRC2KFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/nirc2_k.csv')
        f = pd.read_csv(file, comment='#')
        self.wavelength = np.array(f['Wavelength'])
        self.transmission = (np.array(f['Transmission'])  / np.max(np.array(f['Transmission']))) / 100
        self.wavelength_unit = u.um
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
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

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
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

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
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

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
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)
        
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
        self.transmission = np.array(f['transmission']) / np.max(np.array(f['transmission']))
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
        self.transmission = np.array(f['transmission']) / np.max(np.array(f['transmission']))
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
        self.transmission = np.array(f['transmission']) / np.max(np.array(f['transmission']))
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
        self.transmission = np.array(f['transmission'])  / np.max(np.array(f['transmission']))
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
        self.transmission = ( np.array(f['transmission'])  / np.max(np.array(f['transmission']))) / 100
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

class BesselBFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/Bessel_B.txt')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission']) / 100
        self.wavelength_unit = 'nm'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class BesselIFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/Bessel_I.txt')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission']) / 100
        self.wavelength_unit = 'nm'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class BesselRFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/Bessel_R.txt')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission']) / 100
        self.wavelength_unit = 'nm'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class BesselUFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/Bessel_U.txt')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission']) / 100
        self.wavelength_unit = 'nm'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

class BesselVFilter(object):
    def __init__(self):
        import os
        file = os.path.join(os.path.dirname(__file__), 'filter_curves/Bessel_V.txt')
        f = pd.read_table(file, comment='#', delim_whitespace=True, names=['wavelength','transmission'])
        self.wavelength = np.array(f['wavelength'])
        self.transmission = np.array(f['transmission']) / 100
        self.wavelength_unit = 'nm'
        self.central_wavelength = np.round(np.sum(self.wavelength*self.transmission) / np.sum(self.transmission), decimals = 3)
        from myastrotools.tools import GetFWHM, GetEffectiveWidth
        fwhm = GetFWHM(self.wavelength,self.transmission)
        self.half_max = fwhm[3]
        self.fwhm = fwhm[0]
        self.half_max_low = fwhm[1]
        self.half_max_high = fwhm[2]
        self.eff_width = GetEffectiveWidth(self.wavelength, self.transmission)

def Wright2004MainSequencePolynomialFit(BminusV = np.linspace(0,1.75,1000)):
    ''' A functional form polynomial fit to the Hipparcos main sequence from 
        Wright 2004 https://iopscience.iop.org/article/10.1086/423221/pdf
    '''
    a = np.array([0.909, 
                6.258, 
                -23.022, 
                125.5537, 
                -321.1996, 
                485.5234, 
                -452.3198, 
                249.6461, 
                -73.57645, 
                8.8240])
    M = np.array([np.sum([a[i] * (x**(i)) for i in range(0,len(a))]) for x in BminusV])

    return BminusV, M



############################################################################################################
######################################## Reflected Light Funcs #############################################
############################################################################################################

# Assume 0.5 geometric albedo:
def ComputeFluxRatio(Rp, sep, alpha, Ag = 0.5):
    ''' For a single planet compute planet/star flux ratio using Cahoy 2010 eqn 1
    '''
    angleterm = (np.sin(alpha) + (np.pi - alpha)*np.cos(alpha)) / np.pi
    Rp = Rp.to(u.km)
    sep = sep.to(u.km)
    C = Ag * ((Rp / sep)**2) * angleterm
    return C
    

def MakeSeparationContrastPlotOfNearbyRVPlanets(pl, alpha = 90, Ag = 0.5, plot_proxcen = True):
    from myastrotools.tools import ComputeFluxRatio
    C = []   
    for i in range(len(pl)):
        if not np.isnan(pl.loc[i]['pl_rade']):
            R = pl.loc[i]['pl_rade']
        elif not np.isnan(pl['M2R infered radius [Rearth]'][i]):
            R = pl['M2R infered radius [Rearth]'][i]
        else:
            pass
        sep = pl.loc[i]['pl_orbsmax']
        CC = ComputeFluxRatio(R*u.Rearth, sep*u.au, np.radians(90), Ag = 0.5)
        C.append(CC)
    
    C = np.array(C)

    rad = pl['pl_rade'].copy()
    rad.loc[np.where(np.isnan(rad))[0]] = pl['M2R infered radius [Rearth]'][np.where(np.isnan(rad))[0]]

    spt = pl['SpT Number'].copy()
    spt.loc[np.where(np.isnan(spt))[0]] = pl['Mam Inferred SpT from Teff'][np.where(np.isnan(spt))[0]]

    import matplotlib as mpl

    ticks = np.arange(4.0,7.0,0.5)
    ticklabels = ['G0V','G5V','K0V','K5V','M0V','M5V']

    plotx, ploty = pl['Sep lod'],C
    colorby = spt

    #%matplotlib notebook
    fig = plt.figure()

    ax = fig.add_gridspec(top=0.75, right=0.7).subplots()


    norm = mpl.colors.Normalize(vmin=2.2, vmax=np.max(spt))
    a = ax.scatter(plotx, ploty, 
                c=colorby, cmap='hsv_r', s=rad*15, alpha =0.7, norm=norm, edgecolor='black')
    if plot_proxcen:
        ProxCenInd = np.where(pl['hostname'] == 'Proxima Cen')[0][0]
        ax.scatter(pl['Sep lod'][ProxCenInd], C[ProxCenInd],
          edgecolors = 'red', lw=2, color='None', label='Prox Cen b')
        ax.legend(fontsize=10)

    x,y = 12,5e-7
    ax.scatter(x,y, s = 11*15, color='black', alpha = 0.8)
    ax.annotate(r'11 R$_\oplus$', xy = (x,y), xytext = (15,-5), textcoords='offset points',fontsize=20)
    x,y = 12,2.5e-7
    ax.scatter(x,y, s = 5*15, color='black', alpha = 0.8)
    ax.annotate(r'5 R$_\oplus$', xy = (x,y), xytext = (15,-5), textcoords='offset points',fontsize=20)
    x,y = 12,1.25e-7
    ax.scatter(x,y, s = 1*15, color='black', alpha = 0.8)
    ax.annotate(r'1 R$_\oplus$', xy = (x,y), xytext = (15,-5), textcoords='offset points',fontsize=20)

    ax.annotate('N Planets = {}'.format(len(pl)),
                 xy = (0.5,0.81),xycoords='axes fraction', fontsize=20)

    ax.annotate(r' For A$_g$ = '+str(Ag)+r', $\alpha$ = '+str(alpha)+r'$^\circ$, $\lambda$ = 800$\mu$m', 
                xy = (0.05,0.05), xycoords = ('axes fraction'),
               fontsize=15)


    ax.set_yscale('log')
    ax.set_xlim(0,20)
    ax.set_ylim(bottom=3e-10, top=5e-6)
    ax.set_xlabel(r'Max Projected Separation [$\lambda/D$]')
    ax.set_ylabel('Planet/Star Reflected Light Flux Ratio')
    ax.grid(ls=':')


    ax1 = ax.inset_axes([0,1.05,1,0.25])#, sharex=ax)
    ax1.hist(plotx, histtype='step',lw=4, color='red')
    ax1.set_xlim(0,20)
    ax1.set_yticks([0,50,100])
    ax1.set_xticks([])
    ax1.grid(ls=':')

    ax2 = ax.inset_axes([1.05,0,0.25,1])
    ax2.hist(np.log10(ploty), bins=7,
             histtype='step',lw=4, color='red',orientation="horizontal")
    ax2.set_ylim(bottom=-9, top=-5)
    ax2.set_yticks([])
    ax2.grid(ls=':')

    cbarax = ax.inset_axes([1.33,0,0.05,1])
    fig.colorbar(a, cax=cbarax, orientation="vertical", ticks=ticks)
    cbarax.set_yticklabels(ticklabels)
    cbarax.set_ylim(np.min(spt),np.max(spt))
    cbarax.invert_yaxis()
    return fig, C


def MakeInteractiveSeparationContrastPlotOfNearbyRVPlanets(pl, alpha = 45, Ag = 0.5, saveplot = True,
                                                          output_file_name = 'RVPlanetContrastPlot'):
    from myastrotools.tools import ComputeFluxRatio
    C = []   
    for i in range(len(pl)):
        if not np.isnan(pl.loc[i]['pl_rade']):
            R = pl.loc[i]['pl_rade']
        elif not np.isnan(pl['M2R infered radius [Rearth]'][i]):
            R = pl['M2R infered radius [Rearth]'][i]
        else:
            pass
        sep = pl.loc[i]['pl_orbsmax']
        CC = ComputeFluxRatio(R*u.Rearth, sep*u.au, np.radians(alpha), Ag = Ag)
        C.append(CC)
    
    C = np.array(C)

    rad = pl['pl_rade'].copy()
    rad.loc[np.where(np.isnan(rad))[0]] = pl['M2R infered radius [Rearth]'][np.where(np.isnan(rad))[0]]

    spt = pl['SpT Number'].copy()
    spt.loc[np.where(np.isnan(spt))[0]] = pl['Mam Inferred SpT from Teff'][np.where(np.isnan(spt))[0]]
    
    plotx, ploty = np.array(pl['Sep lod']),C
    multiplier = 2
    datadf = pd.DataFrame(data={'plotx':plotx, 'ploty':ploty, 'color':spt, 'markersize':rad*multiplier,
                               'name':pl['pl_name'], 'rad':rad, 'spt':spt
                               })

    from bokeh.plotting import figure, show, output_file, save
    from bokeh.io import output_notebook
    from bokeh.models import LinearColorMapper, ColumnDataSource, LinearInterpolator
    from bokeh.models import  Range1d, LabelSet, Label, ColorBar, FixedTicker
    from bokeh.palettes import Magma256, Turbo256
    from bokeh.transform import linear_cmap
    output_notebook()


    data=ColumnDataSource(data=datadf)


    mapper = linear_cmap(field_name='color', 
                         #palette=Magma256,
                         palette=Turbo256[::-1],
                         low=min(spt), high=max(spt),
                        low_color=Turbo256[::-1][150], high_color=Turbo256[::-1][200])
    tools = "hover, zoom_in, zoom_out, save, undo, redo, pan"
    tooltips = [
        ('Planet', '@name'),
        #("(x,y)", "($x, $y)"),
        ('Cont', '@ploty'),
        ('Sep', '@plotx{0.00}'),
        ('Rad','@rad{0.00}'),
        ('SpT','@spt{0.0}')
    ]

    p = figure(width=1000, height=850, y_axis_type="log", tools=tools, tooltips=tooltips, toolbar_location="above")


    p.circle('plotx','ploty', source=data, fill_alpha=0.6, size='markersize', 
             line_color=mapper, color=mapper)


    color_bar = ColorBar(color_mapper=mapper['transform'], width=15, 
                         location=(0,0), title="Spectral Type",
                        title_text_font_size = '20pt',
                         major_label_text_font_size = '15pt')

    ticks = np.arange(4.0,7.0,0.5)
    color_bar.ticker=FixedTicker(ticks=ticks)
    color_bar.major_label_overrides = {6.5: 'G0V', 6:'G5V', 5.5:'K0V',5:'K5V',4.5:'M0V',4:'M5V'}
    p.add_layout(color_bar, 'right')


    label = Label(
        text=r'$$\mathrm{For}\; A_g = '+str(Ag)+r', \alpha = '+str(alpha)+r'^\circ, \lambda = 800\mu m\$$',
        #text='yes',
        x=50, y=20,
        x_units="screen", y_units="screen",text_font_size = '20pt'
    )
    p.add_layout(label)

    x,y = 16,1.5e-6
    p.circle(x,y, fill_alpha=0.6, size=11*multiplier, 
             color='black')
    label1 = Label(x=x, y=y, text=r'$$11 R_\oplus$$',
                       x_offset=20, y_offset=-20,
                       text_font_size = '20pt')
    p.add_layout(label1)

    x,y = 16,8.5e-7
    p.circle(x,y, fill_alpha=0.6, size=5*multiplier, 
             color='black')
    label2 = Label(x=x, y=y, text=r'$$5 R_\oplus$$',
                      x_offset=20, y_offset=-20,text_font_size = '20pt')
    p.add_layout(label2)
    x,y = 16,5e-7
    p.circle(x,y, fill_alpha=0.6, size=1*multiplier, 
             color='black')
    label3 = Label(x=x, y=y, text=r'$$1 R_\oplus$$',
                      x_offset=20, y_offset=-20,text_font_size = '20pt')
    p.add_layout(label3)

    p.xaxis.axis_label = r'\[ \mathrm{Max\; Projected\; Separation}\; [\lambda/D]\]'
    p.yaxis.axis_label = r'\[ \mathrm{Planet/Star\; Reflected\; Light\; Flux\; Ratio} \]'
    p.xaxis.axis_label_text_font_size = '20pt'
    p.yaxis.axis_label_text_font_size = '20pt'
    p.yaxis.major_label_text_font_size = "15pt"
    p.xaxis.major_label_text_font_size = "15pt"
    if saveplot:
        output_file(output_file_name+".html")
        save(p)
    else:
        show(p)

    return p


def ComputeTeq(StarTeff, StarRad, sep, Ab = 0.3, fprime = 1/4):
    ''' from Seager 2016 Exoplanet Atmospheres eqn 3.9
    https://books.google.com/books?id=XpaYJD7IE20C
    '''
    StarRad = StarRad.to(u.km)
    sep = sep.to(u.km)
    return (StarTeff * np.sqrt(StarRad/sep) * ((fprime * (1 - Ab))**(1/4))).value

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
    from myastrotools.tools import danby_solve, eccentricity_anomaly
    inc = np.radians(inc)
    argp = np.radians(argp)
    EccAnom = danby_solve(eccentricity_anomaly, MeanAnom, ecc, 0.001, maxnum=50)
    TrueAnom = 2*np.arctan( np.sqrt( (1+ecc)/(1-ecc) ) * np.tan(EccAnom/2) )
    Alpha = np.arccos( np.sin(inc) * np.sin(TrueAnom + argp) )
    
    return np.degrees(Alpha)


def alphas(inc, phis):
    '''
    From Lovis+ 2017 sec 2.1:<br>
    $\cos(\alpha) = - \sin(i) \cos(\phi)$<br>
    where $i$ is inclination and $\phi$ is orbital phase with $\phi= 0$ at inferior conjunction

    args:
        inc [flt]: inclination in degrees
        phis [arr]: array of phi values from zero to 360 in degrees

    returns:
        arr: array of viewing phase angles for an orbit from inferior conjunction back to 
            inferior conjunction
    '''
    alphs = []
    for phi in phis:
        alphs.append(np.degrees(np.arccos(-np.sin(np.radians(inc)) * np.cos(np.radians(phi)))))
    return alphs



def GetPhasesFromOrbit(sma,ecc,inc,argp,lon,Ms,Mp):
    ''' Creates an array of viewing phases for an orbit in the plane of the sky to the observer with the maximum phase
     (faintest contrast) at inferior conjunction (where planet is aligned between star and observer) and minimum phase 
     (brightest) at superior conjunction.

    args:
        sma [flt]: semi-major axis in au 
        ecc [flt]: eccentricity
        inc [flt]: inclination in degrees
        argp [flt]: argument of periastron in degrees
        lon [flt]: longitude of ascending node in degrees
        Ms [flt]: star mass in solar masses
        Mp [flt]: planet mass in Jupiter masses

    returns:
        arr: array of viewing phases from periastron back to periastron.

    '''
    # Getting phases for the orbit described by the mean orbital params:
    import astropy.units as u
    from myastrotools.tools import keplerian_to_cartesian, keplersconstant
    # Find the above functions here: https://github.com/logan-pearce/myastrotools/blob/2bbc284ab723d02b7a7189494fd3eabaed434ce1/myastrotools/tools.py#L2593
    # and here: https://github.com/logan-pearce/myastrotools/blob/2bbc284ab723d02b7a7189494fd3eabaed434ce1/myastrotools/tools.py#L239
    # Make lists to hold results:
    xskyplane,yskyplane,zskyplane = [],[],[]
    phase = []
    # How many points to compute:
    Npoints = 1000
    # Make an array of mean anomaly:
    meananom = np.linspace(0,2*np.pi,Npoints)
    # Compute kepler's constant:
    kepmain = keplersconstant(Ms*u.Msun, Mp*u.Mjup)
    # For each orbit point:
    for m in meananom:
        # compute 3d projected position:
        pos, vel, acc = keplerian_to_cartesian(sma*u.au,ecc,inc,argp,lon,m,kepmain)
        # add to list:
        xskyplane.append(pos[0].value)
        yskyplane.append(pos[1].value)
        zskyplane.append(pos[2].value)

    ##### Getting the phases as a function of mean anom: ###########
    ###### Loc of inf conj:
    # Find all points with positive z -> out of sky plane:
    towardsobsvers = np.where(np.array(zskyplane) > 0)[0]
    # mask everything else:
    maskarray = np.ones(Npoints) * 99999
    maskarray[towardsobsvers] = 1
    # mask x position:
    xtowardsobsvers = np.array(xskyplane)*maskarray
    # find where x position is minimized in the section of orbit towards the observer:
    infconj_ind = np.where( np.abs(xtowardsobsvers) == min(np.abs(xtowardsobsvers)) )[0][0]
    ###### Loc of sup conj:
    # Do the opposite - find where x in minimized for points into the plane/away from observer
    awayobsvers = np.where(np.array(zskyplane) < 0)[0]
    maskarray = np.ones(Npoints) * 99999
    maskarray[awayobsvers] = 1
    xawayobsvers = np.array(xskyplane)*maskarray
    supconj_ind = np.where( np.abs(xawayobsvers) == min(np.abs(xawayobsvers)) )[0][0]

    #### Find max and min value phases for this inclination:
    phis = np.linspace(0,180,Npoints)
    phases = np.array(alphas(inc,phis))
    minphase = min(phases)
    maxphase = max(phases)
    # Generate empty phases array:
    phases_array = np.ones(Npoints)

    ###### Set each side of the phases array to range from min to max phase on either side of 
    # inf/sup conjunctions:
    if supconj_ind > infconj_ind:
        # Set one side of the phases array to phases from max to min
        phases_array[0:len(xskyplane[infconj_ind:supconj_ind])] = np.linspace(maxphase,minphase,
                                                            len(xskyplane[infconj_ind:supconj_ind]))
        # # Set the other side to phases from min to max
        phases_array[len(xskyplane[infconj_ind:supconj_ind]):] = np.linspace(minphase,maxphase,
                                                            len(xskyplane)-len(xskyplane[infconj_ind:supconj_ind]))
        # Finally roll the array to align with the mean anomaly array:
        phases_array = np.roll(phases_array,infconj_ind)


    else:
        # Set one side of the phases array to phases from min to max
        phases_array[0:len(xskyplane[supconj_ind:infconj_ind])] = np.linspace(minphase,maxphase,
                                                            len(xskyplane[supconj_ind:infconj_ind]))
        # # Set the other side to phases from max to min
        phases_array[len(xskyplane[supconj_ind:infconj_ind]):] = np.linspace(maxphase,minphase,
                                                            len(xskyplane)-len(xskyplane[supconj_ind:infconj_ind]))
        # Finally roll the array to align with the mean anomaly array:
        phases_array = np.roll(phases_array,supconj_ind)

    return xskyplane, yskyplane, zskyplane, phases_array



def MakeModelPlanet(pdict, sdict, opacity_db,
                calculation = "planet",
                use_guillotpt = True,
                user_supplied_ptprofile = None,
                compute_climate = True,
                cdict = None,
                climate_pbottom = 2,
                climate_ptop = -6, 
                spectrum_wavelength_range = [0.5,1.8],
                spectrum_calculation = 'reflected',
                spectrum_resolution = 150,
                add_clouds = True,
                clouddict = None,
                molecules = None,
                savemodel = False,
                savefilename = None
             ):
    
    ''' Wrapper for PICASO functions for building a planet model
    Args:
        pdict (dict): dictionary of planet parameter inputs
        sdict (dict): dictionary of star parameter inputs
        opacity_db (jdi.opannection object)
        calculation (str): picaso input for object, "planet" or "brown dwarf"
        use_guillotpt (bool): if True, use Guillot PT approximation. Else user must supply initial PT profile
        user_supplied_ptprofile (df): user supplied pt profile for picaso
        comput_climate (bool): if true use picaso to compute plnet climate
        cdict (dict): dictionary of climate run setup params
        climate_pbottom (flt): log(pressure) at bottom of climate calc
        climate_ptop (flt): log(pressure) at top of climate calc
        spectrum_wavelength_range (list): range in um of wavelengths to compute spectrum
        spectrum_calculation (str): type of spectrum to calculate
        spectrum_resolution (flt): what R to compute the spectrum
        add_clouds (bool): If true, add clouds to model
        clouddict (dict): dictionary of cloud parameters
        molecules (list): list of molecules to compute cloud properties. If None, use virga recommended mols
        savemodel (bool): if true, save the model using the xarray method in picaso
        savefilename (str): filename and path for the model to be saved.
    Returns:
        pl: picaso planet model inputs
        noclouds: picaso object after climate run before clouds
        w_noclouds, f_noclouds: wavelength and flux arrays for noclouds spectrum sampled at spectrum_resolution
        clouds_added: virga output from adding clouds
        clouds_spectrum: spectrum after adding clouds computed from spectrum_calculation
        w_clouds, f_clouds: cloudy spectrum sampled at spectrum_resolution

    '''
    import warnings
    warnings.filterwarnings('ignore')
    import picaso.justdoit as jdi

    add_output={
            'author':"Logan Pearce",
            'contact' : "loganpearce1@arizona.edu",
            'code' : "picaso, virga",
            'planet_params':pdict,
            'stellar_params':sdict,
            'orbit_params':{'sma':pdict['semi_major']}
            }
    
    # initialize model:
    pl = jdi.inputs(calculation= calculation, climate = compute_climate) # start a calculation
    # set up planet:
    pl.effective_temp(pdict['tint']) # input effective temperature
    # add gravity:
    if not pdict['gravity']:
        pl.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        pl.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
        
    # set up star:
    pl.star(opacity_db, temp = sdict['Teff'], metal = sdict['mh'], logg = sdict['logg'], 
            radius = sdict['radius'], radius_unit = u.R_sun, 
            semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database = 'phoenix')
    
    # climate run
    if use_guillotpt:
        pt = pl.guillot_pt(pdict['Teq'], nlevel=cdict['nlevel'], T_int = pdict['tint'], 
                              p_bottom=climate_pbottom, p_top=climate_ptop)
    else:
        pt = user_supplied_ptprofile

    if compute_climate:
        # initial PT profile guess:
        temp_guess = pt['temperature'].values 
        press_guess = pt['pressure'].values
        # Input climate params:
        nstr = np.array([0,cdict['nstr_upper'],cdict['nstr_deep'],0,0,0]) # initial guess of convective zones
        pl.inputs_climate(temp_guess= temp_guess, pressure= press_guess, 
                      nstr = nstr, nofczns = cdict['nofczns'] , rfacv = cdict['rfacv'])
        print('starting climate run')
        # Compute climate:
        noclouds = pl.climate(opacity_db, save_all_profiles=True, with_spec=True)
    else:
        noclouds = pl.copy()
        
    # make a new object for computing the new spectrum:
    opa_mon = jdi.opannection(wave_range=spectrum_wavelength_range)
    noclouds_spec = jdi.inputs(calculation="planet") # start a calculation
    noclouds_spec.phase_angle(0)
    # add gravity:
    if not pdict['gravity']:
        noclouds_spec.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        noclouds_spec.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
        # add same star:
    noclouds_spec.star(opa_mon, temp = sdict['Teff'], metal = sdict['mh'], logg = sdict['logg'], 
        radius = sdict['radius'], radius_unit=u.R_sun, 
        semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'])
    # add new atmosphere computer by climate run:
    noclouds_spec.atmosphere(df=noclouds['ptchem_df'])
    # compute spectrum:
    noclouds_spec_spectrum = noclouds_spec.spectrum(opa_mon, 
                                                    calculation=spectrum_calculation, 
                                                    full_output=True)
    w_noclouds, f_noclouds = jdi.mean_regrid(noclouds_spec_spectrum['wavenumber'],
                          noclouds_spec_spectrum['fpfs_reflected'], R=spectrum_resolution)
    if not add_clouds:
        # if no clouds, save model and finish computation:
        if savemodel:
            preserve = jdi.output_xarray(
                noclouds_spec_spectrum,
                noclouds_spec,
                add_output = add_output,
                savefile=savefilename
                )
        return pl, noclouds, noclouds_spec_spectrum, w_noclouds, f_noclouds
    else:
        # else add clouds:
        print('adding clouds')
        from virga import justdoit as vj
        # pressure temp profile from climate run:
        temperature = noclouds['temperature']
        pressure = noclouds['pressure']
        #metallicity = pdict['mh'] #atmospheric metallicity relative to Solar
        metallicity_TEMP = 0
        # got molecules for cloud species:
        if not molecules:
            # if no user-supplied molecules:
            #get virga recommendation for which gases to run
            # metallicitiy must be in NOT log units
            recommended = vj.recommend_gas(pressure, temperature, 10**(metallicity_TEMP), clouddict['mean_mol_weight'],
                            #Turn on plotting
                             plot=False)
            mols = recommended
            print('virga recommended gas species:',recommended)
        else:
            # otherwise use user supplied mols:
            mols = molecules
        # add atmosphere from climate run:
        atm = noclouds['ptchem_df']
        # add kzz:
        atm['kz'] = [clouddict['kz']]*atm.shape[0]
        
        # Just set all this up again:
        clouds = jdi.inputs(calculation=calculation) # start a calculation
        clouds.phase_angle(0)
        # add gravity:
        if not pdict['gravity']:
            clouds.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
                mass = pdict['mass'], mass_unit=pdict['mass_unit'])
        else:
            clouds.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
        # add star:
        clouds.star(opa_mon, temp = sdict['Teff'], metal = sdict['mh'], logg = sdict['logg'], 
            radius = sdict['radius'], radius_unit=u.R_sun, 
            semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database = 'phoenix')
        # add atmosphere from climate run with kzz:
        clouds.atmosphere(df=atm)
        # get clouds from reference:
        directory ='/Volumes/Oy/virga/virga/reference/RefIndexFiles'
        clouds_added = clouds.virga(mols,directory, fsed=clouddict['fsed'], mh=10**(metallicity_TEMP),
                 mmw = clouddict['mean_mol_weight'], full_output=True)
        # compute spectrum:
        clouds_spectrum = clouds.spectrum(opa_mon, 
                                                calculation=spectrum_calculation, 
                                                full_output=True)
        w_clouds,f_clouds = jdi.mean_regrid(clouds_spectrum['wavenumber'],
                      clouds_spectrum['fpfs_reflected'], R=spectrum_resolution)
        # save:
        if savemodel:
            preserve = jdi.output_xarray(
                clouds_spectrum,
                clouds,
                add_output = add_output,
                savefile=savefilename)
            import pickle
            pickle.dump([pl, noclouds, w_noclouds, f_noclouds, clouds, clouds_added, mols, clouds_spectrum, w_clouds, f_clouds],
                        open(savefilename.replace('.nc','.pkl'),'wb'))
            
        
        return pl, noclouds, w_noclouds, f_noclouds, clouds, clouds_added, mols, clouds_spectrum, w_clouds, f_clouds


##### Broken into modules:
def MakeModelCloudFreePlanet(pdict, sdict,
                calculation = "planet",
                use_guillotpt = True,
                user_supplied_ptprofile = None,
                cdict = None,
                climate_pbottom = 2,
                climate_ptop = -6,
                savemodel = False,
                savefiledirectory = None
             ):
    
    ''' Wrapper for PICASO functions for building a planet model
    Args:
        pdict (dict): dictionary of planet parameter inputs
        sdict (dict): dictionary of star parameter inputs
        opacity_db (jdi.opannection object)
        calculation (str): picaso input for object, "planet" or "brown dwarf"
        use_guillotpt (bool): if True, use Guillot PT approximation. Else user must supply initial PT profile
        user_supplied_ptprofile (df): user supplied pt profile for picaso
        cdict (dict): dictionary of climate run setup params
        climate_pbottom (flt): log(pressure) at bottom of climate calc
        climate_ptop (flt): log(pressure) at top of climate calc
        molecules (list): list of molecules to compute cloud properties. If None, use virga recommended mols
        savemodel (bool): if true, save the model using the xarray method in picaso
        savefilename (str): filename and path for the model to be saved.
    Returns:
        pl: picaso planet model inputs
        noclouds: picaso object after climate run before clouds
    '''
    import warnings
    warnings.filterwarnings('ignore')
    import picaso.justdoit as jdi
    
    import sys
    import os
    os.system('mkdir '+savefiledirectory)
    

    f = open(savefiledirectory+"/terminal_output.txt", 'w')
    sys.stdout = f

    add_output={
            'author':"Logan Pearce",
            'contact' : "loganpearce1@arizona.edu",
            'code' : "picaso, virga",
            'planet_params':pdict,
            'stellar_params':sdict,
            'orbit_params':{'sma':pdict['semi_major']}
            }
    
    # retrieve opacity correlated k-tables database:
    PlanetMH = pdict['mh']
    PlanetCO = pdict['CtoO']
    ck_db = f'/Volumes/Oy/picaso/reference/kcoeff_2020/sonora_2020_feh{PlanetMH}_co_{PlanetCO}.data.196'
    opacity_ck = jdi.opannection(ck_db=ck_db)
    
    # initialize model:
    pl = jdi.inputs(calculation= calculation, climate = True)
    
    # set up planet:
    # input effective temperature
    pl.effective_temp(pdict['tint']) 
    # add gravity:
    if not pdict['gravity']:
        pl.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        pl.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
        
    # set up star:
    pl.star(opacity_ck, temp = sdict['Teff'], metal = sdict['mh'], logg = sdict['logg'], 
            radius = sdict['radius'], radius_unit = u.R_sun, 
            semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database = 'phoenix')
    
    # climate run
    if use_guillotpt:
        pt = pl.guillot_pt(pdict['Teq'], nlevel=cdict['nlevel'], T_int = pdict['tint'], 
                              p_bottom=climate_pbottom, p_top=climate_ptop)
    else:
        pt = user_supplied_ptprofile

    # initial PT profile guess:
    temp_guess = pt['temperature'].values 
    press_guess = pt['pressure'].values
    # Input climate params:
    nstr = np.array([0,cdict['nstr_upper'],cdict['nstr_deep'],0,0,0]) # initial guess of convective zones
    pl.inputs_climate(temp_guess= temp_guess, pressure= press_guess, 
                  nstr = nstr, nofczns = cdict['nofczns'] , rfacv = cdict['rfacv'])
    print('starting climate run')
    # Compute climate:
    noclouds = pl.climate(opacity_ck, save_all_profiles=True, with_spec=True)

    from virga import justdoit as vj
    # pressure temp profile from climate run:
    temperature = noclouds['temperature']
    pressure = noclouds['pressure']
    #metallicity = pdict['mh'] #atmospheric metallicity relative to Solar
    metallicity_TEMP = 0
    mmw = 2.2
    # got molecules for cloud species:
    recommended, fig = vj.recommend_gas(pressure, temperature, 10**(metallicity_TEMP), 
                                   mmw, plot=True, outputplot = True)
    
    
    from bokeh.plotting import figure, output_file, save
    output_file(savefiledirectory+"/recomended-gasses.html")
    save(fig)
    
    import pickle
    pickle.dump([pl, noclouds], open(savefiledirectory+'/cloud-free-model.pkl','wb'))
    pickle.dump([pdict, sdict, cdict], open(savefiledirectory+'/cloud-free-model-inputs.pkl','wb'))
    
    f.close()
    
    
    return pl, noclouds
    

def MakeModelCloudyPlanet(savefiledirectory, clouddict,
                          calculation = 'planet', 
                         molecules = None):
    
    ''' Wrapper for PICASO functions for building a planet model
    Args:
        savefiledirectory (str): directory containing picaso cloud-free model base case.
        clouddict (dict): dictionary of cloud parameter inputs
        calculation (str): picaso input for object, "planet" or "brown dwarf"
        molecules (list): list of molecules to compute cloud properties. If None, use virga recommended mols
    Returns:
        clouds: picaso planet model inputs
        clouds_added: virga cloud run output clouds
    '''
    # import climate run output:
    import pickle
    import picaso.justdoit as jdi
    
    import sys
    f = open(savefiledirectory+"/terminal_output.txt", 'a')
    sys.stdout = f
    print()
    
    pl, noclouds = pickle.load(open(savefiledirectory+'/cloud-free-model.pkl','rb'))
    pdict, sdict, cdict = pickle.load(open(savefiledirectory+'/cloud-free-model-inputs.pkl','rb'))
    
    opa_mon = jdi.opannection()

    from virga import justdoit as vj
    # pressure temp profile from climate run:
    temperature = noclouds['temperature']
    pressure = noclouds['pressure']
    #metallicity = pdict['mh'] #atmospheric metallicity relative to Solar
    metallicity_TEMP = 0
    # got molecules for cloud species:
    if not molecules:
        # if no user-supplied molecules:
        #get virga recommendation for which gases to run
        # metallicitiy must be in NOT log units
        recommended = vj.recommend_gas(pressure, temperature, 10**(metallicity_TEMP), 
                                       clouddict['mean_mol_weight'],
                        #Turn on plotting
                         plot=False)
        mols = recommended
        print('using virga recommended gas species:',recommended)
    else:
        # otherwise use user supplied mols:
        mols = molecules
    # add atmosphere from climate run:
    atm = noclouds['ptchem_df']
    # add kzz:
    atm['kz'] = [clouddict['kz']]*atm.shape[0]

    # Just set all this up again:
    clouds = jdi.inputs(calculation=calculation) # start a calculation
    clouds.phase_angle(0)
    # add gravity:
    if not pdict['gravity']:
        clouds.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        clouds.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
    # add star:
    clouds.star(opa_mon, temp = sdict['Teff'], metal = sdict['mh'], logg = sdict['logg'], 
        radius = sdict['radius'], radius_unit=u.R_sun, 
        semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'])
    # add atmosphere from climate run with kzz:
    clouds.atmosphere(df=atm)
    # get clouds from reference:
    directory ='/Volumes/Oy/virga/virga/reference/RefIndexFiles'
    
    clouds_added = clouds.virga(mols,directory, fsed=clouddict['fsed'], mh=clouddict['mh'],
                         mmw = clouddict['mean_mol_weight'], full_output=True)

    clouddict.update({'condensates':mols})

    pickle.dump(clouds,
                    open(savefiledirectory+'/cloudy-model.pkl','wb'))
    pickle.dump([pdict, sdict, cdict, clouddict],open(savefiledirectory+'/cloudy-model-inputs.pkl','wb'))
    
    f.close()

    return clouds, clouds_added


def MakeModelCloudyAndCloudFreeSpectra(savefiledirectory,
                            spectrum_wavelength_range = [0.5,1.8],
                            spectrum_calculation = 'reflected',
                            spectrum_resolution = 150,
                            calculation = "planet",
                            plot_albedo = False
                                      ):
    
    ''' Wrapper for PICASO functions for building a planet model
    Args:
        savefiledirectory (str): directory containing picaso cloud-free model base case.
        spectrum_wavelength_range (list): range in um of wavelengths to compute spectrum
        spectrum_calculation (str): type of spectrum to calculate
        spectrum_resolution (flt): what R to compute the spectrum
        calculation (str): picaso input for object, "planet" or "brown dwarf"
        plot_albedo (bool): if True, return spectrum in albedo, otherwise return planet/star flux ratio
    Returns:
        clouds: picaso planet model inputs
        clouds_added: virga cloud run output clouds
    '''
    import pickle
    import picaso.justdoit as jdi
    import matplotlib.pyplot as plt

    opa_mon = jdi.opannection(wave_range=spectrum_wavelength_range)
    
    ### Cloud-free spectrum:
    pl, noclouds = pickle.load(open(savefiledirectory+'/cloud-free-model.pkl','rb'))
    pdict, sdict, cdict = pickle.load(open(savefiledirectory+'/cloud-free-model-inputs.pkl','rb'))
    noclouds_spec = jdi.inputs(calculation="planet") # start a calculation
    noclouds_spec.phase_angle(0)
    # add gravity:
    if not pdict['gravity']:
        noclouds_spec.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        noclouds_spec.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
        # add same star:
    noclouds_spec.star(opa_mon, temp = sdict['Teff'], metal = sdict['mh'], logg = sdict['logg'], 
        radius = sdict['radius'], radius_unit=u.R_sun, 
        semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'])
    # add new atmosphere computer by climate run:
    noclouds_spec.atmosphere(df=noclouds['ptchem_df'])
    # compute spectrum:
    noclouds_spec_spectrum = noclouds_spec.spectrum(opa_mon, 
                                                    calculation=spectrum_calculation, 
                                                    full_output=True)
    if plot_albedo:
        w_noclouds, f_noclouds = jdi.mean_regrid(noclouds_spec_spectrum['wavenumber'],
                              noclouds_spec_spectrum['albedo'], R=spectrum_resolution)
    else:
        w_noclouds, f_noclouds = jdi.mean_regrid(noclouds_spec_spectrum['wavenumber'],
                          noclouds_spec_spectrum['fpfs_reflected'], R=spectrum_resolution)
    
    
    ### Cloud-y spectrum:
    pdict, sdict, cdict, clouddict = pickle.load(open(savefiledirectory+'/cloudy-model-inputs.pkl','rb'))
    clouds_spec = pickle.load(open(savefiledirectory+'/cloudy-model.pkl','rb'))
    clouds_spec_spectrum = clouds_spec.spectrum(opa_mon, 
                    calculation='reflected', 
                    full_output=True)
    if plot_albedo:
        w_clouds, f_clouds = jdi.mean_regrid(clouds_spec_spectrum['wavenumber'],
                              clouds_spec_spectrum['albedo'], R=spectrum_resolution)
    else:
        w_clouds, f_clouds = jdi.mean_regrid(clouds_spec_spectrum['wavenumber'],
                          clouds_spec_spectrum['fpfs_reflected'], R=spectrum_resolution)
        
    pickle.dump([noclouds_spec_spectrum,1e4/w_noclouds, f_noclouds],
               open(savefiledirectory+'/cloud-free-spectrum-R'+str(spectrum_resolution)+'.pkl','wb'))
    pickle.dump([clouds_spec_spectrum,1e4/w_clouds, f_clouds],
               open(savefiledirectory+'/cloudy-spectrum-R'+str(spectrum_resolution)+'.pkl','wb'))
    
    # make plot:
    fig = plt.figure()
    plt.plot(1e4/w_clouds, f_clouds, color='black', label='Cloudy')
    plt.plot(1e4/w_noclouds, f_noclouds, color='darkcyan', label='Cloud-Free')
    plt.minorticks_on()
    plt.tick_params(axis='both',which='major',length =10, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =5, width=2,direction='in',labelsize=23)
    plt.xlabel(r"Wavelength [$\mu$m]", fontsize=25)
    plt.ylabel('Planet:Star Contrast', fontsize=25)
    plt.gca().set_yscale('log')
    plt.grid(ls=':')
    plt.legend(fontsize=15, loc='lower left')
    plt.tight_layout()
    plt.savefig('reflected-spectrum-plot.png', bbox_inches='tight')
    
    return noclouds_spec_spectrum, 1e4/w_noclouds, f_noclouds, clouds_spec_spectrum, 1e4/w_clouds, f_clouds, fig






    