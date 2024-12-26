
'''
get_brightstars_MagAO-X.py

A script for generating a list of bright stars near zenith for AO wavefront sensor engineering.  Queries
Simbad catalog for bright sources near the meridian and elevation > 60 deg throughout the night for 
the supplied date from the supplied location.  Returns targets in the catalog format and units that 
the LCO TOs need.  Saves results to a .cat file for the TCS called Bright_AO_stars_cat_'+date+'.cat, 
and a .csv of all the information for each source called Bright_AO_stars_cat_complete_'+date+'.csv

Dependencies:
astropy, astroplan, astroquery, numpy, pandas
Requirements:

Written by Logan Pearce, 2022
https://github.com/logan-pearce; http://www.loganpearcescience.com
'''

from astropy.coordinates import EarthLocation, SkyCoord, AltAz, Angle
from astropy.time import Time
from astroplan import Observer
from astroquery.gaia import Gaia
from astroquery.simbad import Simbad
from astropy.table import Table, vstack
import astropy.units as u
import numpy as np
import pandas as pd
import argparse
# Pandas throws a warning when adding a single element to a table, and we can ignore it:
import warnings
warnings.filterwarnings('ignore')

from astropy.coordinates import EarthLocation, SkyCoord, AltAz
from astropy.time import Time
from astroplan import Observer

# LCO is at UTC-4:
utc_offset = 0*u.hour
# Create earth location object for LCO:
location = EarthLocation.of_site('Las Campanas Observatory')
# Set a date for the observation:
time1 = Time('2022-04-09 00:00:00',scale='utc')
time2 = Time('2022-04-10 00:00:00',scale='utc')
lco = Observer.at_site("Las Campanas Observatory")

# Generate an array of times in half hour increments:
times_in_halfhour_increments = np.arange(time1.decimalyear,
                                         time2.decimalyear,
                                         30*u.min.to(u.yr))

# Compute the LST for each of those times:
lsts_in_halfhour_increments = [Time(t,scale='utc',format='decimalyear').sidereal_time('mean', longitude=location)
                               for t in times_in_halfhour_increments]
lsts_in_ICRS = [lsts_in_halfhour_increments[i].deg for i in range(len(lsts_in_halfhour_increments))]

from astroquery.simbad import Simbad

customSimbad = Simbad()
customSimbad.add_votable_fields('otype','sptype','flux(V)','flux(R)','flux(I)','pmra','pmdec')
cat = customSimbad.query_criteria('dec < 0 & dec > -50 & Imag < 7',otype='Star')

import warnings
warnings.filterwarnings('ignore')
pdcat = cat.to_pandas()
pdcat['RA deg'],pdcat['Dec deg'] = np.nan,np.nan
pdcat['pmra s/yr'], pdcat['pmdec arcsec/yr'] = np.nan, np.nan

for i in range(len(pdcat)):
    # Make a sky coord object:
    ob = SkyCoord(ra = pdcat['RA'][i], dec = pdcat['DEC'][i], frame="icrs", unit=(u.hourangle,u.degree))
    pdcat['RA deg'][i], pdcat['Dec deg'][i] = ob.ra.deg,ob.dec.deg
    # convert to string in hms and dms, and split the string in to [ra,dec]
    pdcat['RA'][i], pdcat['DEC'][i] = pdcat['RA'][i].replace(' ',':'), pdcat['DEC'][i].replace(' ',':')
    pdcat['MAIN_ID'][i] = pdcat['MAIN_ID'][i].replace(' ','')
    pdcat['MAIN_ID'][i] = pdcat['MAIN_ID'][i].replace('*','')
    
from astropy.coordinates import Angle
## convert pmra in mas/yr into s/yr and pmdec in mas/yr to arcsec/yr:
# For each object:
for i in range(len(pdcat)):
    # Create an astropy angl object:
    a = Angle(pdcat['PMRA'][i],u.mas)
    # Convert to hms:
    a2 = a.hms
    # add up the seconds (a2[0] and a2[1] are most likely 0 but just in case):
    a3 = a2[0]*u.hr.to(u.s) + a2[1]*u.min.to(u.s) + a2[2]
    # put into table:
    pdcat['pmra s/yr'][i] = a3
    
    # Dec is easier:
    a = pdcat['PMDEC'][i]*u.mas.to(u.arcsec)
    # put into table:
    pdcat['pmdec arcsec/yr'][i] = a
    
ind = np.argsort(pdcat['RA deg'])
pdcat = pdcat.loc[ind]
pdcat = pdcat.reset_index(drop=True)

pdcat['num'] = np.arange(1,len(pdcat)+1,1)
pdcat['Name'] = pdcat['MAIN_ID']

pdcat_out = pdcat[['num','Name','RA','DEC']]
pdcat_out['Equinox'] = 2000.0
pdcat_out['pmra'] = pdcat['pmra s/yr']
pdcat_out['pmdec'] = pdcat['pmdec arcsec/yr'] 
pdcat_out['rotang'] = 0
pdcat_out['rot_mode'] = 'GRV'
pdcat_out['RA_probe1'],pdcat_out['Dec_probe1'] = '00:00:00.00',  '+00:00:00.0'
pdcat_out['equinox'] = 2000.0
pdcat_out['RA_probe2'],pdcat_out['Dec_probe2'] = '00:00:00.00',  '+00:00:00.0'
pdcat_out['equinox '] = 2000.0
pdcat_out['epoch'] = 2000.0
pdcat_out

pdcat_out.to_csv('Bright_AO_stars_cat.cat', index=False, sep='\t')

pdcat_out2 = pdcat[['num','Name','RA','DEC']]
pdcat_out2['pmra'] = pdcat['pmra s/yr']
pdcat_out2['pmdec'] = pdcat['pmdec arcsec/yr'] 
pdcat_out2['V mag'],pdcat_out2['R mag'],pdcat_out2['I mag'] = pdcat['FLUX_V'],pdcat['FLUX_R'],pdcat['FLUX_I']
pdcat_out2['otype'],pdcat_out2['spt'] = pdcat['OTYPE'],pdcat['SP_TYPE']

pdcat_out2.to_csv('Bright_AO_stars_cat.csv', index=False)