
'''
targets_into_TCScat.py

Supply a list of Simbad-resolvable target names and produce a catalog for LCO TCS.  
Supply names as a list of single-quote strings encompassed by double quote strings using
the flag --names.

To run from command line:
python targets_into_TCScat.py --names="'alf Sco', 'HD 214810A', 'HD 218434'"

Dependencies:
astropy, astroquery, numpy, pandas

Written by Logan Pearce, 2022
https://github.com/logan-pearce; http://www.loganpearcescience.com
'''

from astroquery.simbad import Simbad
import astropy.units as u
import numpy as np
import pandas as pd
import argparse
# Pandas throws a warning when adding a single element to a table, and we can ignore it:
import warnings
warnings.filterwarnings('ignore')


parser = argparse.ArgumentParser()
parser.add_argument('--names', type=str, required=True, help='Supply the Simbad names')
parser.add_argument('--saveas', type=str, help='Filename for catalog. Default = TCScatalog.cat')

args = parser.parse_args()
Names = args.names.split(',')
Names = [Names[i].replace("'","") for i in range(len(Names))]
Names = [Names[i].replace("*","") for i in range(len(Names))]

if args.saveas:
    saveas = args.saveas
else:
    saveas = 'TCScatalog.cat'


pdcat = pd.DataFrame(data={'Name':Names[0]}, index=[0])
for i in range(1,len(Names)):
    data = {'Name':Names[i]}
    pdcat = pdcat.append(data,ignore_index=True)
    

pdcat['RA'], pdcat['DEC'], pdcat['pmra'], pdcat['pmdec'] = np.nan,np.nan,np.nan,np.nan
customSimbad = Simbad()
customSimbad.add_votable_fields('pmra','pmdec')


pdcat['pmra s/yr'], pdcat['pmdec arcsec/yr'] = np.nan, np.nan

from astropy.coordinates import Angle
for i in range(len(pdcat)):
    r = customSimbad.query_object(pdcat['Name'][i])
    pdcat['RA'].loc[i], pdcat['DEC'].loc[i] = r['RA'][0],r['DEC'][0]
    pdcat['pmra'].loc[i], pdcat['pmdec'].loc[i] = r['PMRA'][0],r['PMDEC'][0]
    pdcat['RA'].loc[i], pdcat['DEC'].loc[i] = pdcat['RA'].loc[i].replace(' ',':'), \
        pdcat['DEC'].loc[i].replace(' ',':')
    
    
    # Create an astropy angle object:
    a = Angle(pdcat['pmra'][i],u.mas)
    # Convert to hms:
    a2 = a.hms
    # add up the seconds (a2[0] and a2[1] are most likely 0 but just in case):
    a3 = a2[0]*u.hr.to(u.s) + a2[1]*u.min.to(u.s) + a2[2]
    # put into table:
    pdcat['pmra s/yr'][i] = a3
    
    # Dec is easier:
    a = pdcat['pmdec'][i]*u.mas.to(u.arcsec)
    # put into table:
    pdcat['pmdec arcsec/yr'][i] = a
    
for i in range(len(pdcat)):
    pdcat['Name'][i] = pdcat['Name'][i].replace(' ','')
pdcat['num'] = np.arange(1,len(pdcat)+1,1)

pdcat_out = pdcat[['num']]
pdcat_out['Name'] = pdcat['Name']
pdcat_out['RA'] = pdcat['RA']
pdcat_out['Dec'] = pdcat['DEC']
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
pdcat_out.to_csv(saveas, index=False, sep='\t')
