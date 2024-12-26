
'''
get_brightstars_MagAO-X.py

A script for generating a list of bright stars near zenith for AO wavefront sensor engineering.  Queries
the Gaia catalog for bright sources near the meridian and elevation > 60 deg throughout the night for 
the supplied date from the supplied location, queries Simbad for the star's name and spectral type, 
and computes the color correction for converting from Gaia G magnitudes to magnitudes in the two 
MagAO-X wavefront sensor bands. Returns around ~100 targets in the catalog format and units that 
the LCO TOs need.  Saves results to a .cat file for the TCS called Bright_AO_stars_cat_'+date+'.cat, 
and a .csv of all the information for each source called Bright_AO_stars_cat_complete_'+date+'.csv

Dependencies:
astropy, astroplan, astroquery, numpy, pandas
Requirements:
GaiaG_WFS_color_conversion.csv: https://github.com/logan-pearce/myastrotools/blob/master/myastrotools/GaiaG_WFS_color_conversion.csv

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

parser = argparse.ArgumentParser()
parser.add_argument('--date', type=str, required=True, help='Supply the observation date\
    for the start of the night in YYYY-MM-DD')
parser.add_argument('--location', type=str, help='Optional flag; \
    Supply the name of the observatory. Default is Las Campanas Observatory')
parser.add_argument('--utc_offset', type=float, help='Optional flag; \
    If not at LCO, supply to offset of local time from UTC.')
parser.add_argument('--limiting_mag', type=float, help='Optional flag; Select objects in Gaia\
    catalog brighter than this magnitude in Gaia G band.  Default = 6.5')
parser.add_argument('--catalog',type=str, help='Optional flag: supply the Gaia catalog to pull sources in\
    format gaiaXXX.gaia_source.  Default = gaiaedr3.gaia_source')
parser.add_argument('--epoch',type=str, help='Optional flag: If not using Gaia EDR3, supply the \
    observation epoch for the catalog')

args = parser.parse_args()

if args.location:
    loc = args.location
else:
    loc = 'Las Campanas Observatory'

######### Determine range of LSTs throughout the night:
#if args.utc_offset:
##    utc_offset = args.utc_offset*u.hour
#else:
    # LCO is at UTC-4:
#    utc_offset = 4*u.hour
# Create earth location object for LCO:
location = EarthLocation.of_site(loc)
# Set a date for the observation:
time = Time(args.date+' 00:00:00',scale='utc')
time.sidereal_time('mean', longitude=location)
# Create astroplan observer object:
lco = Observer.at_site(loc)
# Get sunrise and sunset times for that date:
sunset = Time(lco.sun_set_time(time).iso,scale='utc')#+utc_offset
sunrise = Time(lco.sun_rise_time(time).iso,scale='utc')#+utc_offset
# Get Local Sideral Time for sunset and sunrise:
lst_at_sunset = sunset.sidereal_time('mean', longitude=location)
lst_at_sunrise = sunrise.sidereal_time('mean', longitude=location)

# Generate an array of times in half hour increments:
times_in_halfhour_increments = np.arange(sunset.decimalyear-30*u.min.to(u.yr),
                                         sunrise.decimalyear+30*u.min.to(u.yr),
                                         30*u.min.to(u.yr))
# Compute the LST for each of those times:
lsts_in_halfhour_increments = [Time(t,scale='utc',format='decimalyear').sidereal_time('mean', longitude=location)
                               for t in times_in_halfhour_increments]
lsts_in_ICRS = [lsts_in_halfhour_increments[i].deg for i in range(len(lsts_in_halfhour_increments))]

########## Get Gaia targets around LSTs throughout the night:

def update_progress(n,max_value):
    ''' Create a progress bar
    
    Args:
        n (int): current count
        max_value (int): ultimate values
    
    '''
    import sys
    barLength = 20 # Modify this to change the length of the progress bar
    status = ""
    progress = np.round(np.float(n/max_value),decimals=2)
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

if args.limiting_mag:
    limmag = args.limiting_mag
else:
    limmag = 6.5

if args.catalog:
    catalog = args.catalog
else:
    catalog = 'gaiaedr3.gaia_source'

def get_gaia_objects(i, n=5):
    # For each LST range in the array of LSTs:
    # Construct a SQL search string to retrieve Gaia objects within +/- 30 mins of each LST,
    # that are brighter than mag 8 and with good quality solutions:
    search_string = 'SELECT source_id, ra, dec, pmra, pmdec, phot_g_mean_mag, ruwe \
    FROM ' + catalog + ' \
    WHERE ra > '+ str(lsts_in_ICRS[i-1]) + 'AND ra < '+ str(lsts_in_ICRS[i]) +' \
    AND phot_g_mean_mag<=' + str(limmag) +' \
    AND ruwe <1.2 \
    ORDER BY ra ASC'
    # Retrive results from catalog:
    job = Gaia.launch_job(search_string)
    j = job.get_results()
    # Add a column to table for altitudes:
    j['alt'] = np.nan
    # Construct an Alt/Az reference frame at LCO at the given timeL:
    altazframe = AltAz(obstime=Time(times_in_halfhour_increments[i],scale='utc',format='decimalyear'),
                   location=location)
    # For each solution:
    for jj in range(len(j)):
        # Make a SkyCoord object:
        ob = SkyCoord(ra = j[jj]['ra'], dec = j[jj]['dec'], frame="icrs", unit="deg")
        # Transform the object to the AltAz frame:
        obaltazs = ob.transform_to(altazframe)
        # Put altitude into table:
        j['alt'][jj] = obaltazs.alt.value
    # Select only objects with alt greater than 60 deg:
    j2 = j[np.where(j['alt']>60)]
    # Sort by mag:
    ind = np.argsort(j2['phot_g_mean_mag'])
    j3 = j2[ind]
    # select only brighter objects:
    j4 = j3[np.where(j3['phot_g_mean_mag']<6.5)]
    if len(j4) > n:
        # Randomly select n objects:
        rand = np.random.randint(0,len(j4),n)
        j4 = j4[rand]
    return j4

print('Retrieving Gaia objects...')
# get objects in Gaia within LST bins brighter than limiting magnitude:
cat = get_gaia_objects(1, n=25)
for i in range(2,len(lsts_in_halfhour_increments)):
    cat2 = get_gaia_objects(i)
    cat = vstack([cat, cat2])
    update_progress(i,len(lsts_in_halfhour_increments)-1)

# sort in order of ascending ra:
ind = np.argsort(cat['ra'])
cat = cat[ind]

###### Get Simbad Names of objects and SpT/Color Correction:
# Create empty row for names:
cat['Name'],cat['WFS 65/35 color correction'],cat['WFS Ha/IR color correction'] = np.nan, np.nan, np.nan
cat['SpT'] = np.nan
# Convert to Pandas dataframe because it's easier to work with Pandas in my opinion than astropy tables:
pdcat = cat.to_pandas()

# Import color correction from Gaia G to MagAO-X WFS filters:
p = pd.read_csv('GaiaG_WFS_color_conversion.csv')
# Separate dwarf stars (V) from giants (III):
dwarfs = [i for i in p['SpT'] if 'V' in i]
giants = [i for i in p['SpT'] if 'III' in i]
dwarfs_colors = [p['65/35 color'][i] for i in range(len(dwarfs))]
giants_colors = [p['65/35 color'][i] for i in range(len(dwarfs),len(p))]
dwarfs_colors2 = [p['Ha/IR color'][i] for i in range(len(dwarfs))]
giants_colors2 = [p['Ha/IR color'][i] for i in range(len(dwarfs),len(p))]

def get_spt_number(s):
    spt_letter_conv = {'O':0,'B':1,'A':2,'F':3,'G':4,'K':5,'M':6}
    letter = s[0]
    number = spt_letter_conv[letter]
    type_number = np.float(s[1]) / 10
    return number + type_number

def get_gaia_wfs_color_correction_and_simbadname(source_id):
    from astroquery.simbad import Simbad
    # Construct Gaia catalog name that Simbad catalogs:
    string = "Gaia EDR3 "+ str(source_id)
    # Query simbad for the object with that source id:
    # Creat custom Simbad query that includes spectral type (defaul does not):
    customSimbad = Simbad()
    customSimbad.add_votable_fields('sptype')
    # Query simbad for the Gaia source:
    r = customSimbad.query_object(string)
    # Pull out the simbad name from the table"
    name = r['MAIN_ID'][0] 
    # and the spectral type
    spt_number = get_spt_number(r['SP_TYPE'][0])
    # Look up the color correction for the star's spectra type from the table:
    if 'V' in r['SP_TYPE'][0]:
        typenumbers = [p['SpT Number'][i] for i in range(len(dwarfs))]
        colors = dwarfs_colors.copy()
        colors2 = dwarfs_colors2.copy()
    if 'III' in r['SP_TYPE'][0]:
        typenumbers = [p['SpT Number'][i] for i in range(len(dwarfs),len(p))]
        colors = giants_colors.copy()
        colors2 = giants_colors2.copy()
    else:
        pass
    ind = np.where(np.min(np.abs(spt_number - np.array(typenumbers))) == 
                   np.abs(spt_number - np.array(typenumbers)))[0][0]
    return name, colors[ind], colors2[ind], r['SP_TYPE'][0]

print('Retrieving Simbad info...')
# For each line in the table:
for i in range(len(pdcat)):
    try:
        # Query Simbad for the source's Simbad Name and Spectral Type
        sourceid = cat['source_id'][i]
        name, color_correction6535, color_correctionHaIR, s = get_gaia_wfs_color_correction_and_simbadname(
            sourceid)
        # Put the star's main id name into the table:
        pdcat['Name'].loc[i] = name
        # and color correction factor for the two WFSs:
        pdcat['WFS 65/35 color correction'].loc[i] = color_correction6535
        pdcat['WFS Ha/IR color correction'].loc[i] = color_correctionHaIR
        pdcat['SpT'].loc[i] = s
    except:
        # If the star doesn't have an EDR3 source id in Simbad, drop the object from the table:
        pdcat = pdcat.drop([i])
    update_progress(i,len(pdcat))
# Reset the index becase of the dropped objects:
pdcat = pdcat.reset_index(drop=True) 
# Perform the color correction:
pdcat['WFS 65/35 mag'] = pdcat['phot_g_mean_mag'] - pdcat['WFS 65/35 color correction']
pdcat['WFS Ha/IR mag'] = pdcat['phot_g_mean_mag'] - pdcat['WFS Ha/IR color correction']

################ Convert proper motions and ra/dec to what the LCO catalog wants:

pdcat['ra hms'], pdcat['dec dms'], pdcat['pmra s/yr'], pdcat['pmdec arcsec/yr'] = np.nan, np.nan, np.nan, np.nan
## Convert ra/dec in to hms dms:
# For each object:
for i in range(len(pdcat)):
    # Make a sky coord object:
    ob = SkyCoord(ra = pdcat['ra'][i], dec = pdcat['dec'][i], frame="icrs", unit="deg")
    # convert to string in hms and dms, and split the string in to [ra,dec]
    r = ob.to_string('hmsdms').split(' ')
    r = [r[i].replace('h',':') for i in [0,1]]
    r = [r[i].replace('m',':') for i in [0,1]]
    r = [r[i].replace('s','') for i in [0,1]]
    r = [r[i].replace('d',':') for i in [0,1]]
    # put into table:
    pdcat['ra hms'][i], pdcat['dec dms'][i] = r[0],r[1]
    # put into table:
    pdcat['ra hms'][i], pdcat['dec dms'][i] = r[0],r[1]

## convert pmra in mas/yr into s/yr and pmdec in mas/yr to arcsec/yr:
# For each object:
for i in range(len(pdcat)):
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

if args.catalog:
    if args.epoch:
        epoch = args.epoch
    else:
        print('Must supply epoch if not using EDR3')
else:
    epoch = 2016.0
pdcat['epoch'] = epoch
pdcat['equinox'] = epoch
pdcat['num'] = np.arange(1,len(pdcat)+1,1)

for i in range(len(pdcat)):
    pdcat['Name'][i] = pdcat['Name'][i].replace(' ','')

pdcat_out = pdcat[['num']]
pdcat_out['Name'] = pdcat['Name']
pdcat_out['RA'] = pdcat['ra hms']
pdcat_out['Dec'] = pdcat['dec dms']
pdcat_out['Equinox'] = pdcat['epoch']
pdcat_out['pmra'] = pdcat['pmra s/yr']
pdcat_out['pmdec'] = pdcat['pmdec arcsec/yr'] 
pdcat_out['rotang'] = 0
pdcat_out['rot_mode'] = 'GRV'
pdcat_out['RA_probe1'],pdcat_out['Dec_probe1'] = '00:00:00.00',  '+00:00:00.0'
pdcat_out['equinox'] = 2016.0
pdcat_out['RA_probe2'],pdcat_out['Dec_probe2'] = '00:00:00.00',  '+00:00:00.0'
pdcat_out['equinox '] = 2016.0
pdcat_out['epoch'] = pdcat['epoch']

pdcat_out.to_csv('Bright_AO_stars_cat_'+args.date+'.cat', index=False, sep='\t')
pdcat.to_csv('Bright_AO_stars_cat_complete_'+args.date+'.csv', index=False)
print('Done')