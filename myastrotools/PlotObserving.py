'''
Make an alt/az plot of one or multiple objects for a given date from a given location.

terminal execution: 
    if supplying a list of objects:
        python PlotObserving.py --SimbadName "HD 22259","DS Tuc","TYC 26-39-1" --LocationName "Las Campanas Observatory" --DateString "2022-12-10" --UTCOffset 3 --filename test.png
    if supplying a single object: 
        python PlotObserving.py --SimbadName "HD 22259" --LocationName "Las Campanas Observatory" --DateString "2022-12-10" --UTCOffset 3 --filename test.png
'''

def PlotObserving(SimbadName,DateString,LocationName,UTCOffset,
           plt_style = 'default',
           savefig = False,
           filename = 'observing_plot.png',
           form = 'png',
           dpi = 300,
           figsize=(7, 6),
           cmaps = ['Blues','Oranges','Purples','Reds','Greens']
                       ):
    ''' Make an alt/az plot of one or multiple objects for a given date from a given location.
    Args:
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

    
    SimbadName = SimbadName.split(',')

    nobs = len(SimbadName)
    #objects = SkyCoord.from_name(SimbadName)
    oblist = []
    for i in range(len(SimbadName)):
        ob = SkyCoord.from_name(SimbadName[i])
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
        plt.scatter(delta_midnight, obaltazs.alt, c=obaltazs.az, cmap=cmaps[i],label=SimbadName[i],s=8,lw=0)
    
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

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--SimbadName', type=str, required=True, help='Supply the Simbad name. If supplying a list, enclose each in double quotes separated by a comma with no space. Ex: --SimbadName "HD 22259","DS Tuc","TYC 26-39-1"')
parser.add_argument('--LocationName', type=str, required=True, help='Astropy Earth Location recognizeable name, enclose in double quotes to include spaces. Ex: --LocationName "Las Campanas Observatory"')
parser.add_argument('--DateString', type=str, required=True, help='Date of observation in YYYY-MM-DD')
parser.add_argument('--UTCOffset', type=int, required=True, help='UTC Offset of location in hours')
parser.add_argument('--filename', type=str, help='Filename for saved plot. If not provided, name will be SimbadName_DateString_ObservingPlot.png')

args = parser.parse_args()
SimbadName = args.SimbadName
LocationName = args.LocationName
DateString = args.DateString
UTCOffset = args.UTCOffset
if args.filename:
    filename = args.filename
else:
    filename = SimbadName + '_' + DateString + '_ObservingPlot.png'


ax = PlotObserving(SimbadName,DateString,LocationName,UTCOffset,savefig=True,filename=filename)