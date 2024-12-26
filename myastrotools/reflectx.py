import numpy as np
import astropy.units as u
import astropy.constants as c
import os
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
import pickle


def GetP(sheet_id='11u2eirdZcWZdjnKFn3vzqbCtKCodstP-KnoGXC5FdR8', 
             sheet_name='GasGiantsBaseModels'):
    url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&gid={sheet_name}"
    p = pd.read_csv(url)
    p = p.dropna(axis=1, how='all')
    for i in range(len(p)):
        try:
            if np.isnan(p[p.columns[0]][0]):
                p = p.drop(i, axis=0)
        except TypeError:
            pass
    return p


def GetDirectoryLocInP(p, directory):
    T = directory.split('-')
    planettype = T[1]
    Tstar = np.int64(T[2][5:])
    Rstar = np.float64(T[3][5:])
    sep = np.float64(T[5][3:])
    cto = np.float64(T[9][2:])
    phase = np.int64(T[10][5:])
    loc = np.where(
        (p['planet_type'] == planettype) &
        (p['st_teff'] == Tstar) & 
        (p['rstar'] == Rstar) &
        (p['au'] == sep) &
        (p['cto'] == cto) &
        (p['phase'] == phase)
    )[0]
    return loc[0]



def ConvertPlanetMHtoCKStr(m):
    prefixsign = np.sign(m)
    if prefixsign == -1:
        prefix = '-'
    else:
        prefix = '+'
    m = int(np.abs(m))
    if m / 100 >= 1.0:
        m = prefix+str(m)
    elif m == 0:
        m = '+000'
    else:
        m = prefix+'0'+str(m)
    return m

def ConvertLogPlanetMHtoCKStr(m):
    prefixsign = np.sign(m)
    if prefixsign == -1:
        prefix = '-'
    else:
        prefix = '+'
    m = m * 100
    if m >= 100:
        m = prefix+str(m).replace('.0','')
    elif m == 0:
        m = '+000'
    else:
        m = prefix+'0'+str(m).replace('.0','')
    return m

def ConvertCtoOtoStr(c):
    if c >= 1:
        cc = str(c*100).replace('.0','')
    else:
        cc = '0'+str(c*100).replace('.0','')
    return cc

def ComputeTeq(StarTeff, StarRad, sep, Ab = 0.3, fprime = 1/4):
    ''' Given a star's Teff and Rad, compute the equil temp at the given separation.
    from Seager 2016 Exoplanet Atmospheres eqn 3.9
    https://books.google.com/books?id=XpaYJD7IE20C

    Args:
        StarTeff (flt): Star's effective temperature
        StarRad (astropy units object): Star's radius, must be an astropy units object
        sep (astropy units object): Planet-Star separation, must be an astropy units object
        Ab (flt): Bond albedo, default = 0.3
        fprime (flt): function for describing hemisphere heat distribution, default = 1/4
    Returns
        flt: equilibrium temperature
    '''
    StarRad = StarRad.to(u.km)
    sep = sep.to(u.km)
    return (StarTeff * np.sqrt(StarRad/sep) * ((fprime * (1 - Ab))**(1/4))).value

def ComputeSMA(Teq, StarTeff, StarRad, Ab = 0.3, fprime = 1/4):
    ''' Given a star's Teff and Rad, compute the separation that gives the 
        specified equil temp.

    Args:
        Teq (flt): planet's equilibrium temperature
        StarTeff (flt): Star's effective temperature
        StarRad (astropy units object): Star's radius, must be an astropy units object
        Ab (flt): Bond albedo, default = 0.3
        fprime (flt): function for describing hemisphere heat distribution, default = 1/4
    Returns
        astropy units object: Planet-Star separation, an astropy units object
    '''
    StarRad = StarRad.to(u.km)
    sma = (StarTeff / Teq)**2 * ((fprime * (1 - Ab))**(1/2)) * StarRad
    return sma.to(u.au)

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

def MakePTProflePlot(atm_df):
    #### Plot PT profile:
    fig = plt.figure(figsize=(10,10))
    plt.ylabel("Pressure [Bars]", fontsize=25)
    plt.xlabel('Temperature [K]', fontsize=25)
    plt.ylim(500,1e-6)
    #plt.xlim(0,3000)
    plt.semilogy(atm_df['temperature'],
                atm_df['pressure'],color="r",linewidth=3)
    plt.minorticks_on() 
    plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)
    #plt.legend(fontsize=15)

    return fig

def GetPlanetFlux(PlanetWNO,PlanetFPFS,StarWNO,StarFlux, return_resampled_star_flux = False):
    from scipy.interpolate import interp1d
    func = interp1d(StarWNO,StarFlux,fill_value="extrapolate")
    ResampledStarFlux = func(PlanetWNO)
    if return_resampled_star_flux:
        return ResampledStarFlux*PlanetFPFS, PlanetWNO, ResampledStarFlux
    return ResampledStarFlux*PlanetFPFS

def MakeAbundancesPlot(atm_df, n_mols_to_plot = 8):
    sort = np.argsort(atm_df.loc[0])[::-1]
    highest_abundance = np.array(atm_df.keys()[sort])
    highest_abundance = np.delete(highest_abundance,np.where(highest_abundance=='temperature'))
    highest_abundance = np.delete(highest_abundance,np.where(highest_abundance=='pressure'))
    highest_abundance = np.delete(highest_abundance,np.where(highest_abundance=='e-'))
    import matplotlib
    cmap = matplotlib.cm.get_cmap('Spectral')
    mols = highest_abundance[:n_mols_to_plot+1]
    linestyles = ['-','-.']*len(mols)
    lineweights = [4,4,2,2]*len(mols)
    n = len(mols)
    cs = np.linspace(0.1,1,n)
    colors = cmap(cs)
    fig = plt.figure(figsize=(10,10))
    plt.minorticks_on()
    plt.tick_params(axis='both',which='major',length =30, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

    plt.ylabel("Pressure [Bars]", fontsize=25)
    plt.xlabel('Abundance', fontsize=25)
    plt.ylim(500,1e-4)
    plt.xlim(left=1e-6)

    for i,mol in enumerate(mols):
        plt.plot(atm_df[mol],
                atm_df['pressure'],
                color=colors[i],linewidth=lineweights[i],label=mol,ls=linestyles[i])
    plt.gca().set_yscale('log')
    plt.gca().set_xscale('log')
    #plt.gca().invert_yaxis()
    plt.legend(fontsize=20)
    return fig

def Make4SpectrumPlot(PlanetWNO,PlanetAlbedo,PlanetFPFS,StarWNO,StarFlux,colormap = 'magma',
                      plotstyle='magrathea', PlanetFlux = []):
    import matplotlib
    plt.style.use(plotstyle)
    cmap = matplotlib.cm.get_cmap(colormap)
    n = 6
    cs = np.linspace(0,1,n)
    colors = cmap(cs)
    
    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    ax1 = axs[0, 0]
    ax2 = axs[0, 1]
    ax3 = axs[1, 0]
    ax4 = axs[1, 1]
    
    ax1.minorticks_on()
    ax1.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
    ax1.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)
    ax1.plot(1e4/PlanetWNO, PlanetAlbedo, lw=2, color=colors[1])
    ax1.set_yscale('log')
    #ax1.set_xlabel(r'Wavelength [$\mu$m]')
    ax1.set_ylabel('Planet Albedo')
    ax1.grid(ls=':')
    
    ax2.minorticks_on()
    ax2.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
    ax2.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)
    ax2.plot(1e4/PlanetWNO, PlanetFPFS, lw=2, color=colors[2])
    ax2.set_yscale('log')
    #ax2.set_xlabel(r'Wavelength [$\mu$m]')
    ax2.set_ylabel('Planet:Star contrast')
    ax2.grid(ls=':')
    
    ax3.minorticks_on()
    ax3.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
    ax3.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)
    ax3.plot(1e4/StarWNO, StarFlux, lw=2, color=colors[3])
    ax3.set_yscale('log')
    ax3.set_xlabel(r'Wavelength [$\mu$m]')
    ax3.set_ylabel(r'Star flux [ergs cm$^{-2}$ s$^{-1}$ cm$^{-1}$]')
    ax3.grid(ls=':')
    if len(PlanetFlux) == 0:
        PlanetFlux = GetPlanetFlux(PlanetWNO,PlanetFPFS,StarWNO,StarFlux)
    else:
        pass
    ax4.minorticks_on()
    ax4.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
    ax4.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)
    ax4.plot(1e4/PlanetWNO, PlanetFlux, lw=2, color=colors[4])
    ax4.set_yscale('log')
    ax4.set_xlabel(r'Wavelength [$\mu$m]')
    ax4.set_ylabel(r'Planet flux [ergs cm$^{-2}$ s$^{-1}$ cm$^{-1}$]')
    ax4.grid(ls=':')
    plt.tight_layout()
    return fig

def ComputeSpectrum(atm_df, pdict, sdict, specdict, calculation = 'planet', 
                   clouddict = None,
                   save_individual_spectrum_plots = True, savefiledirectory = None, 
                   ComputePlanetFlux = True, make4spectrumplot = True,
                   MixingRatioPlot = True, AbundancesPlot = True, n_mols_to_plot = 8,
                   PTplot = True,
                   plotstyle = 'magrathea'):
    
    import picaso.justdoit as jdi
    if specdict == None:
        opa_mon = jdi.opannection()
    else:
        opacity_db = specdict['opacity_db']
        if opacity_db == None:
            opa_mon = jdi.opannection(wave_range=specdict['wave_range'])
        else:
            opa_mon = jdi.opannection(filename_db = opacity_db, wave_range=specdict['wave_range'])
    spec = jdi.inputs(calculation=calculation)

    spec.phase_angle(phase=pdict['phase']*np.pi/180, num_tangle=pdict['num_tangle'],
                     num_gangle=pdict['num_gangle'])
    if not pdict['gravity']:
        spec.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        spec.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])

    # set up star:
    if 'star_filename' in sdict.keys():
        spec.star(opa_mon, temp=None, metal=None, logg=None ,radius = sdict['star_radius'], radius_unit=u.Rsun,
             semi_major=pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], 
             filename=sdict['star_filename'], w_unit='um', f_unit='FLAM')
    else:
        spec.star(opa_mon, temp = sdict['Teff'], metal = np.log10(sdict['mh']), logg = sdict['logg'], 
            radius = sdict['radius'], radius_unit = u.R_sun, 
            semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database = 'phoenix')
        
    StarWNO,StarFlux = spec.inputs['star']['wno'],spec.inputs['star']['flux']

    spec.atmosphere(df=atm_df)

    if clouddict != None:
        clouds_added = spec.virga(
            clouddict['condensates'], 
            clouddict['virga_mieff'], 
            fsed=clouddict['fsed'], mh=10**(0),
            mmw = clouddict['mean_mol_weight'], full_output=True)
        

    spec_df = spec.spectrum(opa_mon, 
                            calculation='reflected', 
                            full_output=True)
    wno, alb, fpfs, full_output = spec_df['wavenumber'], spec_df['albedo'], \
                                    spec_df['fpfs_reflected'],  spec_df['full_output']
    wno,fpfs = jdi.mean_regrid(spec_df['wavenumber'],
                          spec_df['fpfs_reflected'], R=specdict['R'])
    wno,alb = jdi.mean_regrid(spec_df['wavenumber'],
                          spec_df['albedo'], R=specdict['R'])
    
    if '/' not in savefiledirectory:
        savefiledirectory = savefiledirectory+'/'

    if save_individual_spectrum_plots:
        plt.style.use(plotstyle)
        # Albedo plot
        plt.figure(figsize=(8,6))
        plt.minorticks_on()
        plt.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
        plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

        plt.plot(1e4/wno, alb, lw=2)

        plt.gca().set_xlabel(r'Wavelength [$\mu$m]')
        plt.gca().set_ylabel('Geo Albedo')
        plt.grid(ls=':')
        plt.tight_layout()
        plt.savefig(savefiledirectory+'-albedo-spectrum-R'+str(specdict['R'])+'.png',
                    bbox_inches='tight')
        plt.close()

        # reflected plot
        plt.figure(figsize=(8,6))
        plt.minorticks_on()
        plt.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
        plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

        plt.plot(1e4/wno, fpfs, lw=2)
        plt.gca().set_yscale('log')

        plt.gca().set_xlabel(r'Wavelength [$\mu$m]')
        plt.gca().set_ylabel('Planet:Star contrast')
        plt.grid(ls=':')
        plt.tight_layout()
        plt.savefig(savefiledirectory+'-fpfs-spectrum-R'+str(specdict['R'])+'.png',
                    bbox_inches='tight')
        plt.close()
        
    
    if make4spectrumplot:
        fig = Make4SpectrumPlot(wno,alb,fpfs,StarWNO,StarFlux,colormap = 'magma')
        fig.savefig(savefiledirectory+'-4SpectrumPlot-R'+str(specdict['R'])+'.png',
                    bbox_inches='tight')
        plt.close()
        
    if MixingRatioPlot:
        import picaso.justplotit as jpi
        mrp = jpi.mixing_ratio(full_output, plot_height=900, plot_width=600)
        # Save mixing ratio plot for future inspection
        from bokeh.plotting import output_file, save
        output_file(savefiledirectory+"-mixing-ratios.html")
        save(mrp)

    if AbundancesPlot:
        fig = MakeAbundancesPlot(atm_df, n_mols_to_plot = n_mols_to_plot)
        fig.savefig(savefiledirectory+'-abundances.png',
                bbox_inches='tight')
        plt.close()
        
    if PTplot:
        fig = MakePTProflePlot(atm_df)
        fig.savefig(savefiledirectory+'-PTprofile.png',
                    bbox_inches='tight')
        plt.close()
        
    if ComputePlanetFlux:
        PlanetFlux = GetPlanetFlux(wno, fpfs, StarWNO, StarFlux)
        # reflected plot
        plt.figure(figsize=(8,6))
        plt.minorticks_on()
        plt.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
        plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

        plt.plot(1e4/wno, PlanetFlux, lw=2)
        plt.gca().set_yscale('log')

        plt.gca().set_xlabel(r'Wavelength [$\mu$m]')
        plt.gca().set_ylabel(r'Planet flux [ergs cm$^{-2}$ s$^{-1}$ cm$^{-1}$]')
        plt.grid(ls=':')
        plt.tight_layout()
        plt.savefig(savefiledirectory+'-planet-spectrum-R'+str(specdict['R'])+'.png',
                    bbox_inches='tight')
        plt.close()
        return wno, alb, fpfs, full_output, PlanetFlux, StarWNO, StarFlux

    return wno, alb, fpfs, full_output

def VirgaRecommendHack(pressure, temperature, mh, mmw, 
                       plot=False, return_plot=False, 
                       legend='inside',**plot_kwargs):
    
    from virga.justdoit import available, condensation_t
    from virga.pvaps import NH3
    from virga.justplotit import plot_format
    
    if plot: 
        from bokeh.plotting import figure, show
        from bokeh.models import Legend
        from bokeh.palettes import magma   
        plot_kwargs['y_range'] = plot_kwargs.get('y_range',[1e2,1e-3])
        plot_kwargs['height'] = plot_kwargs.get('plot_height',plot_kwargs.get('height',400))
        plot_kwargs['width'] = plot_kwargs.get('plot_width', plot_kwargs.get('width',600))
        if 'plot_width' in plot_kwargs.keys() : plot_kwargs.pop('plot_width')
        if 'plot_height' in plot_kwargs.keys() : plot_kwargs.pop('plot_height')
        plot_kwargs['x_axis_label'] = plot_kwargs.get('x_axis_label','Temperature (K)')
        plot_kwargs['y_axis_label'] = plot_kwargs.get('y_axis_label','Pressure (bars)')
        plot_kwargs['y_axis_type'] = plot_kwargs.get('y_axis_type','log')        
        fig = figure(**plot_kwargs)
    
    all_gases = available()
    cond_ts = []
    recommend = []
    line_widths = []
    for gas_name in all_gases: #case sensitive names
        if gas_name == 'NH3':
            pass
        else:
            #grab p,t from eddysed
            cond_p,t = condensation_t(gas_name, mh, mmw)
            cond_ts +=[t]

            interp_cond_t = np.interp(pressure,cond_p,t)

            diff_curve = interp_cond_t - temperature

            if ((len(diff_curve[diff_curve>0]) > 0) & (len(diff_curve[diff_curve<0]) > 0)):
                recommend += [gas_name]
                line_widths +=[5]
            else: 
                line_widths +=[1] 
    # NH3:
    temp = np.linspace(10,400,1000)
    pvapNH3 = NH3(temp)
    interp_cond_t = np.interp(pressure,pvapNH3,temp)
    diff_curve = interp_cond_t - temperature
    if ((len(diff_curve[diff_curve>0]) > 0) & (len(diff_curve[diff_curve<0]) > 0)):
        recommend += ['NH3']
        line_widths +=[5]
    else: 
        line_widths +=[1] 
        
    if plot: 
        legend_it = []
        ngas = len(all_gases)
        cols = magma(ngas)
        if legend == 'inside':
            fig.line(temperature,pressure, legend_label='User',color='black',line_width=5,line_dash='dashed')
            for i in range(ngas-1):

                fig.line(cond_ts[i],cond_p, legend_label=all_gases[i],color=cols[i],line_width=line_widths[i])
        else:
            f = fig.line(temperature,pressure, color='black',line_width=5,line_dash='dashed')
            legend_it.append(('input profile', [f]))
            for i in range(ngas):

                f = fig.line(cond_ts[i],cond_p ,color=cols[i],line_width=line_widths[i])
                legend_it.append((all_gases[i], [f]))

        if legend == 'outside':
            legend = Legend(items=legend_it, location=(0, 0))
            legend.click_policy="mute"
            fig.add_layout(legend, 'right')   
        
        plot_format(fig)
        if return_plot: 
            return recommend, fig
        else: 
            show(fig) 
            return recommend

def MakeModelCloudFreePlanet(pdict, sdict,
                calculation = "planet",
                use_guillotpt = True,
                user_supplied_ptprofile = None,
                cdict = None,
                compute_spectrum = True,
                specdict = None,
                savefiledirectory = None,
                record_terminal_output = True
             ):
    
    ''' Wrapper for PICASO functions for building a cloud-free planet base model
    Args:
        pdict (dict): dictionary of planet parameter inputs
        sdict (dict): dictionary of star parameter inputs
        calculation (str): picaso input for object, "planet" or "brown dwarf"
        use_guillotpt (bool): if True, use Guillot PT approximation. Else user must supply initial PT profile
        user_supplied_ptprofile (df): user supplied pt profile for picaso
        cdict (dict): dictionary of climate run setup params
        specdict (dict): dictionary of spectrum setup params
        savefilename (str): filename and path for the model to be saved.

    Returns:
        pl: picaso planet model inputs
        noclouds: picaso object after climate run before clouds

    Writes to disk:
        terminal_output.txt: text file containing terminal output from run
        recommended-gases.html: bokeh plot fro virga of recommended molecules for cloud run
        cloud-free-model.pkl: pickle file of jdi inputs object (pl) and atmosphere (noclouds)
        cloud-free-model-inputs.pkl: pickle file of dictionary inputs pdict, sdict, cdict
        cloud-free-spectrum-full-output-RXXX.pkl: pickle file of dictionary of all spectrum objects - 
                wavenumber, albedo spectrum, planet/star flux contrast, planet flux, star spectrum wavenumber, star flux
        cloud-free-fpfs-spectrum-RXXX.png: plot of planet star flux contrast
        cloud-free-alb-spectrum-RXXX.png: plot of planet albedo spectrum
        cloud-free-4SpectrumPlot-RXXX.png: plot of planet albedo, contrast, star flux, and planet flux
    '''
    import warnings
    warnings.filterwarnings('ignore')
    import picaso.justdoit as jdi
    
    import sys
    #import os
    # Make directory to store run results:
    #os.system('mkdir '+savefiledirectory)
    
    # Set all terminal output to write to file:
    if record_terminal_output:
        f = open(savefiledirectory+"/terminal_output.txt", 'w')
        sys.stdout = f
    # Additional info for xarray (not used)
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
    PlanetCO = ConvertCtoOtoStr(PlanetCO)
    if type(pdict['planet_mh_str']) == str:
        PlanetMHStr = pdict['planet_mh_str']
    else:
        PlanetMHStr = ConvertPlanetMHtoCKStr(pdict['planet_mh_str'])

    if pdict['noTiOVO']:
        ck_db_name = pdict['local_ck_path'] + f'sonora_2020_feh{PlanetMHStr}_co_{PlanetCO}_noTiOVO.data.196'
    else:
        ck_db_name = pdict['local_ck_path'] + f'sonora_2020_feh{PlanetMHStr}_co_{PlanetCO}.data.196'
    print(ck_db_name)
    # Set opacity connection:
    if specdict == None:
        opacity_ck = jdi.opannection(ck_db=ck_db_name)
    else:
        opacity_ck = jdi.opannection(ck_db=ck_db_name, wave_range = specdict['wave_range'])
    
    # initialize model:
    pl = jdi.inputs(calculation= calculation, climate = True)
    
    ### set up planet:
    # input effective temperature
    pl.effective_temp(pdict['tint']) 
    # add gravity:
    if not pdict['gravity']:
        pl.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        pl.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
        
    # set up star:
    if 'star_filename' in sdict.keys():
        pl.star(opacity_ck, temp=None, metal=None, logg=None ,radius = sdict['star_radius'], radius_unit=u.Rsun,
             semi_major=pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], 
             filename=sdict['star_filename'], w_unit='um', f_unit='FLAM')
    else:
        pl.star(opacity_ck, temp = sdict['Teff'], metal = np.log10(sdict['mh']), logg = sdict['logg'], 
            radius = sdict['radius'], radius_unit = u.R_sun, 
            semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database = 'phoenix')
    
    # add phase:
    phase = pdict['phase']
    # If full phase:
    if phase == 0:
        # Use symmetry to speed up calculation.
        num_tangle = 1
        pdict['num_tangle'] = 1
    else:
        num_tangle = pdict['num_tangle']
    pl.phase_angle(phase=phase*np.pi/180, num_tangle=num_tangle, num_gangle=pdict['num_gangle'])
    
    # Add initial P(T) profile guess:
    if use_guillotpt:
        pt = pl.guillot_pt(pdict['Teq'], nlevel=cdict['nlevel'], T_int = pdict['tint'], 
                              p_bottom=cdict['climate_pbottom'], p_top=cdict['climate_ptop'])
    else:
        pt = user_supplied_ptprofile

    # initial PT profile guess:
    temp_guess = pt['temperature'].values 
    press_guess = pt['pressure'].values
    # Input climate params:
    nstr = np.array([0,cdict['nstr_upper'],cdict['nstr_deep'],0,0,0]) # initial guess of convective zones
    # Set up climate run inputs:
    pl.inputs_climate(temp_guess= temp_guess, pressure= press_guess, 
                  nstr = nstr, nofczns = cdict['nofczns'] , rfacv = cdict['rfacv'])
    print('starting climate run')
    # Compute climate:
    noclouds = pl.climate(opacity_ck, save_all_profiles=True, with_spec=True)
    # Set atm to climate run results:
    pl.atmosphere(df=noclouds['ptchem_df'])

    # Generate recommended molecules for clouds:
    from virga import justdoit as vj
    # pressure temp profile from climate run:
    temperature = noclouds['temperature']
    pressure = noclouds['pressure']
    # metallicity = pdict['mh'] #atmospheric metallicity relative to Solar
    metallicity_TEMP = 0 # mh must be 1 for this part
    mmw = 2.2
    # got molecules for cloud species:
    # recommended, fig = vj.recommend_gas(pressure, temperature, 10**(metallicity_TEMP), 
    #                                mmw, plot=True, return_plot = True)

    ### 08-17-2023: There is an issue with virga.recommend for 'NH3', it's not correct.  So hacking together
    ### a solution:
    recommended, fig = VirgaRecommendHack(pressure, temperature, 10**(metallicity_TEMP), 
                                mmw, plot=True, return_plot = True)
    pickle.dump(recommended,open(savefiledirectory+'/virga-recommended-mols.pkl','wb'))
    
    print('Virga recommended molecules:', recommended)
    
    # Save recommended plot for future inspection
    from bokeh.plotting import figure, output_file, save
    output_file(savefiledirectory+"/recomended-gases.html")
    save(fig)

    sdict.update({'flux_unit':'ergs cm^-2 s^-1 cm^-1'})

    # Pickle info and save:
    #pickle.dump(pl, open(savefiledirectory+'/cloud-free-model.pkl','wb'))
    pickle.dump(noclouds, open(savefiledirectory+'/cloud-free-model.pkl','wb'))
    pickle.dump([pdict, sdict, cdict], open(savefiledirectory+'/cloud-free-model-inputs.pkl','wb'))

    # If you want to compute spectrum here (recommended)
    if compute_spectrum:
        ### Make cloud-free spectrum:
        wno, alb, fpfs, full_output, PlanetFlux, StarWNO, StarFlux = ComputeSpectrum(noclouds['ptchem_df'], 
                                                                  pdict, sdict, specdict, 
                                                                  calculation = 'planet',
                                                                  save_individual_spectrum_plots = False, 
                                                                  clouddict = None,
                                                                  savefiledirectory = savefiledirectory+'/cloud-free', 
                                                                  ComputePlanetFlux = True, 
                                                                  make4spectrumplot = True,
                                                                  AbundancesPlot = True, 
                                                                  MixingRatioPlot = True,
                                                                  PTplot = True
                                                                  )
        # dump it out:
        outdict = {'wavenumber':wno, 'albedo spectrum':alb,
                   'planet star contrast':fpfs, 'full_output':full_output,
                   'planet flux': PlanetFlux, 'star wavenumber':StarWNO, 'star flux':StarFlux}
        pickle.dump(outdict, 
                    open(savefiledirectory+'/cloud-free-spectrum-full-output-R'+str(specdict['R'])+'.pkl','wb'))
        from scipy.interpolate import interp1d
        func = interp1d(StarWNO,StarFlux,fill_value="extrapolate")
        ResampledStarFlux = func(wno)
        outdf = pd.DataFrame(data={'wavelength [um]':1e4/wno,
                                   'planet flux [ergs/cm2/s/cm]':PlanetFlux,
                                   'star flux [ergs/cm2/s/cm]':ResampledStarFlux,
                                   'albedo':alb,
                                   'planet-star contrast':fpfs }
                             )
        outdf.to_csv(savefiledirectory+'/cloud-free-spectrum-R'+str(specdict['R'])+'.csv', index=False, sep=' ')

    else:
        # wno, alb, fpfs, full_output = ComputeSpectrum(noclouds['ptchem_df'], 
        #                                                           pdict, sdict, specdict, 
        #                                                           calculation = 'planet',
        #                                                           saveplots = False, 
        #                                                           clouddict = None,
        #                                                           savefiledirectory = savefiledirectory+'/cloud-free', 
        #                                                           ComputePlanetFlux = False, 
        #                                                           make4spectrumplot = False,
        #                                                           AbundancesPlot = True, 
        #                                                           MixingRatioPlot = False,
        #                                                           PTplot = True
        #                                                           )
        fig1 = MakePTProflePlot(noclouds['ptchem_df'])
        fig1.savefig(savefiledirectory+'/cloudfree-PTprofile.png',
                    bbox_inches='tight')
        fig2 = MakeAbundancesPlot(noclouds['ptchem_df'], n_mols_to_plot = 8)
        fig2.savefig(savefiledirectory+'/cloudfree-abundances.png',
                bbox_inches='tight')
        noclouds['ptchem_df'].to_csv(savefiledirectory+'/cloud-free-atm-df.csv', index=False)
    if record_terminal_output:
        f.close()
    
    return pl, noclouds, recommended
    

def MakeModelCloudyPlanet(savefiledirectory, clouddict, 
                          specdict,
                          compute_spectrum = True,
                          saveplots = False,
                          calculation = 'planet',
                          record_terminal_output = True
                          ):
    
    ''' Wrapper for PICASO functions for building a planet model
    Args:
        savefiledirectory (str): directory containing picaso cloud-free model base case.
        clouddict (dict): dictionary of cloud parameter inputs
        cloud_filename_prefix (str): name to prepend output files
        calculation (str): picaso input for object, "planet" or "brown dwarf"

    Returns:
        clouds: picaso planet model inputs
        clouds_added: virga cloud run output clouds
    '''
    # import climate run output:
    import pickle
    import picaso.justdoit as jdi
    
    import sys
    if record_terminal_output:
        f = open(savefiledirectory+"/terminal_output.txt", 'w')
        sys.stdout = f
    
    
    pl, noclouds = pickle.load(open(savefiledirectory+'/cloud-free-model.pkl','rb'))
    pdict, sdict, cdict = pickle.load(open(savefiledirectory+'/cloud-free-model-inputs.pkl','rb'))
    
    #opa_mon = jdi.opannection()
    PlanetMHStr = ConvertPlanetMHtoCKStr(pdict['planet_mh_str'])
    PlanetCO = ConvertCtoOtoStr(pdict['CtoO'])
    if pdict['noTiOVO']:
        ck_db_name = pdict['local_ck_path'] + f'sonora_2020_feh{PlanetMHStr}_co_{PlanetCO}_noTiOVO.data.196'
    else:
        ck_db_name = pdict['local_ck_path'] + f'sonora_2020_feh{PlanetMHStr}_co_{PlanetCO}.data.196'
    print(ck_db_name)
    # Set opacity connection:
    opacity_ck = jdi.opannection(ck_db=ck_db_name, wave_range = specdict['wave_range'])

    from virga import justdoit as vj
    # pressure temp profile from climate run:
    temperature = noclouds['temperature']
    pressure = noclouds['pressure']
    #metallicity = pdict['mh'] #atmospheric metallicity relative to Solar
    metallicity_TEMP = 0
    # got molecules for cloud species:
    if clouddict['condensates'] == 'virga recommend':
        # if no user-supplied molecules:
        #get virga recommendation for which gases to run
        # metallicitiy must be in NOT log units
        recommended = VirgaRecommendHack(pressure, temperature, 10**(metallicity_TEMP), 
                                clouddict['mean_mol_weight'], plot=False, return_plot = False)
        # recommended = vj.recommend_gas(pressure, temperature, 10**(metallicity_TEMP), 
        #                                clouddict['mean_mol_weight'],
        #                 #Turn on plotting
        #                  plot=False)
        mols = recommended
        print('using virga recommended gas species:',recommended)
    else:
        # otherwise use user supplied mols:
        mols = clouddict['condensates']
    # add atmosphere from climate run:
    atm = noclouds['ptchem_df']
    # add kzz:
    atm['kz'] = [clouddict['kz']]*atm.shape[0]

    # Just set all this up again:
    clouds = jdi.inputs(calculation=calculation) # start a calculation
    
    # add phase:
    phase = pdict['phase']
    # If full phase:
    if phase == 0:
        # Use symmetry to speed up calculation.
        num_tangle = 1
        pdict['num_tangle'] = 1
    else:
        num_tangle = pdict['num_tangle']
    pl.phase_angle(phase=phase*np.pi/180, num_tangle=num_tangle, num_gangle=pdict['num_gangle'])

    # add gravity:
    if not pdict['gravity']:
        clouds.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        clouds.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
    # add star:
    # clouds.star(opa_mon, temp = sdict['Teff'], metal = np.log10(sdict['mh']), logg = sdict['logg'], 
        # radius = sdict['radius'], radius_unit=u.R_sun, 
        # semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database='phoenix')
    clouds.star(opacity_ck, temp = sdict['Teff'], metal = np.log10(sdict['mh']), logg = sdict['logg'], 
        radius = sdict['radius'], radius_unit=u.R_sun, 
        semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database='phoenix')
    # add atmosphere from climate run with kzz:
    clouds.atmosphere(df=atm)
    # get clouds from reference:

    meiff_directory ='/Volumes/Oy/virga/virga/reference/RefIndexFiles'
    clouddict['virga_mieff'] = meiff_directory
    #meiff_directory = clouddict['virga_mieff']

    metallicity_TEMP = 0
    clouds_added = clouds.virga(mols, meiff_directory, fsed=clouddict['fsed'], mh=10**(metallicity_TEMP),
                         mmw = clouddict['mean_mol_weight'], full_output=True)

    clouddict.update({'condensates':mols})

    # savefiledirectory = savefiledirectory+'/'
    # outdir = savefiledirectory+cloud_filename_prefix+'/'
    # os.system('mkdir '+ outdir)

    # pickle.dump([clouds, clouds_added],
    #                 open(outdir+'-cloudy-model.pkl','wb'))
    # pickle.dump([pdict, sdict, cdict, clouddict],
    #             open(outdir+'-cloudy-model-inputs.pkl','wb'))
    
    # If you want to compute spectrum here (recommended)
    if compute_spectrum:
        ### Make cloud-free spectrum:
        wno, alb, fpfs, full_output, PlanetFlux, StarWNO, StarFlux = ComputeSpectrum(noclouds['ptchem_df'], 
                                                                  pdict, sdict, specdict, 
                                                                  clouddict = clouddict,
                                                                  calculation = 'planet',
                                                                  saveplots = False, 
                                                                  #savefiledirectory = outdir, 
                                                                  ComputePlanetFlux = True, 
                                                                  make4spectrumplot = True,
                                                                  MixingRatioPlot = False,
                                                                  AbundancesPlot = False,
                                                                  PTplot = False)
        # dump it out:
        # outdict = {'wavenumber':wno, 'albedo spectrum':alb,
        #            'planet star contrast':fpfs, 'full output':full_output,
        #            'planet flux': PlanetFlux, 'star wavenumber':StarWNO, 'star flux':StarFlux}
        # pickle.dump(outdict, 
        #             open(outdir+'-spectrum-full-output-R'+str(specdict['R'])+'.pkl','wb'))
        # from scipy.interpolate import interp1d
        # func = interp1d(StarWNO,StarFlux,fill_value="extrapolate")
        # ResampledStarFlux = func(wno)
        # outdf = pd.DataFrame(data={'wavelength [um]':1e4/wno,
        #                            'planet flux [ergs/cm2/s/cm]':PlanetFlux,
        #                            'star flux [ergs/cm2/s/cm]':ResampledStarFlux,
        #                            'albedo':alb,
        #                            'planet-star contrast':fpfs }
        #                      )
        # outdf.to_csv(outdir+'-spectrum-R'+str(specdict['R'])+'.csv', index=False, sep=' ')
    f.close()
    # GenerateInitialReadMe(savefiledirectory,planettype,T_star,r_star,Teq_str,semi_major,radius,massj,
    #                planet_mh,planet_mh_CtoO,phase,pdict,sdict,cdict,specdict,recommended)

    return clouds, clouds_added, wno, alb, fpfs, full_output, PlanetFlux, StarWNO, StarFlux



def RegridSpectrum(directory, spectrum_full_output_pickle_file, R, make4spectrumplot = False):
    ''' Regrid a spectrum
    
    Args:
        directory (str): location of filder containing orginal FULL OUTPUT spectrum pickle file
        spectrum_full_output_pickle_file (str): pickle file of dataframe of original spectrum. 
                    Ex: /cloud-free-spectrum-full-output-R2000.pkl
        R (flt): R of new spectrum
    
    '''
    import pickle
    import os
    import picaso.justdoit as jdi

    df = pickle.load(open(directory+spectrum_full_output_pickle_file,'rb'))
    wno, alb, fpfs, full_output, PlanetFlux, StarWNO, StarFlux = df['wavenumber'], df['albedo spectrum'], df['planet star contrast'],\
        df['full_output'], df['planet flux'], df['star wavenumber'], df['star flux']
    
    if 'cloud-free' in spectrum_full_output_pickle_file:
        fileout = 'cloud-free-'
    elif 'cloudy' in spectrum_full_output_pickle_file:
        fileout = 'cloudy-'

    if make4spectrumplot:
        fig = Make4SpectrumPlot(wno,alb,fpfs,StarWNO,StarFlux,colormap = 'magma')
        fig.savefig(directory+fileout+'-4SpectrumPlot-R'+str(R)+'.png',
                    bbox_inches='tight')
    
    # Albedo plot
    Rwno, Ralb = jdi.mean_regrid(wno, alb , R = R)
    plt.figure()
    plt.figure(figsize=(8,6))
    plt.minorticks_on()
    plt.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

    plt.plot(1e4/Rwno, Ralb, lw=2)

    plt.gca().set_xlabel(r'Wavelength [$\mu$m]')
    plt.gca().set_ylabel('Geo Albedo')
    plt.grid(ls=':')
    plt.tight_layout()
    plt.savefig(directory+fileout + 'albedo-spectrum-R' + R + '.png',bbox_inches='tight')

    # reflected plot
    Rwno, Rfpfs = jdi.mean_regrid(wno, fpfs , R = R)
    plt.figure()
    plt.figure(figsize=(8,6))
    plt.minorticks_on()
    plt.tick_params(axis='both',which='major',length =20, width=2,direction='in',labelsize=23)
    plt.tick_params(axis='both',which='minor',length =10, width=2,direction='in',labelsize=23)

    plt.plot(Rwno, Rfpfs, lw=2)
    plt.gca().set_yscale('log')

    plt.gca().set_xlabel(r'Wavelength [$\mu$m]')
    plt.gca().set_ylabel('Planet:Star contrast')
    plt.grid(ls=':')
    plt.tight_layout()
    plt.savefig(directory+fileout + 'fpfs-spectrum-R' + R + '.png',bbox_inches='tight')

    pickle.dump([wno, alb, fpfs, full_output], open(directory+fileout+'-spectrum-full-output-R' + str(R) + '.pkl','wb'))
    
    from scipy.interpolate import interp1d

    from scipy.interpolate import interp1d
    func = interp1d(StarWNO,StarFlux,fill_value="extrapolate")
    ResampledStarFlux = func(Rwno)
    RPlanetFlux = ResampledStarFlux*Rfpfs

    outdf = pd.DataFrame(data={'wavelength [um]':1e4/Rwno,
                                'planet flux [ergs/cm2/s/cm]':RPlanetFlux,
                                'star flux [ergs/cm2/s/cm]':ResampledStarFlux,
                                'albedo':Ralb,
                                'planet-star contrast':Rfpfs }
                            )
    outdf.to_csv(directory+fileout+cloud_filename_prefix+'-spectrum-R'+str(R)+'.csv', index=False, sep=' ')
    


def MakeModelCloudFreePlanetWithDisEqChemKZZHack(pdict, sdict,
                calculation = "planet",
                use_guillotpt = True,
                user_supplied_ptprofile = None,
                cdict = None,
                compute_spectrum = True,
                specdict = None,
                savefiledirectory = None,
                record_terminal_output = True
             ):
    
    ''' Wrapper for PICASO functions for building a cloud-free planet base model
    Args:
        pdict (dict): dictionary of planet parameter inputs
        sdict (dict): dictionary of star parameter inputs
        calculation (str): picaso input for object, "planet" or "brown dwarf"
        use_guillotpt (bool): if True, use Guillot PT approximation. Else user must supply initial PT profile
        user_supplied_ptprofile (df): user supplied pt profile for picaso
        cdict (dict): dictionary of climate run setup params
        specdict (dict): dictionary of spectrum setup params
        savefilename (str): filename and path for the model to be saved.

    Returns:
        pl: picaso planet model inputs
        noclouds: picaso object after climate run before clouds

    Writes to disk:
        terminal_output.txt: text file containing terminal output from run
        recommended-gases.html: bokeh plot fro virga of recommended molecules for cloud run
        cloud-free-model.pkl: pickle file of jdi inputs object (pl) and atmosphere (noclouds)
        cloud-free-model-inputs.pkl: pickle file of dictionary inputs pdict, sdict, cdict
        cloud-free-spectrum-full-output-RXXX.pkl: pickle file of dictionary of all spectrum objects - 
                wavenumber, albedo spectrum, planet/star flux contrast, planet flux, star spectrum wavenumber, star flux
        cloud-free-fpfs-spectrum-RXXX.png: plot of planet star flux contrast
        cloud-free-alb-spectrum-RXXX.png: plot of planet albedo spectrum
        cloud-free-4SpectrumPlot-RXXX.png: plot of planet albedo, contrast, star flux, and planet flux
    '''
    import warnings
    warnings.filterwarnings('ignore')
    import picaso.justdoit as jdi
    
    import sys
    import os
    # Make directory to store run results:
    #os.system('mkdir '+savefiledirectory)
    
    # Set all terminal output to write to file:
    if record_terminal_output:
        f = open(savefiledirectory+"/terminal_output.txt", 'w')
        sys.stdout = f
    # Additional info for xarray (not used)
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
    PlanetCO = ConvertCtoOtoStr(PlanetCO)
    if type(pdict['planet_mh_str']) == str:
        PlanetMHStr = pdict['planet_mh_str']
    else:
        PlanetMHStr = ConvertPlanetMHtoCKStr(pdict['planet_mh_str'])

    if pdict['noTiOVO']:
        ck_db_name = pdict['local_ck_path'] + f'sonora_2020_feh{PlanetMHStr}_co_{PlanetCO}_noTiOVO.data.196'
    else:
        ck_db_name = pdict['local_ck_path'] + f'sonora_2020_feh{PlanetMHStr}_co_{PlanetCO}.data.196'
    print(ck_db_name)
    # Set opacity connection:
    opacity_ck = jdi.opannection(ck_db=ck_db_name, wave_range = specdict['wave_range'])
    
    # initialize model:
    pl = jdi.inputs(calculation= calculation, climate = True)
    
    ### set up planet:
    # input effective temperature
    pl.effective_temp(pdict['tint']) 
    # add gravity:
    if not pdict['gravity']:
        pl.gravity(radius=pdict['radius'], radius_unit=pdict['radius_unit'], 
            mass = pdict['mass'], mass_unit=pdict['mass_unit'])
    else:
        pl.gravity(gravity=pdict['gravity'], gravity_unit=pdict['gravity_unit'])
        
    # set up star:
    if 'star_filename' in sdict.keys():
        pl.star(opacity_ck, temp=None, metal=None, logg=None ,radius = sdict['star_radius'], radius_unit=u.Rsun,
             semi_major=pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], 
             filename=sdict['star_filename'], w_unit='um', f_unit='FLAM')
    else:
        pl.star(opacity_ck, temp = sdict['Teff'], metal = np.log10(sdict['mh']), logg = sdict['logg'], 
            radius = sdict['radius'], radius_unit = u.R_sun, 
            semi_major = pdict['semi_major'], semi_major_unit = pdict['semi_major_unit'], database = 'phoenix')
    
    # add phase:
    phase = pdict['phase']
    # If full phase:
    if phase == 0:
        # Use symmetry to speed up calculation.
        num_tangle = 1
        pdict['num_tangle'] = 1
    else:
        num_tangle = pdict['num_tangle']
    pl.phase_angle(phase=phase*np.pi/180, num_tangle=num_tangle, num_gangle=pdict['num_gangle'])
    
    # Add initial P(T) profile guess:
    if use_guillotpt:
        pt = pl.guillot_pt(pdict['Teq'], nlevel=cdict['nlevel'], T_int = pdict['tint'], 
                              p_bottom=cdict['climate_pbottom'], p_top=cdict['climate_ptop'])
    else:
        pt = user_supplied_ptprofile

    # initial PT profile guess:
    temp_guess = pt['temperature'].values 
    press_guess = pt['pressure'].values
    # Input climate params:
    nstr = np.array([0,cdict['nstr_upper'],cdict['nstr_deep'],0,0,0]) # initial guess of convective zones
    # Set up climate run inputs:
    pl.inputs_climate(temp_guess= temp_guess, pressure= press_guess, 
                  nstr = nstr, nofczns = cdict['nofczns'] , rfacv = cdict['rfacv'])
    print('starting climate run')
    # Compute climate:
    #noclouds = pl.climate(opacity_ck, save_all_profiles=True, with_spec=True)
    kzz = temp_guess*0+1e-500
    noclouds = pl.climate(opacity_ck, save_all_profiles=True,
                diseq_chem=True,on_fly=True, kz=kzz,chemeq_first=False,
                gases_fly=['CO','CH4','H2O','NH3','CO2',
                    'N2','HCN','H2','PH3','C2H2','C2H4','H2S',
                    'C2H6','Na','K','FeH'])

    # Set atm to climate run results:
    pl.atmosphere(df=noclouds['ptchem_df'])

    # Generate recommended molecules for clouds:
    from virga import justdoit as vj
    # pressure temp profile from climate run:
    temperature = noclouds['temperature']
    pressure = noclouds['pressure']
    # metallicity = pdict['mh'] #atmospheric metallicity relative to Solar
    metallicity_TEMP = 0 # mh must be 1 for this part
    mmw = 2.2
    # got molecules for cloud species:
    recommended, fig = vj.recommend_gas(pressure, temperature, 10**(metallicity_TEMP), 
                                   mmw, plot=True, return_plot = True)
    print('Virga recommended molecules:', recommended)
    import pickle
    pickle.dump(recommended, open(savefiledirectory+'/virga-recommended-molecules.pkl','wb'))
    
    # Save recommended plot for future inspection
    from bokeh.plotting import figure, output_file, save
    output_file(savefiledirectory+"/recomended-gasses.html")
    save(fig)

    sdict.update({'flux_unit':'ergs cm^-2 s^-1 cm^-1'})

    # Pickle info and save:
    pickle.dump([pl, noclouds], open(savefiledirectory+'/cloud-free-model.pkl','wb'))
    pickle.dump([pdict, sdict, cdict], open(savefiledirectory+'/cloud-free-model-inputs.pkl','wb'))

    # If you want to compute spectrum here (recommended)
    if compute_spectrum:
        ### Make cloud-free spectrum:
        wno, alb, fpfs, full_output, PlanetFlux, StarWNO, StarFlux = ComputeSpectrum(noclouds['ptchem_df'], 
                                                                  pdict, sdict, specdict, 
                                                                  calculation = 'planet',
                                                                  saveplots = True, 
                                                                  clouddict = None,
                                                                  savefiledirectory = savefiledirectory+'/cloud-free', 
                                                                  ComputePlanetFlux = True, 
                                                                  make4spectrumplot = True,
                                                                  AbundancesPlot = True, 
                                                                  MixingRatioPlot = True,
                                                                  PTplot = True
                                                                  )
        # dump it out:
        outdict = {'wavenumber':wno, 'albedo spectrum':alb,
                   'planet star contrast':fpfs, 'full_output':full_output,
                   'planet flux': PlanetFlux, 'star wavenumber':StarWNO, 'star flux':StarFlux}
        pickle.dump(outdict, 
                    open(savefiledirectory+'/cloud-free-spectrum-full-output-R'+str(specdict['R'])+'.pkl','wb'))
        from scipy.interpolate import interp1d
        func = interp1d(StarWNO,StarFlux,fill_value="extrapolate")
        ResampledStarFlux = func(wno)
        outdf = pd.DataFrame(data={'wavelength [um]':1e4/wno,
                                   'planet flux [ergs/cm2/s/cm]':PlanetFlux,
                                   'star flux [ergs/cm2/s/cm]':ResampledStarFlux,
                                   'albedo':alb,
                                   'planet-star contrast':fpfs }
                             )
        outdf.to_csv(savefiledirectory+'/cloud-free-spectrum-R'+str(specdict['R'])+'.csv', index=False, sep=' ')

    else:
        wno, alb, fpfs, full_output = ComputeSpectrum(noclouds['ptchem_df'], 
                                                                  pdict, sdict, specdict, 
                                                                  calculation = 'planet',
                                                                  saveplots = True, 
                                                                  clouddict = None,
                                                                  savefiledirectory = savefiledirectory+'/cloud-free', 
                                                                  ComputePlanetFlux = False, 
                                                                  make4spectrumplot = False,
                                                                  AbundancesPlot = True, 
                                                                  MixingRatioPlot = False,
                                                                  PTplot = True
                                                                  )
        noclouds['ptchem_df'].to_csv(savefiledirectory+'/cloud-free-atm-df.csv', index=False)
    if record_terminal_output:
        f.close()
    
    return pl, noclouds


def GenerateInitialReadMe(savefiledirectory,planettype,T_star,r_star,Teq_str,semi_major,radiusj,massj,
                   planet_mh,planet_mh_CtoO,phase,pdict,sdict,cdict,specdict,recommended):
    import time
    t = time.localtime()
    outtime = str(t.tm_year)+'-'+str(t.tm_mon)+'-'+str(t.tm_mday)+'-'+str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec)

    readme = f'''
    ReflectX Gas Giant Picaso model 
    -------------------------------
    generated {outtime} in python 3.8 
    Consult www.loganpearcescience/reflectx.html for more details 

    Model: 
     - Planet type: {planettype} 
     - Star T_eff [K]: {T_star} 
     - Star radius [Rsun]: {r_star} 
     - Planet Eq Temp [K]: {int(Teq_str)} 
     - Planet semi-major axis [au]: {semi_major} 
     - Planet radius [Rjup]: {radiusj} 
     - Planet mass [Mjup]: {massj} 
     - Planet metallicity: {planet_mh} 
     - Planet C/O ratio: {planet_mh_CtoO} 
     - Phase: {phase} 

    Picaso input parameters: 
    These dictionaries are in machine-readable format (pickle) in '/cloud-free-model-inputs.pkl' 
    To read them in, run: `pdict, sdict, cdict = pickle.load(open(path_to_model+'/cloud-free-model-inputs.pkl','rb'))` 
     - Planet Properties Dictionary: {pdict} 
     - Star Properties Dictionary: {sdict} 
     - Climate Run Parameters Dictionary: {cdict} 
     - Spectrum Calculation Dictionary: {specdict} 

    Virga-recommended condensation gases: {recommended} 

    DIRECTORY CONTENTS:
    ------------------

    Cloud-free full PICASO output located in '/cloud-free-model.pkl'
    To read it in: `model = pickle.load(open(path_to_model+'/cloud-free-model.pkl','rb'))`

    Cloud-free spectrum located in '/cloud-free-spectrum-full-output-R'+str(specdict['R'])': 
     - Machine readable pickle file: `spectrum_dictionary = pickle.load(open(path_to_model+'/cloud-free-spectrum-full-output-R'+str(specdict['R']+'.pkl'), 'rb'))`
     - Human and machine readable csv: `spectrum_df = pandas.read_csv(path_to_model+'/cloud-free-spectrum-full-output-R'+str(specdict['R'])+'.csv', delim_whitespace=True)`

    `/terminal-output.txt`: A text file containing terminal output during the run. Useful for assessing
     model convergence.

    Plots:

     - `4SpectrumPlot`: plot of planet albedo spectrum, planet contrast (Fp/Fs), stellar spectrum, and planet spectrum (Fp/Fs * Fs)
     - `Abundances`: molecular abundances of the 8 most abundant molecules as a function of pressure (altitude)
     - `PTprofile`: Pressure/Temperature profile from climate calculation
     - `recommended-gases.html`: plot of PT profile against gas condensation curves with the gases used in the spectrum identified with heavy lines
     - `mixing-ratios.html`: mixing ratios for every gas as a function of pressure

    '''
    k = open(savefiledirectory+'/readme.txt','w')
    k.write(readme)
    k.close()


def GenerateTerrestrialPlanetReadMe(savefiledirectory,planettype,T_star,r_star,Teq_str,semi_major,radiusj,massj,
                   planet_mh,planet_mh_CtoO,phase,pdict,sdict,cdict,specdict,recommended):
    import time
    t = time.localtime()
    outtime = str(t.tm_year)+'-'+str(t.tm_mon)+'-'+str(t.tm_mday)+'-'+str(t.tm_hour)+':'+str(t.tm_min)+':'+str(t.tm_sec)

    readme = f'''
    ReflectX Gas Giant Picaso model 
    -------------------------------
    generated {outtime} in python 3.8 
    Consult www.loganpearcescience/reflectx.html for more details 

    Model: 
     - Planet type: {planettype} 
     - Star T_eff [K]: {T_star} 
     - Star radius [Rsun]: {r_star} 
     - Planet Eq Temp [K]: {int(Teq_str)} 
     - Planet semi-major axis [au]: {semi_major} 
     - Planet radius [Rjup]: {radiusj} 
     - Planet mass [Mjup]: {massj} 
     - Planet metallicity: {planet_mh} 
     - Planet C/O ratio: {planet_mh_CtoO} 
     - Phase: {phase} 

    Picaso input parameters: 
    These dictionaries are in machine-readable format (pickle) in '/cloud-free-model-inputs.pkl' 
    To read them in, run: `pdict, sdict, cdict = pickle.load(open(path_to_model+'/cloud-free-model-inputs.pkl','rb'))` 
     - Planet Properties Dictionary: {pdict} 
     - Star Properties Dictionary: {sdict} 
     - Climate Run Parameters Dictionary: {cdict} 
     - Spectrum Calculation Dictionary: {specdict} 

    Virga-recommended condensation gases: {recommended} 

    DIRECTORY CONTENTS:
    ------------------

    Cloud-free full PICASO output located in '/cloud-free-model.pkl'
    To read it in: `model = pickle.load(open(path_to_model+'/cloud-free-model.pkl','rb'))`

    Cloud-free spectrum located in '/cloud-free-spectrum-full-output-R'+str(specdict['R'])': 
     - Machine readable pickle file: `spectrum_dictionary = pickle.load(open(path_to_model+'/cloud-free-spectrum-full-output-R'+str(specdict['R']+'.pkl'), 'rb'))`
     - Human and machine readable csv: `spectrum_df = pandas.read_csv(path_to_model+'/cloud-free-spectrum-full-output-R'+str(specdict['R'])+'.csv', delim_whitespace=True)`

    `/terminal-output.txt`: A text file containing terminal output during the run. Useful for assessing
     model convergence.

    Plots:

     - `4SpectrumPlot`: plot of planet albedo spectrum, planet contrast (Fp/Fs), stellar spectrum, and planet spectrum (Fp/Fs * Fs)
     - `Abundances`: molecular abundances of the 8 most abundant molecules as a function of pressure (altitude)
     - `PTprofile`: Pressure/Temperature profile from climate calculation
     - `recommended-gases.html`: plot of PT profile against gas condensation curves with the gases used in the spectrum identified with heavy lines
     - `mixing-ratios.html`: mixing ratios for every gas as a function of pressure

    '''
    k = open(savefiledirectory+'/readme.txt','w')
    k.write(readme)
    k.close()

def PlotMixingRatios(full_output,limit=50):
    import matplotlib
    molecules = full_output['weights'].keys()
    pressure = full_output['layer']['pressure']
    mixingratios = full_output['layer']['mixingratios']
    to_plot=mixingratios.max().sort_values(ascending=False)[0:limit].keys()
    
    cmap = matplotlib.colormaps.get_cmap('magma')
    cols = cmap(np.linspace(0,0.9,limit))
     
    fig, ax = plt.subplots(figsize = (6,6))
    for mol , c in zip(to_plot,cols):
        ind = np.where(mol==np.array(molecules))[0][0]
        f = ax.plot(mixingratios[mol],pressure, color=c, lw=3,
                    alpha=1.0, label = mol)
    ax.legend(loc=(1.01,0.0), ncols=2)  
    ax.set_xlim(1e-20,1e1)
    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_ylabel('Pressure [Bars]')
    ax.set_xlabel('Mixing Ratio [v/v]')
    ax.invert_yaxis()
    ax.grid(ls=':')
    
    return fig

def PlotCloud(full_output):    
    from scipy.interpolate import interp1d
    from matplotlib.colors import LogNorm, Normalize


    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True)

    ax = axes[0]
    w0 = full_output['layer']['cloud']['w0']
    im = ax.imshow(w0, aspect=2500, cmap='magma')
    ax.set_xlabel('Wavelength [um]')
    ax.set_title('Single Scattering \n Albedo')
    cax = fig.add_axes([0.145,0.0,0.23,0.04])
    plt.colorbar(mappable=im, cax=cax, orientation='horizontal')
    cax.set_xticks([0,0.25,0.5,0.75,1], labels = [0,0.25,0.5,0.75,1], rotation=45)

    n = 5
    xtickslocs = np.linspace(0,w0.shape[1], n)
    xticklabels = np.round(np.linspace(0.3,2,n),decimals=1)
    ax.set_xticks(xtickslocs, labels=xticklabels, rotation=45)
    ax.invert_yaxis()
    func = interp1d(np.linspace(-6,2, 1000), np.linspace(0,w0.shape[0],1000))
    yin = np.flip(np.array([2,1,0,-1,-2,-3,-4,-5,-6]))
    yticklocs = func(yin)
    yticklabels = 10.**yin
    yticklabels = ['{:.0e}'.format(y) for y in yticklabels]
    ax.set_yticks(yticklocs, labels=yticklabels)
    ax.set_ylabel('Pressure [bars]')
    

    ax = axes[1]
    opd = full_output['layer']['cloud']['opd']
    im = ax.imshow(opd+1e-60, aspect=2500, cmap='viridis', norm=LogNorm(vmin=1e-5, vmax=np.max(opd)))
    ax.invert_yaxis()
    ax.set_xticks(xtickslocs, labels=xticklabels, rotation=45)
    ax.set_xlabel('Wavelength [um]')
    ax.set_title('Cloud Optical \n Depth')
    cax = fig.add_axes([0.44,0.0,0.23,0.04])
    plt.colorbar(mappable=im, cax=cax, orientation='horizontal', extend='min')
    caxticks = np.arange(-5,np.floor(np.log10(np.max(opd)))+1,2)
    #caxticklabels = ['{:.0e}'.format(y) for y in caxticks]
    expo = [str(c).replace('.0','') for c in caxticks]
    caxticklabels = ['$10^{'+e+'}$' for e in expo]
    cax.set_xticks(10**caxticks, labels = caxticklabels, rotation=45)

    ax = axes[2]
    g0 = full_output['layer']['cloud']['g0']
    im = ax.imshow(g0+1e-60, aspect=2500, cmap='Greys', norm=Normalize(vmin=0, vmax=1))
    ax.set_xlabel('Wavelength [um]')
    ax.set_title('Asymmetry \n Parameter')
    ax.invert_yaxis()
    ax.set_xticks(xtickslocs, labels=xticklabels, rotation=45)
    cax = fig.add_axes([0.73,0.0,0.23,0.04])
    plt.colorbar(mappable=im, cax=cax, orientation='horizontal')
    cax.set_xticks([0,0.25,0.5,0.75,1], labels = [0,0.25,0.5,0.75,1], rotation=45)

    plt.tight_layout()
    
    return fig

def find_nearest_2d(array,value,axis=1):
    #small program to find the nearest neighbor in a matrix
    all_out = []
    for i in range(array.shape[axis]):
        ar , iar ,ic = np.unique(array[:,i],return_index=True,return_counts=True)
        idx = (np.abs(ar-value)).argmin(axis=0)
        if ic[idx]>1: 
            idx = iar[idx] + (ic[idx]-1)
        else: 
            idx = iar[idx]
        all_out+=[idx]
    return all_out

def cumsum(mat):
    """Function to compute cumsum along axis=0 to bypass numba not allowing kwargs in 
    cumsum 
    """
    new_mat = np.zeros(mat.shape)
    for i in range(mat.shape[1]):
        new_mat[:,i] = np.cumsum(mat[:,i])
    return new_mat

def PlotPhotonAttenuation(full_output, at_tau=0.5, igauss=0):
    wave = 1e4/full_output['wavenumber']

    dtaugas = full_output['taugas'][:,:,igauss]
    dtaucld = full_output['taucld'][:,:,igauss]*full_output['layer']['cloud']['w0']
    dtauray = full_output['tauray'][:,:,igauss]
    shape = dtauray.shape
    taugas = np.zeros((shape[0]+1, shape[1]))
    taucld = np.zeros((shape[0]+1, shape[1]))
    tauray = np.zeros((shape[0]+1, shape[1]))

    #comptue cumulative opacity
    taugas[1:,:]=cumsum(dtaugas)
    taucld[1:,:]=cumsum(dtaucld)
    tauray[1:,:]=cumsum(dtauray)

    pressure = full_output['level']['pressure']

    at_pressures = np.zeros(shape[1]) #pressure for each wave point

    ind_gas = find_nearest_2d(taugas, at_tau)
    ind_cld = find_nearest_2d(taucld, at_tau)
    ind_ray = find_nearest_2d(tauray, at_tau)

    at_pressures_gas = np.zeros(shape[1])
    at_pressures_cld = np.zeros(shape[1])
    at_pressures_ray = np.zeros(shape[1])

    for i in range(shape[1]):
        at_pressures_gas[i] = pressure[ind_gas[i]]
        at_pressures_cld[i] = pressure[ind_cld[i]]
        at_pressures_ray[i] = pressure[ind_ray[i]]

    
    fig,ax = plt.subplots()
    import matplotlib
    cmap = matplotlib.colormaps.get_cmap('viridis')
    colors = cmap(np.linspace(0.2,0.8,3))

    ax.plot(wave,at_pressures_gas, lw=3, color=colors[0], label='Gas Opacity')
    ax.plot(wave,at_pressures_cld, lw=3, color=colors[1], label='Cloud Opacity')
    ax.plot(wave,at_pressures_ray, lw=3, color=colors[2], label='Rayleigh Opacity')
    
    ax.legend(loc=(1.01,0.1), fontsize=15) 
    ax.set_yscale('log')
    ax.invert_yaxis()
    ax.set_ylabel('Pressure [bars]')
    ax.set_xlabel('Wavelength [um]')
    ax.set_ylim(top=1e-6)

    #finally add color sections 
    gas_dominate_ind = np.where((at_pressures_gas<at_pressures_cld) & (at_pressures_gas<at_pressures_ray))[0]
    cld_dominate_ind = np.where((at_pressures_cld<at_pressures_gas) & (at_pressures_cld<at_pressures_ray))[0]
    ray_dominate_ind = np.where((at_pressures_ray<at_pressures_cld) & (at_pressures_ray<at_pressures_gas))[0]

    gas_dominate = np.zeros(shape[1]) + 1e-8
    cld_dominate = np.zeros(shape[1]) + 1e-8
    ray_dominate = np.zeros(shape[1])+ 1e-8

    gas_dominate[gas_dominate_ind] = at_pressures_gas[gas_dominate_ind]
    cld_dominate[cld_dominate_ind] = at_pressures_cld[cld_dominate_ind]
    ray_dominate[ray_dominate_ind] = at_pressures_ray[ray_dominate_ind]
    
    from matplotlib.patches import Rectangle
    if len(gas_dominate) > 0  :
        band_x = np.append(np.array(wave), np.array(wave[::-1]))
        band_y = np.append(np.array(gas_dominate), np.array(gas_dominate)[::-1]*0+1e-8)
#         for i in gas_dominate_ind:
#             dw = np.abs(wave[i] - wave[i-1])
#             rect = Rectangle([wave[i],gas_dominate[i]], dw, gas_dominate[i]-1e6, color=colors[0], alpha=0.3)
#             ax.add_artist(rect)
        #ax.fill_between(wave[gas_dominate_ind],gas_dominate[gas_dominate_ind],1e-8, color=colors[0],alpha=0.3)
        #fig.patch(band_x,band_y, color=Colorblind8[0], alpha=0.3)
#     if len(cld_dominate) > 0  :
#         band_x = np.append(np.array(wave), np.array(wave[::-1]))
#         band_y = np.append(np.array(cld_dominate), np.array(cld_dominate)[::-1]*0+1e-8)
#         fig.patch(band_x,band_y, color=Colorblind8[3], alpha=0.3)
#     if len(ray_dominate) > 0  :
#         band_x = np.append(np.array(wave), np.array(wave[::-1]))
#         band_y = np.append(np.array(ray_dominate), np.array(ray_dominate)[::-1]*0+1e-8)
#         fig.patch(band_x,band_y, color=Colorblind8[6], alpha=0.3)

    return fig


def PlotCloudTau1DProfile(full_output):
    press = full_output['layer']['pressure']
    opd = np.mean(full_output['layer']['cloud']['opd'], axis=1)
    fig,ax = plt.subplots()
    ax.plot(opd, press)
#     try:
#         plt.axhline(y=press[H2OcrossedDict[str(p)][0]+1], ls=':', color='purple', label='H2O condensation region')
#         plt.axhline(y=press[H2OcrossedDict[str(p)][1]+1], ls=':', color='purple')
#     except:
#         pass

#     plt.axhline(y=press[S8crossedDict[str(p)][0]+1], ls=':', color='orangered', label='S8 condensation region')
#     plt.axhline(y=press[S8crossedDict[str(p)][1]+1], ls=':', color='orangered')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.invert_yaxis()
    #ax.legend(fontsize = 15)
    return fig

def PlotHeatmapTaus(out, R=0):
    """
    Plots a heatmap of the tau fields (taugas, taucld, tauray)

    Parameters
    ----------
    out : dict 
        full_ouput dictionary
    R : int 
        Resolution to bin to (if zero, no binning)
    """
    titles = ['Gas', 'Cloud', 'Rayleigh']
    nrow = 1
    ncol = 3 #at most 3 columns
    #fig = plt.figure(figsize=(6*ncol,4*nrow))
    fig, axes = plt.subplots(nrows = nrow, ncols = ncol, sharey = True, figsize=(10,4))
    for it, itau in enumerate(['taugas','taucld','tauray']):
        #ax = fig.add_subplot(nrow,ncol,it+1)
        ax = axes[it]
        tau_bin = []
        for i in range(out['full_output'][itau].shape[0]):
            if R == 0 : 
                x,y = out['wavenumber'], out['full_output'][itau][i,:,0]
            else: 
                from picaso.justdoit import mean_regrid
                x,y = mean_regrid(out['wavenumber'],
                                  out['full_output'][itau][i,:,0], R=R)
            tau_bin += [[y]]
        tau_bin = np.array(tau_bin)
        tau_bin[tau_bin==0]=1e-100
        tau_bin = np.log10(tau_bin)[:,0,:]
        X,Y = np.meshgrid(1e4/x,out['full_output']['layer']['pressure'])
        Z = tau_bin
        pcm=ax.pcolormesh(X, Y, Z,shading='auto',cmap='RdBu_r')
        #cbar=fig.colorbar(pcm, ax=ax)
        pcm.set_clim(-3.0, 3.0)
        #ax.set_title(itau)
        ax.set_title(titles[it])
        ax.set_yscale('log')
        ax.set_ylim([1e2,1e-3])
        if it == 0:
            ax.set_ylabel('Pressure(bars)')
        ax.set_xlabel('Wavelength(um)')
        #cbar.set_label('log Opacity')
    cax = fig.add_axes([1.005,0.2,0.015,0.7])
    cbar = plt.colorbar(mappable=pcm, cax=cax)
    cbar.set_label('log Opacity')
    plt.tight_layout()
        
    return fig


############### Estimating SNR as a function of time from ReflectX models"
def GetPhotonsPerSec(wavelength, flux, filt, distance, radius, primary_mirror_diameter,
                    return_ergs_flux_times_filter = False, Omega = None):
    ''' Given a spectrum with wavelengths in um and flux in ergs cm^-1 s^-1 cm^-2, convolve 
    with a filter transmission curve and return photon flux in photons/sec. 
    Following eqns in Chp 7 of my thesis
    
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
    import astropy.constants as const
    # energy in ergs:
    energy_in_ergs_per_photon_per_wavelength = const.h.cgs * const.c.cgs / wavelength # Eqn 7.14
    # Flux in photons/cm s cm^2: number of photons per area per sec per lambda:
    nphotons_per_wavelength = flux / energy_in_ergs_per_photon_per_wavelength # Eqn 7.15
    
    # Combine flux with filter curve:
    w = filt.wavelength*filt.wavelength_unit.to(u.um)
    t = filt.transmission
    # make interpolation function:
    ind = np.where((wavelength > min(w)) & (wavelength < max(w)))[0]
    # of spectrum wavelength and Flux in photons/cm s cm^2:
    from scipy.interpolate import interp1d
    f = interp1d(wavelength[ind], nphotons_per_wavelength[ind], fill_value="extrapolate")
    # interpolate the filter flux onto the spectrum wavelength grid:
    flux_on_filter_wavelength_grid = f(w)

    # multiply flux time filter transmission
    filter_times_flux = flux_on_filter_wavelength_grid * t # F_l x R(l) in eqn 7.16
    
    # Now sum:
    dl = (np.mean([w[j+1] - w[j] for j in range(len(w)-1)]) * u.um).to(u.cm)

    total_flux_in_filter = np.sum(filter_times_flux * dl.value) # Eqn 7.16
    
    
    area_of_primary = np.pi * ((0.5*primary_mirror_diameter).to(u.cm))**2

    #Total flux in photons/sec:
    total_flux_in_photons_sec = total_flux_in_filter * area_of_primary.value # Eqn 7.17
    
    if return_ergs_flux_times_filter:
        f = interp1d(wavelength[ind], flux[ind], fill_value="extrapolate")
        # interpolate the filter flux onto the spectrum wavelength grid:
        flux_ergs_on_filter_wavelength_grid = f(w)
        filter_times_flux_ergs = flux_ergs_on_filter_wavelength_grid * t
        
        return total_flux_in_photons_sec * (1/u.s), filter_times_flux_ergs, w
    return total_flux_in_photons_sec * (1/u.s)

def GetGuideStarMagForIasTableLookup(actual_star_magnitude):
    # Tables are available for these guide star mags:
    available_mags = np.array(['0', '2.5', '5', '7', '9', '10', '11',
                        '11.5', '12', '12.5', '13','13.5', '14', '14.5', '15'])
    available_mags = np.array([float(m) for m in available_mags])
    # find the one closest to actual guidestar mag:
    idx = (np.abs(available_mags - actual_star_magnitude)).argmin()
    table_guidestarmag = str(available_mags[idx]).replace('.0','')
    return table_guidestarmag

def Get_LOD_in_mas(central_wavelength, D):
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

def GetIasFromTable(guidestarmag, wfc, sep, pa, 
                   path_to_maps = '/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/GMagAO-X-noise/'):
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
    IasMap = fits.getdata(path_to_maps+f'varmap_{guidestarmag}_{wfc}.fits')
    center = [0.5*(IasMap.shape[0]-1),0.5*(IasMap.shape[1]-1)]
    dx = sep * np.cos(np.radians(pa + 90))
    dy = sep * np.sin(np.radians(pa + 90))
    if int(np.round(center[0]+dx, decimals=0)) < 0:
        return np.nan
    try:
        return IasMap[int(np.round(center[0]+dx, decimals=0)),int(np.round(center[1]+dy,decimals=0))]
    except IndexError:
        return np.nan

def GetIas(guidestarmag, wfc, sep, pa, wavelength, 
           path_to_maps = '/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/GMagAO-X-noise/'):
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
    from astropy.io import fits
    ## Load map:
    # IasMap = fits.getdata(path_to_maps+f'contrast_{guidestarmag}_{wfc}.fits')
    IasMap = fits.getdata(path_to_maps+f'varmap_{guidestarmag}_{wfc}.fits')
    
    # Look up Ias from table
    Ias = GetIasFromTable(guidestarmag, wfc, sep, pa, path_to_maps = path_to_maps)
    # Correct for differnce in wavelength between lookup table and filter wavelength:
    wavelength = wavelength.to(u.um)
    Ias = Ias * (((0.8*u.um/wavelength))**2).value
    if np.isnan(Ias):
        raise Exception('Sep/PA is outside noise map boundaries')
    else:
        return Ias
    
def ComputeSignalSNR(Ip, Is, Ic, Ias, Iqs, tau_as, tau_qs, deltat, strehl, QE, flux_in_core,
                           RN = None, Isky = None, Idc = None, texp = None):
    ''' Get S/N for a planet signal when speckle noise dominated. Using eqns in Chap 7 of
    my thesis.

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
    # When using the contrast Ias maps, Strelh applies to the numerator but not denom
    # (Males, priv. comm.)
    signal = Ip * deltat * strehl * QE * flux_in_core # Eqn 7.20
    
    Istar = Is * QE * flux_in_core
    photon_noise = Ic + Ias + Iqs
    atm_speckles = Istar * ( tau_as * (Ias**2 + 2*(Ic*Ias + Ias*Iqs)) )
    qs_speckles = Istar * ( tau_qs * (Iqs**2 + 2*Ic*Iqs) )
    
    Ipstar = Ip *  strehl * QE * flux_in_core
    
    sigma_sq_h = Istar * deltat * (photon_noise + atm_speckles + qs_speckles) + signal # Eqn 7.19
    
    if RN is not None:
        skyanddetector = Isky*deltat + Idc*deltat + (RN * deltat/texp)**2
        noise = np.sqrt(sigma_sq_h + skyanddetector)
    else:
        noise = np.sqrt(sigma_sq_h)
        
    return signal / noise



def GetSNR(wavelength, planet_contrast, 
           star_flux,
           primary_mirror_diameter, 
           planet_radius, star_radius, 
           distance, sep, wfc,
           filters, observationtime,
           Ic = 1e-20,
           Iqs = 1e-20,
           tau_as = 0.02, # sec, from Fig 10 in Males+ 2021
           tau_qs = 0.05,
           RN = None, Isky = None, Idc = None, texp = None, pa = None,
           MagAOX = False,
           path_to_maps = '/Users/loganpearce/Dropbox/astro_packages/myastrotools/myastrotools/GMagAO-X-noise/'
          ):
    
    '''Compute S/N as a function of time for a ReflectX model in specific filters.
    
    Args:
        wavelength (arr): array of wavelengths for planet and star spectrum
        planet_contrast (arr): array of planet flux values on wavelength grid
        star_flux (arr): array of star flux on wavelength grid
        primary_mirror_diameter (astropy unit object): mirrior diameter. Ex: 25.4*u.m
        planet_radius (astropy unit object): planet radius
        star_radius (astropy unit object): star radius
        distance (astropy unit object): distance to star
        sep (astropy unit object): planet-star separation
        wfc (str): wavefront control. Either 'lp' or 'si'
        filters (filter object): list of filter objects to compute snr for
        observationtime (arr): array of exposure times
    
    '''
    
    ## 1. Compute star's observed flux from model intensity and compute star's magnitude:
    star_flux = star_flux * ComputeOmega(distance, star_radius)
    star_mag = []
    for i in range(len(filters)):
        filt = filters[i]
        star_mag.append(GetStarMagnitude(star_wavelength, star_flux, filt))
        
    ## 2. Get planet flux from model contrast and distance-corrected star flux:
    planet_flux = star_flux * planet_contrast
    
    ## 3. Compute planet and star signal in photons/sec/lambda/D in each filter:
    planet_signal = []
    for filt in filters:
        planet_signal.append(GetPhotonsPerSec(wavelength, planet_flux, filt, distance, 
                                              planet_radius, primary_mirror_diameter).value)
    planet_signal = np.array(planet_signal)
    
    star_signal = []
    for filt in filters:
        star_signal.append(GetPhotonsPerSec(wavelength, star_flux, filt, distance, 
                                            star_radius, primary_mirror_diameter).value)
    star_signal = np.array(star_signal)
    
    ## 4. Get atmospheric speckle intensity I_as at planet location using maps from Males et al. 2021:
    # Find relevant table:
    star_gsm_for_Ias_tables = []
    for i,filt in enumerate(filters):
        star_gsm_for_Ias_tables.append(GetGuideStarMagForIasTableLookup(star_mag[i]))
    # Get location of planet signal:
    sep_au = sep.to(u.au)
    sep_mas = (sep_au.value/distance.value)*u.arcsec.to(u.mas)
    lods_in_mas = [Get_LOD_in_mas(f.central_wavelength*f.wavelength_unit, primary_mirror_diameter) 
                   for f in filters]
    sep_lods = [sep_mas/lod for lod in lods_in_mas]
    if pa is not None:
        pass
    else:
        pa = 90 # deg
    # Lookup Ias:
    Ias = []
    for i in range(len(filters)):
        wavelength = filters[i].central_wavelength*filters[i].wavelength_unit
        Ias.append(GetIas(star_gsm_for_Ias_tables[i], wfc, sep_lods[i], pa, wavelength))
    Ias = np.array(Ias)
    
    ## 5. Compute throughput:
    # 5a. Get QE (encompasses telescope and instrument throughput):
    if not MagAOX:
        QE = [0.1]*len(filters)
    else:
        QE = [0.13, 0.12, 0.06, 0.04, 0.1, 0.1]
    # 5b. Get Strehl ratio:
    strehl_table = pd.read_table(path_to_maps+f'strehl_{wfc}.txt', 
                                 delim_whitespace=True, names=['mag','strehl'])
    from scipy.interpolate import interp1d
    strehl_table2 = strehl_table.drop([4])
    strehl_table2 = strehl_table2.reset_index(drop=True)
    available_mags = np.array([float(m) for m in strehl_table2['mag']])
    func = interp1d(available_mags, strehl_table2['strehl'])
    strehls = []
    for i in range(len(filters)):
        strehl = func(star_mag[i])
        # scale for wavelength from strehl at 800 nm to filter central wavelength:
        wavelength = filters[i].central_wavelength*filters[i].wavelength_unit
        wavelength = wavelength.to(u.um)
        strehl = strehl * (((0.8*u.um/wavelength))**2).value
        strehls.append(strehl)
    # 5c. Amount of star flux in Airy core:
    star_flux_in_Airy_core_multiple = np.pi/4
    
    ## 6. Compute signal to noise ratios
    if type(observationtime) == np.ndarray:
        # For an array of times:
        all_snrs = []
        for i in range(len(filters)):
            snrs = []
            for t in observationtime:
                snrs.append(ComputeSignalSNR(planet_signal[i], star_signal[i], 
                                             Ic, Ias[i], Iqs, tau_as, tau_qs, t,
                                             strehls[i], QE[i], star_flux_in_Airy_core_multiple,
                                   RN = None, Isky = None, Idc = None, texp = None))
            all_snrs.append(snrs)
        return all_snrs, planet_signal, star_signal
    else:
        # for a single time:
        snrs = []
        for i in range(len(filters)):
            snrs.append(ComputeSignalSNR(planet_signal[i], star_signal[i], Ic, Ias[i], Iqs, tau_as, tau_qs, 
                                observationtime, strehl[i], QE[i], star_flux_in_Airy_core_multiple,
                                RN = RN, Isky = Isky, Idc = Idc, texp = texp))
        return snrs