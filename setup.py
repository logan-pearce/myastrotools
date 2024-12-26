from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()

setup(name='myastrotools',
      version='0.3',
      description='python tools for my astro research',
      url='https://github.com/logan-pearce/myastrotools',
      author='Logan Pearce',
      author_email='loganpearce55@gmail.com',
      license='MIT',
      packages=['myastrotools'],
      install_requires=['numpy','scipy','astropy','matplotlib'],
      dependency_links=['https://github.com/logan-pearce/myastrotools/tarball/master#egg=package-1.0'],
      package_data={'myastrotools': ['El-Badry-all_columns_catalog.fits','model_spectra/*mod*.fits',
                    'model_spectra/calspec_model_info.csv'
                    'isochrones/*','filter_curves/*']},
      include_package_data=True,
      zip_safe=False)
