VLA Site Forecast
=================
A weather forecasting system for the Very Large Array (`VLA`_) site in the
Plains of San Augustin, NM.  Forecasts, plots, and reports are created for:

* Basic weather properties such as near-surface wind speed, gusts, and
  precipitation.
* Cloud coverage and cloud optical depth.
* Observing efficiency time given opacity, sky brightness temperature, and
  cloud coverage.
* Astronomical phase stability predicted from a machine learning model trained
  on NWP forecast data and measured values of the phase RMS up to the forecast
  prediction window.

The primary numerical weather prediction sources are the NOAA's Global
Forecasting System (`GFS`_) and High-Resolution Rapid Refresh (`HRRR`_).  These
are queried using the free `open-meteo`_ API for single-location time series
data and, when needed, `Herbie`_ for cut-out maps of the full-frame CONUS data.
The images are primarily for estimating cloud properties and coverage.

Documentation on how to install and use this package may be found at
https://vla_forecast.readthedocs.io . Atmospheric radiative transfer is
performed using Scott Paine's code AM via the Python wrapper ``amwrap``.

Installation
------------
Currently only Unix-like operating systems (i.e., Linux and macOS) are
supported. Building the depdencies `AM`_ and ``amwrap`` requires GNU Make and a
C compiler, such as GCC. The parallel version of AM requires a C compiler with
OpenMP support.

To install ``site_forecast``, run the following from the command line:

.. code-block:: bash

   pip install git+https://github.com/autocorr/site_forecast.git

or alternatively:

.. code-block:: bash

   git clone https://github.com/autocorr/site_forecast.git
   cd site_forecast
   pip install .

Quickstart
----------
Forecasts can be generated using the ``Forecast`` class.

.. code-block:: python

   from site_forecast import Forecast
   fc = Forecast()
   print(fc)
   fc.create_plots()
   fc.save()

License
-------
This software is authored by Brian Svoboda copyright 2025 and released under the
GNU General Public License Agreement Version 3 (GPLv3). The full text of the
license is supplied in the ``LICENSE`` file included with the software.

AM is authored by Scott Paine of the Smithsonian Astrophysical Observatory.
The AM software is a work of the United States and may be used freely, with
attribution and credit to the Smithsonian Astrophysical Observatory. The
program is intended for educational, scholarly or research purposes.

.. _VLA: https://public.nrao.edu/telescopes/vla/
.. _GFS: https://www.emc.ncep.noaa.gov/emc/pages/numerical_forecast_systems/gfs.php
.. _HRRR: https://rapidrefresh.noaa.gov/hrrr/
.. _open-meteo: https://open-meteo.com/
.. _Herbie: https://herbie.readthedocs.io/en/stable/
.. _Scott Paine: https://lweb.cfa.harvard.edu/~spaine/am/index.html
.. _AM: https://zenodo.org/records/13748403
