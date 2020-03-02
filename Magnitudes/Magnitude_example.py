from obspy.core import read

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit,minimize

from mtspec import mtspec
import matplotlib.pyplot as plt

import mag_functions as magfn

event_locs=pd.read_csv(
    "../../../OneDrive - University of Bristol/1-Coding/5-Example_code/Magnitudes/Data/NewOllertonEvents.csv", usecols=['ID', 'Date', 'Time', 'Lat', 'Lon', 'Depth', 'Mag'])
stations=pd.read_csv(
    "../../../OneDrive - University of Bristol/1-Coding/5-Example_code/Magnitudes/Data/NOL_stations.csv")
nol_mags=pd.read_csv('../../../OneDrive - University of Bristol/1-Coding/5-Example_code/Magnitudes/Data/NOL_mags.csv')
nol_mags=nol_mags.set_index(['ID'])

# Define some parameters
vel = 1970
rho = 2400
R = 0.6
F = 2
win_len = 2.

example_events = [20140718044958, 20140822015232, 20140910165727, 20140717144824]

plt.figure(figsize=[6, 4])
for event in example_events:
    for station in ['NOLF']:

        name = 'Data/NOLF/GB.%s.%s.HHE.sac' % (station, event)
        st = read(name)
        name = 'Data/NOLF/GB.%s.%s.HHN.sac' % (station, event)
        st += read(name)

        s_arrival = st[0].stats.sac['t0']

        print(st)

        st_eq = st.copy()
        start = st[0].stats.starttime

        st_eq = st_eq.slice(start + s_arrival - 0.2, start + s_arrival + win_len)
        st_eq = st_eq.taper(type='cosine', max_percentage=0.05)

        delta = st[0].stats.delta

        # Use MTSPEC to produce an amplitude spectrum.
        powerE, freq = mtspec(st_eq[0].data, delta, 2, nfft=501)
        powerN, freq = mtspec(st_eq[1].data, delta, 2, nfft=501)

        power = np.zeros(len(powerE))
        for i in range(0, len(powerE)):
            power[i] = (powerE[i] + powerN[i]) / 2

        amp = np.sqrt(power)

        # Get the hypocentral distance using the event and station catalogue
        event_loc = event_locs[event_locs['ID'] == event]
        station_loc = stations[stations['Code'] == station]

        epi_distance = magfn.latlong_distn(event_loc.Lat.values[0], event_loc.Lon.values[0],
                                           station_loc.Latitude.values[0], station_loc.Longitude.values[0])
        hypo_distance = np.sqrt(epi_distance ** 2 + (event_loc.Depth.values[0] / 1000) ** 2) * 1000
        tt=hypo_distance/vel

        # Use the low level plateau to derive omega 0 and subsequently M0 and MW
        omega_0 = np.mean(amp[1:10])

        M_0 = ((4. * np.pi * rho * (vel ** 3) * hypo_distance * (omega_0)) / (R * F))
        M_W = ((2. / 3.0) * np.log10(M_0)) - 6.0

        # Interpolate into log space before fitting spectral fitting
        amp_log = np.log10(amp)
        freq_log = np.log10(freq)

        freq_log_interp = np.arange(freq_log[1], np.max(freq_log), 0.01)
        amp_log_interp = np.interp(freq_log_interp, freq_log, amp_log)

        amp_ma = magfn.moving_average(amp_log_interp[1:], n=10)
        freq_ma = magfn.moving_average(freq_log_interp[1:], n=10)

        amp_interp = 10 ** (amp_ma)
        freq_interp = 10 ** (freq_ma)

        # Create array which is used in Brune function
        for kappa in [0.035]:
            tt_array = np.linspace(tt, tt, len(freq_interp))
            omega_0_array = np.linspace(omega_0, omega_0, len(freq_interp))
            kappa_array = np.linspace(kappa, kappa, len(freq_interp))
            X = [tt_array, freq_interp, omega_0_array, kappa_array]

            # Use curvefit to derive Q and fc from brune spectrum.
            p0 = (50, 15)
            popt, pcov = curve_fit(magfn.brune_kappa, X, amp_ma, p0, bounds=([50, 0], [300, 100]))
            print('Event %s; Mag %s,Q %s; fc%s' % (event, nol_mags.loc[event, 'MW'], popt[0], popt[1]))

            brunekappa = 10 ** magfn.brune_kappa(X, popt[0], popt[1])

            plt.plot(freq_interp, amp_interp, label="$M_W$=%2.1f, $f_c$=%2.0fHz" % (nol_mags.loc[event, 'MW'], popt[1]))

            plt.loglog(freq_interp, brunekappa, 'k--', lw=1.5, label='')

plt.xlim(0.5, 50)
plt.ylim(1e-11, 5e-6)
plt.grid()
plt.xlabel('Frequency (Hz)', size=14)
plt.ylabel('Amplitude (m)', size=14)
plt.legend(loc=3, fontsize='small')
# plt.savefig('Images/Example_kappa_0_060220.png',dpi=300)
plt.show()