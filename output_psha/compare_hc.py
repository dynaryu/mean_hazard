import pandas as pd
import re
import numpy as np
import matplotlib.pyplot as plt

def read_hcurve(hazard_curve):
    with open(hazard_curve) as f:
        first_line = f.readline()
        try:
            inv_time = re.search('time=(.+?), imt', first_line).group(1)
        except:
            print(re.search('time=(.+?), imt', first_line))
        else:
            inv_time = float(inv_time)
        a = pd.read_csv(hazard_curve, skiprows=1)
        idx = re.search('hazard_curve-mean-PGA_(.+?).csv', hazard_curve).group(1)
        pga = [float(x.split('poe-')[1]) for x in a.columns if 'poe' in x]
        # convert to annual prob. exceedance
        poe = a.iloc[:, 3:].values
        poe_1yr = 1.0 - np.exp(1/inv_time*np.log(1-poe))

    return pga, poe_1yr


pga_c, poe_c = read_hcurve('./hazard_curve-mean-PGA_1556.csv')
pga_e, poe_e = read_hcurve('../output_epsha/hazard_curve-mean-PGA_1559.csv')

plt.figure()
plt.semilogy(pga_c, poe_c[0, :], label='classical')
plt.semilogy(pga_e, poe_e[0, :], '--', label='event-based')
plt.legend(loc=1)
plt.xlabel('PGA(g')
plt.ylabel('AEP')
plt.savefig('comp_hc1.png')


