import pandas as pd
import os
import re
import numpy as np
import matplotlib.pyplot as plt

PATH = '/home/hyeuk/openquake/src/oq-master/demos/hazard/Disaggregation'

def get_hcurve(hazard_curve):
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
        ae = -1.0 * np.log(1-poe)/inv_time
        poe_1yr = 1.0 - np.exp(-ae*1.0)
    return pga, poe, inv_time


# event based PSHA
jid = 1561
ruptures = pd.read_csv(os.path.join(PATH, f'output_epsha/ruptures_{jid}.csv'), index_col=0, skiprows=1)
events = pd.read_csv(os.path.join(PATH, f'./output_epsha/events_{jid}.csv'), index_col=0)
gmf = pd.read_csv(os.path.join(PATH, f'./output_epsha/gmf-data_{jid}.csv'), index_col=0)

hazard_curve = os.path.join(PATH, f'output_epsha/hazard_curve-mean-PGA_{jid}.csv')
pga_e, poe_e, inv_time = get_hcurve(hazard_curve)


# classical mean hazard curve
hazard_curve = os.path.join(PATH, 'output_psha/hazard_curve-mean-PGA_1556.csv')
pga_c, poe_c, inv_time = get_hcurve(hazard_curve)

events['gmv_PGA'] = gmf['gmv_PGA']
events['trt'] = events['rup_id'].apply(lambda x: ruptures.loc[x, 'trt'])

ses_per_logic_tree_path = len(events['ses_id'].value_counts())

num_ex = np.array([(events['gmv_PGA'] > x).sum() for x in pga_e])

#poes = 1 - numpy.exp(- num_exceeding / ses_per_logic_tree_path) 
poes = 1-np.exp(-np.array(num_ex)/ses_per_logic_tree_path)
#nu = (-1 / T) * log (1 - P)
#annual_rate[key] = -1.0 * np.log(1 - prob_ex) / inv_time
#weights[key] = len(grp) / len(events)
#weights[key] = 0.5

plt.figure()
plt.semilogy(pga_c, poe_c[0, :], label='OQ_classical')
plt.semilogy(pga_e, poe_e[0, :], label='OQ_event-based')
plt.semilogy(pga_e, poes, label='extracted', marker='o')
plt.xlabel('PGA(g')
plt.ylabel(f'POE in {inv_time} years')
plt.legend(loc=1)
plt.savefig('comp_hc_OQ_vs_extract.png', dpi=200)


