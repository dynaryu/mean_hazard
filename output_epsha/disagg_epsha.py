import pandas as pd
import os

PATH = '/home/hyeuk/openquake/src/oq-master/demos/hazard/Disaggregation'
jid = 1559

ruptures = pd.read_csv(os.path.join(PATH, f'output_epsha/ruptures_{jid}.csv'), index_col=0, skiprows=1)

events = pd.read_csv(os.path.join(PATH, f'./output_epsha/events_{jid}.csv'), index_col=0)

gmf = pd.read_csv(os.path.join(PATH, f'./output_epsha/gmf-data_{jid}.csv'), index_col=0)

# classical mean hazard curve
hazard_curve = os.path.join(PATH, 'output_psha/hazard_curve-mean-PGA_1556.csv')
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


events['gmv_PGA'] = gmf['gmv_PGA']
events['trt'] = events['rup_id'].apply(lambda x: ruptures.loc[x, 'trt'])

annual_rate = {}
weights = {}
for key, grp in events.groupby('trt'):
    num_ex = []
    for x in pga:
        num_ex.append((grp['gmv_PGA'] > x).sum())

    prob_ex = np.array(num_ex)/len(grp)
    #nu = (-1 / T) * log (1 - P)
    annual_rate[key] = -1.0 * np.log(1 - prob_ex) / inv_time
    weights[key] = len(grp) / len(events)
    #print(key)
    #print(annual_rate_exceedance)

weighted = np.zeros_like(annual_rate[list(annual_rate.keys())[0]])
for key, value in annual_rate.items():
    weighted += weights[key] * value

POE_weighted = 1 - np.exp(-weighted * inv_time)
