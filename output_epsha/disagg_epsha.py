import pandas as pd
import os
import sys
import re
import numpy as np
import matplotlib.pyplot as plt
import socket

# see https://groups.google.com/g/openquake-users/c/VfBIx4kju3A/m/-10tppAAqjcJ
if 'gadi' in socket.gethostname():
    PATH = '/home/547/hxr547/Projects/demos/hazard/mean_hazard'
else:
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


def get_index_matching_string(line_str, _str):

    try:
        idx = [i for i, x in enumerate(line_str) if _str in x][0]
    except IndexError:
        print('Something wrong')
    else:
        return line_str[idx]


def get_disagg(disagg_file):
    with open(disagg_file) as f:
        first_line = f.readline().split(', ')
        # inv_time 
        inv_time = get_index_matching_string(first_line, 'investigation_time')
        inv_time = float(re.search('\d+\.\d+', inv_time).group(0))

        # imt
        #imt = get_index_matching_string(first_line, 'imt')
        #imt = re.search('imt\=\w+')

        # poe
        poe_target = get_index_matching_string(first_line, 'poe')
        poe_target = float(re.search('(\d+\.\d+)', poe_target).group(0))
        #try:
        #    inv_time = re.search('time=(.+?), imt', first_line).group(1)
        #except:
        #    print(re.search('time=(.+?), imt', first_line))
        #else:
        #    inv_time = float(inv_time)
        a = pd.read_csv(disagg_file, skiprows=2, names=['lon', 'lat', 'mag', 'poe'])

        # POE(x > x | M, T) -> P(M | x>x)
        poe_cal =  1 - (1 - a['poe']).prod()
        # convert to rate 
        #nu = (-1 / 50) * log (1 - P)
        a['rate_ex'] = -1.0 / inv_time * np.log(1 - a['poe'])
        total_rate_of_ex = -1.0 / inv_time * np.log (1.0 - poe_cal)
        a['norm_rate'] = a['rate_ex'] / total_rate_of_ex
    return a, poe_target

def main(jid):

    # event based PSHA
    #jid = 1567
    ruptures = pd.read_csv(os.path.join(PATH, f'output_epsha/ruptures_{jid}.csv'), index_col=0, skiprows=1)
    events = pd.read_csv(os.path.join(PATH, f'./output_epsha/events_{jid}.csv'), index_col=0)
    gmf = pd.read_csv(os.path.join(PATH, f'./output_epsha/gmf-data_{jid}.csv'), index_col=0)

    hazard_curve = os.path.join(PATH, f'output_epsha/hazard_curve-mean-PGA_{jid}.csv')
    pga_e, poe_e, inv_time = get_hcurve(hazard_curve)

    # classical mean hazard curve
    hazard_curve = os.path.join(PATH, f'output_epsha/cl/hazard_curve-mean-PGA_{jid+1}.csv')
    pga_c, poe_c, inv_time = get_hcurve(hazard_curve)

    # classical disagg results
    disagg_file = os.path.join(PATH, 'output/rlz-0-PGA-sid-0-poe-0_Mag_Lon_Lat_1.csv')
    df, poe_target = get_disagg(disagg_file)
    sdf = df.loc[df.poe > 0].copy()
    sdf['lon'] = sdf['lon'].round(decimals=2)
    sdf['lat'] = sdf['lat'].round(decimals=2)
    _file = os.path.join(PATH,'output/disagg_table_classical.csv')
    sdf.loc[sdf.norm_rate.sort_values(ascending=False).index].to_csv(_file)

    # disagg mean hazard curve
    hazard_curve = os.path.join(PATH, f'output/hazard_curve-mean-PGA_1.csv')
    pga_d, poe_d, inv_time = get_hcurve(hazard_curve)

    # compare hazard curves
    plt.figure()
    plt.semilogy(pga_c, poe_c[0, :], label='OQ_classical')
    plt.semilogy(pga_e, poe_e[0, :], label='OQ_event-based')
    plt.semilogy(pga_d, poe_d[0, :], label='OQ_disagg', marker='o', linestyle='--')
    plt.xlabel('PGA(g')
    plt.ylabel(f'POE in {inv_time} years')
    plt.legend(loc=1)
    plt.savefig(f'comp_hc_{jid}.png', dpi=200)

    # computing disaggregation
    events['gmv_PGA'] = gmf['gmv_PGA']
    #events['trt'] = events['rup_id'].apply(lambda x: ruptures.loc[x, 'trt'])
    events['mag'] = events['rup_id'].apply(lambda x: ruptures.loc[x, 'mag'])
    events['lon'] = events['rup_id'].apply(lambda x: ruptures.loc[x, 'centroid_lon'])
    events['lat'] = events['rup_id'].apply(lambda x: ruptures.loc[x, 'centroid_lat'])

    ses_per_logic_tree_path = len(events['ses_id'].value_counts())
    num_ex = np.array([(events['gmv_PGA'] > x).sum() for x in pga_e])
    #poes = 1 - numpy.exp(- num_exceeding / ses_per_logic_tree_path) 
    poes = 1 - np.exp(-1.0 * num_ex / ses_per_logic_tree_path)

    # compare poes vs poe_e
    plt.figure()
    #plt.semilogy(pga_c, poe_c[0, :], label='OQ_classical')
    plt.semilogy(pga_e, poe_e[0, :], label='OQ_event-based')
    plt.semilogy(pga_e, poes, label='extracted', marker='o', linestyle='--')
    plt.xlabel('PGA(g')
    plt.ylabel(f'POE in {inv_time} years')
    plt.legend(loc=1)
    plt.savefig(f'comp_hc_extract_{jid}.png', dpi=200)

    #nu = (-1 / T) * log (1 - P)
    #annual_rate[key] = -1.0 * np.log(1 - prob_ex) / inv_time
    #weights[key] = len(grp) / len(events)
    #weights[key] = 0.5

    pga_target = np.interp(poe_target, poe_e[0, :][::-1], pga_e[::-1])

    # filter exceeding pga_target
    sel = events.loc[events['gmv_PGA'] > pga_target].copy()

    lon_bin = [-1.298738625980141, -1.0988787786490142, -0.8990189313178873, -0.6991590839867605, -0.4992992366556337, -0.29943938932450687, -0.09957954199338004, 0.10028030533774679, 0.3001401526688736, 0.5000000000000004, 0.6998598473311273, 0.8997196946622539, 1.099579541993381, 1.299439389324508, 1.4992992366556346, 1.6991590839867612, 1.8990189313178882, 2.0988787786490155, 2.2987386259801417]
    lat_bin = [-2.39864, -2.198791111111111, -1.998942222222222, -1.799093333333333, -1.599244444444444, -1.399395555555555, -1.199546666666666, -0.9996977777777772, -0.7998488888888883, -0.5999999999999993, -0.40015111111111035, -0.20030222222222138, -0.00045333333333241743, 0.19939555555555655, 0.3992444444444455, 0.5990933333333345, 0.7989422222222234, 0.9987911111111124, 1.1986400000000015]
    mag_bin = [5.0, 6.0, 7.0, 8.0]

    sel['mag_bin'] = pd.cut(sel.mag, mag_bin)
    sel['lat_bin'] = pd.cut(sel.lat, lat_bin)
    sel['lon_bin'] = pd.cut(sel.lon, lon_bin)


    counted = sel.groupby(['mag_bin', 'lat_bin', 'lon_bin']).size()
    df_counted = counted.reset_index()
    df_counted.rename(columns={0: 'count'}, inplace=True)

    # compute P(M=m | x > x)
    df_counted['PM'] = df_counted['count'] / df_counted['count'].sum()

    _file = os.path.join(PATH,'output/disagg_table_event.csv')
    df_counted.loc[df_counted['PM'].sort_values(ascending=False).index].to_csv(_file)


if __name__=="__main__":
    jid = int(sys.argv[1])
    main(jid)
