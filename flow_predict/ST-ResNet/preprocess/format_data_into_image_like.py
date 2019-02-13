#!/usr/bin/env python
#coding:utf-8
#author:zhujianqi

import sys
import numpy as np

def load_geo(filename):
    geo_index = {}
    with open(filename) as f:
        index = 0
        for line in f:
            line = line.strip('\r\n')
            geo_index[line] = index
            index += 1
    return geo_index

seq_lens = [3, 1, 1] # [length of closeness, length of period, length of trend]
min_indexes = [30, 74, 86]
#closeness_range = range(30, 74) 
#period_range = range(74, 86)
#trend_range = range(86, 94) 
extend_range = range(4, 30) 

def main():
    geo_index = load_geo("")
    img_len = len(geo_index)
    day_hour_feature = {}
    for line in sys.stdin:
        line = line.strip('\r\n')
        info = line.split('\t')
        geo, day, hour = info[0], info[1], info[4]

        if geo not in geo_index:
            continue

        img_index = geo_index[geo]

        day_hour = "%s-%s" % (day, hour)
        if day_hour not in day_hour_feature:
            day_hour_feature[day_hour] = [[] for i in range(5)]
            for i, seq_len in enumerate(seq_lens):
                day_hour_feature[day_hour][i] = [[0.0 for k in range(2*img_len)] for j in range(seq_len)]
            day_hour_feature[day_hour][3] = [(float(info[ele]) if info[ele] != "NULL" else 0.0)  for ele in extend_range] 
            day_hour_feature[day_hour][4] = [0.0 for k in range(2*img_len)]
    
        for i, seq_len in enumerate(seq_lens):
            for j in range(seq_len):
                day_hour_feature[day_hour][i][j][img_index] = float(info[min_indexes[i]+j*2]) if info[min_indexes[i]+j*2] != "NULL" else 0.0
                day_hour_feature[day_hour][i][j][img_index+img_len] = float(info[min_indexes[i]+j*2+1]) if info[min_indexes[i]+j*2+1] != "NULL" else 0.0
    
        day_hour_feature[day_hour][4][img_index] = float(info[2]) if info[2] != "NULL" else 0.0
        day_hour_feature[day_hour][4][img_index+img_len] = float(info[3]) if info[3] != "NULL" else 0.0

    num_cnt = {}
    for key, value in day_hour_feature.items():
        #sys.stderr.write("%s\n" % len(value[3])) 
        for i, seq_len in enumerate(seq_lens):
            for j in range(seq_len):
                for k in range(len(value[i][j])):
                    ele = int(value[i][j][k])
                    if ele not in num_cnt:
                        num_cnt[ele] = 0
                    num_cnt[ele] += 1
        for ele in value[4]:
            ele = int(ele)
            if ele not in num_cnt:
                num_cnt[ele] = 0
            num_cnt[ele] += 1
        print(key, value)
    for k,v in sorted(num_cnt.items(), key=lambda x:x[0]):
        sys.stderr.write('k=%s, v=%s\n' % (k, v))

if __name__ == "__main__":
    main()

