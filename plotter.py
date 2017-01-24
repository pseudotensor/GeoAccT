# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 15:04:25 2016

@author: McKinney
"""
import matplotlib.pylab as pl
import re
import sys
import numpy as np
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Plot experimental results for GeoAcc project.")
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--name', nargs=2)
    args = ['--search']
    args = parser.parse_args(args)

    if not args.search:
        fin1 = open(args.name[0], 'r')
        fin2 = open(args.name[1], 'r')
        pattern = re.compile(r'loss:\s([0-9]+\.[0-9]+)', re.MULTILINE)
        cost1 = '\n'.join(fin1.readlines()[:50])
        cost1 = re.findall(pattern, cost1)
        fin1.close()
        cost1 = [float(x) for x in cost1]
        cost2 = '\n'.join(fin2.readlines()[:50])
        cost2 = re.findall(pattern, cost2)
        fin2.close()
        cost2 = [float(x) for x in cost2]
        t = np.r_[0: len(cost1)]
        pl.plot(t, cost1, 'r-')
        pl.plot(t, cost2, 'b-')
        pl.legend(['$f_1$', '$f_2$'])
        pl.savefig(sys.argv)
    else:
        fin = open('search.sh')
        script = fin.read()


        def read(param):
            pat_lr = re.compile(param + "\s+in(.*)$", re.MULTILINE)
            lr = re.findall(pat_lr, script)[0].strip().split()
            lr = [float(x.strip('"')) for x in lr]
            return lr


        lr = read('lr')
        var = read('var')
        damp = read('damp')
        '''
        for a in lr:
            for b in var:
                for c in damp:
                    log_geo = 'results/log_geo_lr_' + str(a) + '_var_' + str(b) + '_damp_' + str(c)
                    log_ng = 'results/log_ng_lr_' + str(a) + '_var_' + str(b) + '_damp_' + str(c)
                    tcost_pat = re.compile('train.*cost:\s([0-9]+\.[0-9]+)', re.MULTILINE)
                    fin_geo = open(log_geo, 'r')
                    tcost_geo = re.findall(tcost_pat, fin_geo.read())
                    fin_geo.close()
                    fin_ng = open(log_ng, 'r')
                    tcost_ng = re.findall(tcost_pat, fin_ng.read())
                    fin_ng.close()
                    tcost_geo = [float(x) for x in tcost_geo]
                    tcost_ng = [float(x) for x in tcost_ng]
                    pl.figure()
                    pl.plot(np.r_[0:len(tcost_geo)], tcost_geo, 'r-')
                    pl.plot(np.r_[0:len(tcost_ng)], tcost_ng, 'b-')
                    pl.legend(['geo','ng'])
                    pl.xlabel('iter')
                    pl.ylabel('training cost')
                    pl.savefig('train_cost_lr_' + str(a) + '_var_' + str(b) + '_damp_' + str(c) + '.png')
                    pl.title('lr_' + str(a) + '_var_' + str(b) + '_damp_' + str(c))
                    pl.close()
        '''
        fout_geo = open('survey_geo.txt', 'w')
        fout_ng = open('survey_ng.txt', 'w')
        for a in lr:
            for b in var:
                for c in damp:
                    log_geo = 'results/log_geo_lr_' + str(a) + '_var_' + str(b) + '_damp_' + str(c)
                    log_ng = 'results/log_ng_lr_' + str(a) + '_var_' + str(b) + '_damp_' + str(c)
                    loss_pat = re.compile('average\straining\sloss:\s([0-9]+\.[0-9]+)', re.MULTILINE)
                    fin_geo = open(log_geo, 'r')
                    fin_ng = open(log_ng, 'r')
                    loss_geo = re.findall(loss_pat, fin_geo.read())
                    if not loss_geo:
                        loss_geo = float('nan')
                    else:
                        loss_geo = float(loss_geo[0].strip())
                    fin_geo.close()
                    loss_ng = re.findall(loss_pat, fin_ng.read())
                    if not loss_ng:
                        loss_ng = float('nan')
                    else:
                        loss_ng = float(loss_ng[0].strip())
                    fin_ng.close()
                    fout_geo.write('%.10f %.10f %.10f %.10f\n' % (a, b, c, loss_geo))
                    fout_ng.write('%.10f %.10f %.10f %.10f\n' % (a, b, c, loss_ng))
        fout_geo.close()
        fout_ng.close()
