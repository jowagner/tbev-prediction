#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import hashlib
import os
import random
import sys
import StringIO
import time

def print_usage():
    print """Usage:'
find tbemb-sampling/ | ./select_model.py > results-by-model.tsv

Options:
    --median     Restrict selection to median seed (using best of middle 2 for even number of runs)

Todo: Is it faster with -name "*.eval.txt" in "find"?'

Creates a file empty-eval-txt.txt if there is an empty eval file'
"""

opt_median = False
opt_help = False

while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option == '--median':
        opt_median = True
    elif option == '--future-option':
        opt_not_used = sys.argv[1]
        del sys.argv[1]
    else:
        print 'Unsupported option %s' %option
        opt_help = True
        break

if len(sys.argv) != 1:
    opt_help = True

if opt_help:
    print_usage()
    sys.exit(0)

results = {}
pud = {}

empty = None
while True:
    line = sys.stdin.readline()
    if not line:
        break
    filename = line.rstrip()
    if filename.endswith('.eval.txt'):
        if os.path.getsize(filename) < 100:
            if empty is None:
                empty = open('empty-eval-txt.txt', 'wb')
            empty.write(line)
            continue
        last_slash = filename.rfind('/')
        parse_name = filename[last_slash+1:-len('.eval.txt')]
        # combined-with-weight-decay-and-random-tiebreaker
        # combined-with-weight-decay-and-hashed-tiebreaker
        # combined-with-uniform-weights-and-random-tiebreaker
        # combined-with-uniform-weights-and-hashed-tiebreaker
        if 'combined-with-weight-decay' in parse_name:
            combinerWeights = 'decay'
        elif 'combined-with-uniform-weights' in parse_name:
            combinerWeights = 'uniform'
        else:
            combinerWeights = '?'
        if 'and-random-tiebreaker' in parse_name:
            combinerTiebreaker = 'random'
        elif 'and-hashed-tiebreaker' in parse_name:
            combinerTiebreaker = 'hashed'
        else:
            combinerTiebreaker = '?'
        folder = filename[:last_slash]
        # tbemb-sampling/oodom-no-neg-weights-data-rich/en_sewta-en_sewte-en_sewtr-on-en_sewtn/seeds-12-sampling-1-with-first/run-6-exp-2908/predictions
        te_tbid = '?'
        tr_tbids = '?'
        domain = '?'
        weights = '?'
        learning = '?'
        seeds = '?'
        samples = '?'
        method = '?'
        sentenceDistanceMeasure = '?'
        run = '?'
        exp_id = '?'
        for component in filename.split('/'):
            if '-on-' in component:
                fields = component.split('-')
                if fields[-1] in ('dev', 'test', 'train'):
                    te_tbid = fields[-2]
                    tr_tbids = '+'.join(fields[:-3])
                else:
                    te_tbid = fields[-1]
                    tr_tbids = '+'.join(fields[:-2])
            elif component[2:5] == 'dom' or component[:3] == 'pud':
                if component[:3] == 'pud':
                    domain = 'pud'
                else:
                    domain = component[:5]
                if 'no-neg-weights' in component:
                    weights = 'no-negative'
                elif 'any-weights' in component:
                    weights = 'any-weights'
                elif 'corner-weights' in component:
                    weights = 'corners'
                elif 'm04-weights' in component:
                    weights = 'm04-weights'
                else:
                    raise ValueError, 'type of weights not detected in component %r of %r' %(component, filename)
                if 'pedantic' in component:
                    learning = 'pedantic'
                elif 'data-rich' in component:
                    learning = 'learning-from-data' # 'use-all-data'
                elif 'oracle' in component:
                    learning = 'oracle'
                elif 'no-learning' in component:
                    learning = 'no-learning'
                else:
                    raise ValueError, 'type of learning not detected in %r' %filename
            elif component[:5] == 'seeds':
                fields = component.split('-')
                try:
                    seeds = fields[1]
                    samples = fields[3]
                    method = '-'.join(fields[5:])
                except:
                    raise ValueError, 'unrecognised %r in experiment %r' %(fields, filename)
            elif component[:8] == 'distance':
                sentenceDistanceMeasure = component[9:]
            elif component[:3] == 'run':
                fields = component.split('-')
                run = fields[1]
                exp_id = fields[3]
        try:
            num_parses = '%d' %(int(seeds)*int(samples))
        except:
            num_parses = '?'
        key = (
            learning,
            domain,
            weights,
            combinerWeights,
            combinerTiebreaker,
            te_tbid,
            tr_tbids,
            method,
            sentenceDistanceMeasure,
            samples,
            seeds,
            num_parses,
        )
        if key not in results:
            results[key] = []
        txt = open(filename, 'rb')
        example_txt = """
        Metric     |     Precision |        Recall |      F1 Score |     AligndAcc
        -----------+---------------+---------------+---------------+---------------
        [...]
        LAS        |  77.633230629 |  77.633230629 |  77.633230629 |  77.633230629
        [...]
        Sentence        LAS     len(gold)       len(predicted)
        0       0.482758621     29      29
        [...]
        """
        # read standard conll18 eval header
        line = txt.readline()
        if not line.startswith('Metric') or not 'F1 Score' in line:
            raise ValueError, 'wrong file format (1) in %r: %r' %(txt, line)
        las_f1 = None
        while True:
            line = txt.readline()
            if not line or line.startswith('Sentence\t'):
                break
            fields = line.replace('|', ' ').split()
            if fields and fields[0] == 'LAS':
                las_f1 = float(fields[3])
        txt.close()
        results[key].append((las_f1, run, exp_id))
        # prepare model selection for PUD
        if te_tbid.endswith('pud'):
            raise ValueError, 'Found PUD test results. Please run on dev output.'
        pud_tbid = te_tbid.split('_')[0]+'_pud'
        key = (learning, 'oodom',) + key[2:5] + (pud_tbid,) + key[6:]
        if key not in pud:
            pud[key] = []
        pud[key].append((run, te_tbid, las_f1, exp_id))

for key in pud:
    scores = pud[key]
    scores.sort()
    num_scores = len(scores)
    current_run = '-1'
    exp_id_list = []
    las_f1_list = []
    te_tbid_list = []
    i = 0
    while True:
        if (i == num_scores or scores[i][0] != current_run) and las_f1_list:
            if len(las_f1_list) == 4:
                avg_las_f1 = sum(las_f1_list) / float(len(las_f1_list))
                if key not in results:
                    results[key] = []
                results[key].append((
                    avg_las_f1,
                    current_run,
                    '%d:%s' %(len(exp_id_list), '+'.join(exp_id_list))
                ))
            exp_id_list = []
            las_f1_list = []
            te_tbid_list = []
        if i == num_scores:
            break
        current_run, te_tbid, las_f1, exp_id = scores[i]
        exp_id_list.append(exp_id)
        las_f1_list.append(las_f1)
        te_tbid_list.append(te_tbid)
        i += 1

if opt_median:
    for key in results:
        las_and_more_list = results[key]
        las_and_more_list.sort()
        while len(las_and_more_list) > 2:
            del las_and_more_list[-1]
            del las_and_more_list[0]
        if len(las_and_more_list) > 1:
            # cannot use average as must select a model
            # --> pick the 2nd (better) item
            del las_and_more_list[0]
        results[key] = las_and_more_list

columns = """
    learning
    domain
    tbembWeights
    combWeights
    combTiebreaker
    test
    parserTraining
    method
    sentenceDistanceMeasure
    samples
    seeds
    parses
    runs
    LAS
    Run
    ExpID
    SelectionKey
""".split()
sys.stdout.write('\t'.join(columns))
sys.stdout.write('\n')

if not empty is None:
    empty.close()
    sys.stderr.write('There were some empty or too small eval.txt files.\n')
    sys.stderr.write('Please check empty-eval-txt.txt\n')

def update(selection, skey, las, selector, run, tr_tbids, row):
    update = False
    if skey in selection:
        best_las = selection[skey][0]
        if las == selector(las, best_las):
            update = True
    else:
        update = True
    if update:
        selection[skey] = (las, run, tr_tbids, row)

selection = {}

for key in sorted(results.keys()):
    scores_and_runs = results[key]
    for score, run, exp_id in scores_and_runs:
        columns = []
        for field in key:
            columns.append(field)
        columns.append('%d' %(len(scores_and_runs)))
        columns.append('%.9f' %score)
        columns.append(run)
        columns.append(exp_id)
        row = '\t'.join(columns)
        mkey1 = tuple(key[:2]) # learning, domain
        weights_type = key[2]         # tbembWeights
        mkey3 = tuple(key[3:7]) # combWeights, combTiebreaker, test, parserTraining
        tr_tbids = key[6]
        method = key[7]
        mkey4 = tuple(key[8:12]) # sentenceDistanceMeasure, samples, seeds, parses
        if mkey3[2].endswith('_pud'):
            mkey3 = mkey3[:3]
        if method.startswith('pick'):
            skey = ('proxy-best', mkey1, mkey3, mkey4)
            update(selection, skey, score, max, run, tr_tbids, row)
        elif method == 'equal':
            skey = ('equal', mkey1, mkey3, mkey4)
            update(selection, skey, score, max, run, tr_tbids, row)
        elif method.startswith('knn'):
            skey = (method, weights_type, mkey1, mkey3, mkey4[1:])
            update(selection, skey, score, max, run, tr_tbids, row)
        else:
            skey = (method, mkey1, mkey3, mkey4)
            update(selection, skey, score, max, run, tr_tbids, row)

for key in sorted(results.keys()):
    scores_and_runs = results[key]
    for score, run, exp_id in scores_and_runs:
        columns = []
        for field in key:
            columns.append(field)
        columns.append('%d' %(len(scores_and_runs)))
        columns.append('%.9f' %score)
        columns.append(run)
        columns.append(exp_id)
        row = '\t'.join(columns)
        mkey1 = tuple(key[:2])
        mkey3 = tuple(key[3:7])
        mkey4 = tuple(key[8:12])
        if mkey3[2].endswith('_pud'):
            tr_tbids = mkey3[3]
            mkey3 = mkey3[:3]
        else:
            tr_tbids = None
        if key[7].startswith('pick'):
            # retrieve run, i.e. seed (and tr_tbids for pud) used for proxy-best
            skey = ('proxy-best', mkey1, mkey3, mkey4)
            # for PUD, also require the tr_tbids to match
            if tr_tbids:
                best_tr_tbids = selection[skey][2]
                if best_tr_tbids != tr_tbids:
                    continue
            # require run (=seed-300) to match
            # but not if using median results
            best_run = selection[skey][1]
            if best_run == run or opt_median:
                skey = ('proxy-worst', mkey1, mkey3, mkey4)
                update(selection, skey, score, min, run, tr_tbids, row)

for skey in selection:
    row = selection[skey][-1]
    sys.stdout.write(row)
    sys.stdout.write('\t')
    sys.stdout.write(repr(skey))
    sys.stdout.write('\n')

