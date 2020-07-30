#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018, 2019 Dublin City University
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
    print """Usage:
find tbemb-sampling/ | ./collect_sampling_results.py [options] > results.tsv

Todo: Is it faster with -name "*.eval.txt" in "find"?

Creates a file empty-eval-txt.txt if there is an empty eval file

Options:
    --model-seeds-from FILE  restrict model seeds to (model_tbids, seed) pairs listed in FILE
"""

opt_allowed_model_seeds = None
opt_help = False

while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option == '--model-seeds-from':
        opt_allowed_model_seeds = sys.argv[1]
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

allowed_model_runs = {}
if opt_allowed_model_seeds:
    f = open(opt_allowed_model_seeds, 'rb')
    while True:
        line = f.readline()
        if not line:
            break
        fields = line.split()
        if len(fields) != 2:
            raise ValueError, 'wrong number of columns in model seeds file'
        key = (fields[0].replace(':', '+'), '%d' %(int(fields[1])-300))
        allowed_model_runs[key] = None
    f.close()

def remove_zero_padding(n):
    while len(n) > 1 and n[0] == '0':
        n = n[1:]
    return n

results = {}

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
        te_tbid = '?'
        tr_tbids = '?'
        domain = '?'
        weights = '?'
        num_weights = '0'
        learning = '?'
        seeds = '?'
        samples = '?'
        method = '?'
        sentenceDistanceMeasure = '?'
        sent_rep_type = '?'
        sent_rep_n1 = '?'
        sent_rep_n2 = '?'
        sent_rep_size = '?'
        sent_rep_unk_fraction = '?'
        sent_rep_first_stage = '?'
        sent_rep_rerank = '?'
        sent_rep_offset = '?'
        reference = '?'
        query = '?'
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
            elif component.startswith('with-offset-'):
                sent_rep_offset = remove_zero_padding(component.split('-')[-1])
            elif component.startswith('retrieving'):
                fields = component.split('-')
                if fields[2] != 'for':
                    raise ValueError, 'retrieval not detected in %r' %filename
                reference = fields[1]
                query = fields[3]
            elif component[2:5] == 'dom' or component[:3] == 'pud':
                if component[:3] == 'pud':
                    domain = 'pud'
                else:
                    domain = component[:5]
                if 'no-neg-weights' in component:
                    weights = 'no-negative'
                    num_weights = '31'
                elif 'no-m04-weights' in component:
                    weights = 'minus-040'
                    num_weights = '120'
                elif 'any-weights' in component:
                    weights = 'any-weights'
                    num_weights = '200'
                elif 'corner-weights' in component:
                    weights = 'corners'
                    num_weights = '3'
                else:
                    raise ValueError, 'type of weights not detected in %r' %filename
                if 'pedantic' in component:
                    learning = 'pedantic'
                elif 'data-rich' in component:
                    learning = 'standard'
                elif 'no-learning' in component:
                    learning = 'no-learning'
                elif 'oracle' in component:
                    learning = 'oracle'
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
                fields = component.split('-')
                for index, field in enumerate(fields):
                    if field == 'unk':
                        sent_rep_unk_fraction = remove_zero_padding(
                            fields[index+1]
                        )
                    elif field == 'size':
                        sent_rep_size = remove_zero_padding(fields[index+1])
                    elif field in ('token', 'char', 'elmoformanylangs'):
                        sent_rep_type = field
                        if field == 'token':
                            sent_rep_size = '4096'
                        elif field.startswith('elmo'):
                            sent_rep_size = '1024'
                    elif field == 'lpc':
                        sent_rep_rerank = 'length-punct-cosine'
                    elif field == 'n':
                        sent_rep_n1 = fields[index+1]
                        sent_rep_n2 = fields[index+3]
                    elif field == '567':
                        sent_rep_n1 = '5'
                        sent_rep_n2 = '7'
                        sent_rep_size = '4096'
                    elif field in ('L2', 'SphereL2', 'SphL2', 'Random'):
                        sent_rep_first_stage = field
            elif component[:3] == 'run':
                fields = component.split('-')
                run = fields[1]
                exp_id = fields[3]
        if opt_allowed_model_seeds:
            key = (tr_tbids, run)
            if not key in allowed_model_runs:
                continue
        try:
            num_parses = '%d' %(int(seeds)*int(samples))
        except:
            num_parses = '?'
        key = (
            learning,
            domain,
            weights,
            num_weights,
            combinerWeights,
            combinerTiebreaker,
            te_tbid,
            tr_tbids,
            method,
            sentenceDistanceMeasure,
            sent_rep_type,
            sent_rep_n1,
            sent_rep_n2,
            sent_rep_size,
            sent_rep_unk_fraction,
            sent_rep_offset,
            sent_rep_first_stage,
            sent_rep_rerank,
            reference,
            query,
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

method_avg = {}
for key in results:
    mkey = key[:6] + key[8:]
    scores = map(lambda x: x[0], results[key])
    scores.sort()
    while len(scores) > 2:
        del scores[0]
        del scores[-1]
    median = sum(scores) / float(len(scores))
    if mkey not in method_avg:
        method_avg[mkey] = []
    method_avg[mkey].append(median)

lang_and_method_avg = {}
for key in results:
    lcode = key[6].split('_')[0]
    mkey = key[:6] + (lcode,) + key[8:]
    scores = map(lambda x: x[0], results[key])
    scores.sort()
    while len(scores) > 2:
        del scores[0]
        del scores[-1]
    median = sum(scores) / float(len(scores))
    if mkey not in lang_and_method_avg:
        lang_and_method_avg[mkey] = []
    lang_and_method_avg[mkey].append(median)

columns = """
    learning
    domain
    tbembWeights
    tbembNumWeights
    combWeights
    combTiebreaker
    test
    parserTraining
    method
    sentenceDistanceMeasure
    sentRepType
    sentRepN1
    sentRepN2
    sentRepSize
    sentRepUnkFraction
    sentRepOffset
    sentRepFirstStage
    sentRepRerank
    referenceUnit
    queryUnit
    samples
    seeds
    parses
    runs
    min
    max
    avg
    median
    stddev
    MethodAvgOfMedians
    Median1
    Median2IfEven
    BestRun
    BestExpID
    LangMethodAvgOfMedians
""".split()
sys.stdout.write('\t'.join(columns))
sys.stdout.write('\n')

if not empty is None:
    empty.close()
    sys.stderr.write('There were some empty or too small eval.txt files.\n')
    sys.stderr.write('Please check empty-eval-txt.txt\n')

for key in sorted(results.keys()):
    columns = []
    for field in key:
        columns.append(field)
    scores_and_runs = results[key]
    scores = map(lambda x: x[0], scores_and_runs)
    columns.append('%d' %(len(scores)))
    columns.append('%.9f' %(min(scores)))
    columns.append('%.9f' %(max(scores)))
    avg = sum(scores)/float(len(scores))
    columns.append('%.9f' %avg)
    if len(scores) > 1:
        sum_d2 = 0.0
        for value in scores:
            sum_d2 += (value-avg)**2
        stddev = (sum_d2/(len(scores)-1.0))**0.5
    else:
        stddev = None
    # get run of best score
    scores_and_runs.sort()
    best_run, best_exp_id = scores_and_runs[-1][1:]
    # prune list to element(s) relevant to median
    while len(scores_and_runs) > 2:
        del scores_and_runs[-1]
        del scores_and_runs[0]
    scores = map(lambda x: x[0], scores_and_runs)
    median = sum(scores) / len(scores)
    columns.append('%.9f' %median)
    if stddev is None:
        columns.append('NaN')
    else:
        columns.append('%.9f' %stddev)
    # method average
    mkey = key[:6] + key[8:]
    scores = method_avg[mkey]
    columns.append('%.9f' %(sum(scores) / float(len(scores))))
    # run(s) used in median
    columns.append(scores_and_runs[0][1])
    if len(scores_and_runs) > 1:
        columns.append(scores_and_runs[1][1])
    else:
        columns.append('-')
    columns.append(best_run)
    columns.append(best_exp_id)
    # language and method average
    lcode = key[6].split('_')[0]
    mkey = key[:6] + (lcode,) + key[8:]
    scores = lang_and_method_avg[mkey]
    columns.append('%.9f' %(sum(scores) / float(len(scores))))
    sys.stdout.write('\t'.join(columns))
    sys.stdout.write('\n')

