#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys
import time

import common_gen_training

options = common_gen_training.Options(
    taskfile_template = '%s-pud-training.tfm',
    usage = """\
./gen_pud_training.py [options] < pud-situation.txt > pud-training.tsv

where pud-situation.txt has been generated with assess-pud-situation.sh

See source code of common_gen_training.py for options.
"""
)

min_num_tr_tb = 2

ignore_tbids = """
ar_nyuad
en_esl
en_gumreddit
fr_ftb
fro_srcmf
ja_bccwj
qhe_hiencs
swl_sslc
""".split()


tasks = open(options.taskfile, 'wb')

example_input = """
== Arabic ==
-rw-r--r-- 1 alice users     5292 Jun 21 16:48 UD_Arabic-PUD/README.md
-rw-r--r-- 1 alice users  2084082 Jun 21 16:48 UD_Arabic-PUD/ar_pud-ud-test.conllu

-rw-r--r-- 1 alice users 47051771 Jun 26 21:00 UD_Arabic-NYUAD/ar_nyuad-ud-train.conllu
-rw-r--r-- 1 alice users 40004054 Apr 12  2018 UD_Arabic-PADT/ar_padt-ud-train.conllu

-rw-r--r-- 1 alice users  5867292 Jun 26 21:00 UD_Arabic-NYUAD/ar_nyuad-ud-dev.conllu
-rw-r--r-- 1 alice users  5320123 Apr 12  2018 UD_Arabic-PADT/ar_padt-ud-dev.conllu

== Chinese ==
"""

def most_frequent_token(filename):
    t2f = {}
    for line in open(filename, 'r'):
        if '\t' in line:
            token = line.split('\t')[1]
            try:
                f = t2f[token]
            except:
                f = 0
            t2f[token] = f + 1
    max_f = 0
    retval = None
    for token in t2f:
        f = t2f[token]
        if f > max_f:
            retval = token
            max_f  = f
    #sys.stdout.write('# %s --> %s\n' %(filename, retval))
    return retval

sys.stdout.write('Lang Test NumTr NumDev Training'.replace(' ', '\t'))
sys.stdout.write('\n')

language = None
while True:
    line = sys.stdin.readline()
    if not line or line.startswith('==') and language:
        if num_tr >= min_num_tr_tb:
            # output row
            training = ':'.join(training)
            sys.stdout.write('%s\t%s\t%d\t%d\t%s\n' %(
                language, test_tbid, num_tr, num_dev, training
            ))
            for seed in options.seeds:
                tasks.write('%s %s %d %d %.1f\n' %(
                    options.script, training, seed,
                    options.epochs, options.deadline
                ))
        language = None
    if not line:
        break
    if line.startswith('==') or not language:
        language = line.split()[1]
        test_tbid = '?'
        num_tr = 0
        num_dev = 0
        training = []
    ignore = False
    for tbid in ignore_tbids:
        if '/%s-ud' %tbid in line:
            ignore = True
            break
    if ignore:
        continue
    fields = line.split()
    if len(fields) > 5:
        filename = line.split()[-1]
        tr_path = '/'.join((
            options.treebank_folder,
            filename,
        ))
        if '/' in filename and most_frequent_token(tr_path) in ('_', 'UNK'):
            sys.stderr.write('Warning: Excluding %r with most frequent token in (_, UNK). Please add to ignore_tbids.\n' %filename)
            continue
    if 'PUD' in line:
        if 'test.conllu' in line:
            line = line.replace('/', ' ')
            line = line.replace('-', ' ')
            test_tbid = line.split()[-3]
    elif 'train.conllu' in line:
        num_tr += 1
        line = line.replace('/', ' ')
        line = line.replace('-', ' ')
        training.append(line.split()[-3])
    elif 'dev.conllu' in line:
        num_dev += 1

