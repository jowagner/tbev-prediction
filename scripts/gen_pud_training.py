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
en_sewta
en_sewte
en_sewtn
en_sewtr
en_sewtw
fr_ftb
fro_srcmf
ja_bccwj
qhe_hiencs
swl_sslc
""".split()

ignore_concat_tbids = True


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
    sys.stderr.write('Checking most frequent token in %s\n' %filename)
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
        sys.stderr.write('\n')
    if not line:
        break
    if line.startswith('==') or not language:
        language = line.split()[1]
        sys.stderr.write('\n== %s ==\n' %language)
        test_tbid = '?'
        num_tr = 0
        num_dev = 0
        training = []
    if not '/' in line:
        #sys.stderr.write('Ignoring line without /\n')
        continue
    _, filename = line.rsplit('/', 1)
    if filename.startswith('README'):
        #sys.stderr.write('Ignoring README\n')
        continue
    tbid, _, dataset_type = filename.split('-')
    sys.stderr.write('\ntbid = %s\n' %tbid)
    if tbid in ignore_tbids:
        sys.stderr.write('Ignoring tbid in ignore_tbids\n')
        continue
    if ignore_concat_tbids and tbid.endswith('_concat'):
        sys.stderr.write('Ignoring concat tbid\n')
        continue
    fields = line.split()
    if len(fields) > 5:
        rel_path = line.split()[-1]
        tr_path = '/'.join((
            options.treebank_folder,
            rel_path,
        ))
        if '/' in rel_path and most_frequent_token(tr_path) in ('_', 'UNK'):
            sys.stderr.write('Warning: Excluding %r with most frequent token in (_, UNK). Please add to ignore_tbids.\n' %rel_path)
            continue
    dataset_type = dataset_type.split('.')[0]
    sys.stderr.write('dataset_type = %s\n' %dataset_type)
    if tbid.endswith('_pud'):
        if dataset_type == 'test':
            test_tbid = tbid
    elif dataset_type == 'train':
        num_tr += 1
        training.append(tbid)
    elif dataset_type == 'dev':
        num_dev += 1

