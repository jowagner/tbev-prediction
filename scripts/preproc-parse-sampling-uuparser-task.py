#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018, 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import random
import sys
import time

filename = sys.argv[1]

opt_filter_num_seeds_and_num_samples = True
opt_skip_oracle_runs = False    # warning: currently only applied if also opt_filter_num_seeds_and_num_samples
opt_skip_pedantic_runs = True
opt_skip_indomain_runs = True
opt_skip_if_previously_preprocessed = False    # True = inserts code into script to quit early when it was previously pre-processed
opt_keep_log_files   = False
opt_max_num_parses   = 1
opt_retrict_knn_to_suffixes = 'ctr-1 ctr-3 ctr-9 doc-3 doc-9 doc-16 plain-avg'.split()

list_of_num_seeds_and_num_samples = []
for num_seeds in (1,3,6,9):
    for num_samples in (1,3,6,9):
        num_parses = num_seeds * num_samples
        if num_parses <= opt_max_num_parses:
            list_of_num_seeds_and_num_samples.append((num_seeds, num_samples))

f = open(filename, 'rb')
lines = f.readlines()
f.close()

# detect already modified scripts
is_preprocessed = False
for line in lines:
    if line.startswith('WORK_DIR=/dev/shm/'):
        is_preprocessed = True
        break
# we do not need to update the script if it has been updated
# before and we do not need to add code to quit the script
# immediately to skip parsing
if is_preprocessed and not opt_skip_if_previously_preprocessed:
    sys.exit(0)

os.unlink(filename)

tmp_name = []
for i in range(3):
    tmp_name.append(''.join(random.sample('abcdefghijklmnopqrstuvwxyz', 5)))
tmp_name = '-'.join(tmp_name)

task_id = 'unk'

# experiment indom-no-neg-weights-data-rich/en_sewta-en_sewte-en_sewtw-on-en_sewtw/seeds-20-sampling-1-with-random/run-6-exp-49999 parse 20-302-7
# numeric: 49999-7

f = open(filename, 'wb')
fast_exit = False
if is_preprocessed and opt_skip_if_previously_preprocessed:
    fast_exit = True

for line in lines:
    if line.startswith('MODEL_BASE='):
        f.write('MODEL_BASE=${RESULT_DIR}/uuparser_multilingual_std/uuparser-tbemb-v23-naacl-2019-repeat\n')
    elif line.startswith('PARSER_DIR='):
        f.write('PARSER_DIR=${UUPARSER_DIR}\n')
    elif line.startswith('for EPOCH in best'):
        f.write("""
EPOCH=$(fgrep "Model score after epoch" \\
    ${MODEL_BASE}/seed-${SEED}/${MODEL_TBIDS}/stdout.txt | \\
    sort -n -k 6 | tail -n 1 | cut -d' ' -f5 | cut -d: -f1)

if [ -z ${EPOCH} ] ; then
    echodate "Error: Cannot find best epoch."
    exit 1
else
""")
    elif 'neuralfirstorder' in line:
        line = line.replace('neuralfirstorder', 'barchybrid')
        f.write(line)
    elif '--bibi-lstm' in line:
        f.write('        --userl                        \\\n')
        f.write('        --k 3                          \\\n')
    elif line.startswith('done'):
        f.write('fi\n')
    elif line.startswith('WORK_DIR='):
        # replace this with a workdir under /dev/shm
        f.write('WORK_DIR=/dev/shm/tbemb-%s-%s\n' %(task_id, tmp_name))
        # slow down to reduce risk of filesystem overload
        f.write('sleep $[ ( $RANDOM % 5 )  + 2 ]s\n')
    elif line.startswith('exit'):
        # remove previous filter: do not write this line
        pass
    elif line.startswith('# experiment') and opt_filter_num_seeds_and_num_samples:
        # experiment oodom-no-neg-weights-oracle/en_gum-en_...
        if opt_skip_oracle_runs and 'oracle' in line:
            fast_exit = True
        if opt_skip_pedantic_runs and 'pedantic' in line:
            fast_exit = True
        if opt_skip_indomain_runs and 'indom' in line:
            fast_exit = True
        for component in line.replace('/', ' ').split():
            if component.startswith('seeds'):
                fields = component.split('-')
                num_seeds = int(fields[1])
                num_samples = int(fields[3])
                if not (num_seeds, num_samples) in list_of_num_seeds_and_num_samples:
                    fast_exit = True
                    break
                if opt_retrict_knn_to_suffixes and 'knn' in component:
                    found_suffix = False
                    for suffix in opt_retrict_knn_to_suffixes:
                        if component.endswith(suffix):
                            found_suffix = True
                            break
                    if not found_suffix:
                        fast_exit = True
                        break
        f.write(line)
    elif line.startswith('# numeric:'):
        task_id = line.split()[-1]
        f.write(line)
        if fast_exit:
            f.write('\n\nexit\n\n')
            break
    elif line.startswith('tar -cjf ${PREDICT_DIR}/'):
        tarname = line.split()[2]
        f.write('rm -f %s\n' %tarname)
        if opt_keep_log_files:
            f.write(line)
    elif line.startswith('    gzip'):
        # 0    1 2                                 3 4
        # gzip < ${OUTPUT_DIR}/${FAKE_TBID}.conllu > ${PREDICTION}.gz
        fields = line.split()
        gzfile = fields[4]
        f.write("""\
    if [ -e %s ] ; then
        mv %s %s.old
    fi
""" %(gzfile, gzfile, gzfile))
        f.write(line)
    else:
        f.write(line)
f.close()

# https://stackoverflow.com/questions/12791997/how-do-you-do-a-simple-chmod-x-from-within-python

def make_executable(path):
    mode = os.stat(path).st_mode
    mode |= (mode & 292) >> 2    # copy R bits to X
    os.chmod(path, mode)

make_executable(filename)

