#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys
import StringIO
import time

def print_usage():
    print """
export PRJ_DIR=${HOME}/tbemb/ADAPT-DCU
export UD_TREEBANK_DIR=${HOME}/ud-parsing/gold-token-pred-pos

find tbemb-sampling/ | ./gen_tbemb_combine.py dev > combine-001.tfm
or
find tbemb-sampling/ | ./gen_tbemb_combine.py test > combine-001.tfm

It is safe to add -type f to the find command. Todo: Is this faster or slower?
To also update existing combiner output, add | fgrep -v /combined
To also update existing eval output, add | fgrep -v .eval.txt

Options:
    --project-dir PATH
         Where to find scripts and configuration (default: use $PRJ_DIR environment variable and fall back to $HOME/ADAPT-DCU if it is not set)
    --treebank-dir PATH
         Where to find the gold .conllu files for evaluation (default: use $UD_TREEBANK_DIR environment variable)
"""

worker_file = sys.stdout

num_workers = 1
try:
    opt_project_dir = os.environ['PRJ_DIR']
except:
    opt_project_dir = os.environ['HOME'] + '/ADAPT-DCU'
try:
    opt_treebank_dir = os.environ['UD_TREEBANK_DIR']
except:
    opt_treebank_dir = None
opt_combiner_dir         = None    # will be set below if not set by user
opt_help                 = False
opt_max_parses           = 300
opt_max_k                = 60
opt_max_samples          = 60
opt_epochs               = None # ('e5', )  # files from any other epochs will be ignored
opt_decay                = 0.999

# TODO: add options for all variables above
while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option == '--project-dir':
        opt_project_dir = sys.argv[1]
        del sys.argv[1]
    elif option == '--treebank-dir':
        opt_treebank_dir = sys.argv[1]
        del sys.argv[1]
    else:
        print 'Unsupported option %s' %option
        opt_help = True
        break

if len(sys.argv) != 2:
    opt_help = True

if opt_help:
    print_usage()
    sys.exit(0)

test_type = sys.argv[1]  # test on dev or test data?

if opt_combiner_dir is None:
    opt_combiner_dir = opt_project_dir + '/combination'

if opt_treebank_dir is None:
    raise ValueError, 'Treebank_dir not set (load environment or use --treebank-dir)'

tbnames = map(
    lambda x: x.split(),
    open('%s/config/tbnames.tsv' %opt_project_dir, 'rb').readlines()
)

example_input = """
tbemb-sampling/oodom-no-neg-weights-data-rich/en_sewta-en_sewte-en_sewtn-on-en_sewtr/seeds-12-sampling-1-with-first/run-0-exp-92
tbemb-sampling/oodom-no-neg-weights-data-rich/en_sewta-en_sewte-en_sewtn-on-en_sewtr/seeds-12-sampling-1-with-first/run-0-exp-92/predictions/12-339-10_log.tar.bz2
tbemb-sampling/oodom-no-neg-weights-data-rich/en_sewta-en_sewte-en_sewtn-on-en_sewtr/seeds-12-sampling-1-with-first/run-0-exp-92/predictions/12-326-3_e5.conllu.gz
tbemb-sampling/oodom-no-neg-weights-data-rich/en_sewta-en_sewte-en_sewtw-on-en_sewtn/seeds-3-sampling-1-with-first/run-7-exp-5028/predictions/combined-with-weight-decay.conllu
"""

combined = {}
evaluated = {}
evaluate = {}
candidates = {}
counts = {}

while True:
    line = sys.stdin.readline()
    if not line:
        break
    if '/workdir' in line:
        continue
    filename = line.rstrip()
    if filename.endswith('_log.tar.bz2'):
        os.unlink(filename)
    elif filename.endswith('.conllu.gz') or filename.endswith('.conllu'):
        last_slash = filename.rfind('/')
        if filename.endswith('.gz'):
            parse_name = filename[last_slash+1:-len('.conllu.gz')]
        else:
            parse_name = filename[last_slash+1:-len('.conllu')]
        fields = parse_name.replace('_', '-').split('-')
        folder = filename[:last_slash]
        # Is it a parser prediction or a combiner output?
        if fields[0] == 'combined':
            combined[folder] = None
            if not folder in evaluate:
                evaluate[folder] = []
            evaluate[folder].append(filename)
        elif len(fields) == 4:
            epoch = fields[-1]
            if opt_epochs and epoch not in opt_epochs:
                continue
            if not folder in candidates:
                candidates[folder] = []
            candidates[folder].append((filename, int(fields[2])))
            count = int(fields[0])
            try:
                if counts[folder] != count:
                    raise ValueError, 'Inconsistent count of parses in %r' %folder
            except KeyError:
                counts[folder] = count
        else:
            sys.stderr.write('Ignoring unsupported file %r\n' %filename)
    elif filename.endswith('.eval.txt'):
        last_slash = filename.rfind('/')
        folder = filename[:last_slash]
        evaluated[folder] = None

overwrite_test_type = None
for folder in candidates:
    #sys.stdout.write('# == Folder %s ==\n' %folder)
    if not folder in combined:
        parses = candidates[folder]
        actual_count = len(parses)
        target_count = counts[folder]
        if actual_count > target_count:
            raise ValueError, 'too many parses in %r' %folder
        elif actual_count == target_count:
            # generate combiner script
            #for weight_decay in (False, True):
            #    for random_tiebreaker in (False, True):
            if True:
                    weight_decay = True
                    random_tiebreaker = False
                    command = []
                    command.append('%s/parser.py' %opt_combiner_dir)
                    if weight_decay:
                        weights = []
                        for (filename, rank) in parses:
                            weight = opt_decay ** rank
                            weights.append('%.6f' %weight)
                        command.append('--weights')
                        command.append(':'.join(weights))
                    command.append('--prune-labels')
                    if random_tiebreaker:
                        command.append('--random-tiebreaker')
                    else:
                        command.append('--seed 100')
                    command.append('--outfile')
                    if weight_decay and random_tiebreaker:
                        command.append('%s/combined-with-weight-decay-and-random-tiebreaker.conllu' %folder)
                    elif weight_decay:
                        command.append('%s/combined-with-weight-decay-and-hashed-tiebreaker.conllu' %folder)
                    elif random_tiebreaker:
                        command.append('%s/combined-with-uniform-weights-and-random-tiebreaker.conllu' %folder)
                    else:
                        command.append('%s/combined-with-uniform-weights-and-hashed-tiebreaker.conllu' %folder)
                    for filename, _ in parses:
                        command.append(filename)
                    command = ' '.join(command)
                    worker_file.write('%s\n' %command)
    elif not folder in evaluated:
        for filename in evaluate[folder]:
            # test tbid
            test_tbid = None
            for component in filename.split('/'):
                if '-on-' in component:
                    test_tbid = component.split('-')[-1]
                    if test_tbid in ('dev', 'test', 'train'):
                        overwrite_test_type = test_tbid
                        test_tbid = component.split('-')[-2]
                    else:
                        overwrite_test_type = None
                    #sys.stdout.write('# tbid from component %s\n' %component)
            if not test_tbid:
                raise ValueError, 'could not find test tbid in %r' %filename
            # treebank long name (UD...)
            test_tbname = None
            for candidate_tbname, candidate_tbid in tbnames:
                if candidate_tbid == test_tbid:
                    test_tbname = candidate_tbname
                    #sys.stdout.write('# tbname %s from tbid %s\n' %(test_tbname, test_tbid))
                    break
            if not test_tbname:
                raise ValueError, 'could not find tbname for tbid %r' %test_tbid
            # filenames
            if overwrite_test_type:
                use_test_type = overwrite_test_type
            else:
                use_test_type = test_type
            goldfile = '%s/%s/%s-ud-%s.conllu' %(
                opt_treebank_dir, test_tbname, test_tbid, use_test_type
            )
            evalfile = filename[:-len('.conllu')] + '.eval.txt'
            command = []
            command.append('%s/scripts/conll18_ud_eval.py' %opt_project_dir)
            command.append('-v')
            command.append('-o')
            command.append(evalfile)
            command.append(goldfile)
            command.append(filename)
            command = ' '.join(command)
            worker_file.write('%s\n' %command)

sys.stderr.write('Finished\n')

