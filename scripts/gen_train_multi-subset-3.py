#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import os
import sys
import time

import common_gen_training

options = common_gen_training.Options(
    defaults = {
        'epochs': 30,
    },
    taskfile_template = '%s-dev-training.tfm',
    usage = """\
./gen_train_multi-subset-3.py

See source code of common_gen_training.py for options.
"""
)

tasks = open(options.taskfile, 'wb')

for collection in [
    ('cs_cac',   'cs_cltt',  'cs_fictree', 'cs_pdt',),
    ('fr_gsd',   'fr_partut', 'fr_sequoia', 'fr_spoken',),
    ('en_ewt',   'en_gum',   'en_lines',   'en_partut',),
]:
    k = len(collection)
    for i1 in range(k-2):
        for i2 in range(i1+1, k-1):
            for i3 in range(i2+1, k):
                s1 = collection[i1]
                s2 = collection[i2]
                s3 = collection[i3]
                for seed in options.seeds:
                    tasks.write('%s %s:%s:%s %d %d %.1f\n' %(
                        options.script, s1, s2, s3, seed,
                        options.epochs, options.deadline
                    ))

tasks.close()

