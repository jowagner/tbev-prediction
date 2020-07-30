#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

import os
import sys
import time

import project_3_weights

min_num_scores = 5       # for median rows
min_age        = 600.0   # ignore files that are younger than this (in seconds)
min_size       = 700     # to prevent reading files with the old format
opt_project_3  = True    # project 3d to 2d; will be ignored for non-3d inputs

"""
Usage:
source ${PRJ_DIR}/config/locations.sh
find ${RESULT_DIR}/subsets-3/eval -type f | ./collect_data_points.py

writes to output_dir defined below

This script is for producing the sentence-level LAS tables for training and dev
data that are used in the k-NN model and to draw 2D colour plots.
"""

output_dir="results"
if not os.path.exists(output_dir):
    os.mkdir(output_dir)

input_example = """
/.../eval/en_sewta:en_sewte:en_sewtw-on-en_sewtr-train/seed-303/en_sewta:0.510375+en_sewte:-0.722679+en_sewtw:1.212304.txt
/.../eval/en_sewta:en_sewte:en_sewtw-on-en_sewtr-dev/seed-303/en_sewta:0.491940+en_sewte:0.234548+en_sewtw:0.273512.txt
"""

def write_median_table(tsv, median_info, num_weights = 3):
    last_weights_s = None
    for key in sorted(list(median_info)):
        weights_s, sent_i = key
        if weights_s != last_weights_s:
            if last_weights_s and columns:
                tsv.write('\t'.join(columns))
                tsv.write('\n')
            columns = []
            columns.append('Median-if-%d-or-more' %min_num_scores)
            weights = weights_s.split(':')
            for weight in weights:
                columns.append(weight)
            if opt_project_3 and num_weights == 3:
                x, y, z = project_3_weights.project([
                    float(weights[0]),
                    float(weights[1]),
                    float(weights[2]),
                ])
                columns.append('%.9f' %x)
                columns.append('%.9f' %y)
                columns.append('%.9f' %z)
            last_weights_s = weights_s
        scores = median_info[key]
        if sent_i == -2:
            mtime = max(scores)
            columns.append('%.2f' %mtime)
        elif len(scores) < min_num_scores:
            # do not output this row
            columns = []
        else:
            columns[0] = 'Median-of-%d' %len(scores)
            scores.sort()
            while len(scores) > 2:
                del scores[-1]
                del scores[0]
            median = sum(scores) / len(scores)
            columns.append('%.9f' %median)
    if last_weights_s and columns:
        tsv.write('\t'.join(columns))
        tsv.write('\n')

lines = sys.stdin.readlines()
lines.sort()

last_setting_dir = None
last_num_weights = None
last_verbose = 0.0
for line_index, line in enumerate(lines):
    now = time.time()
    if now > last_verbose + 1.0:
        sys.stderr.write(' %.1f%% done\r' %(100.0*line_index/len(lines)))
        last_verbose = now
    line = line.rstrip()
    if time.time() - os.path.getmtime(line) < min_age:
        continue
    if os.path.getsize(line) < min_size:
        continue
    if not line.endswith('.txt'):
        continue
    path_components = line.split('/')
    txt_name = path_components[-1]
    seed_dir = path_components[-2]
    setting_dir = path_components[-3]
    if not seed_dir.startswith('seed-'):
        raise ValueError, 'wrong folder structure (1)'
    seed = seed_dir[5:]
    txt_fields = txt_name[:-4].replace('+', ':').split(':')
    setting = []
    weights = []
    i = 0
    while i < len(txt_fields):
        setting.append(txt_fields[i])
        weights.append(txt_fields[i+1])
        i += 2
    num_weights = len(weights)
    setting_start = ':'.join(setting)
    setting_start = setting_start + '-on-'
    if not setting_dir.startswith(setting_start):
        raise ValueError, 'wrong folder structure (2)'
    if setting_dir != last_setting_dir:
        if last_setting_dir:
            write_median_table(tsv, median_info, last_num_weights)
            tsv.close()
        tsv = open('%s/%s.tsv' %(output_dir, setting_dir), 'wb')
        header_written = False
        median_info = {}
        last_setting_dir = setting_dir
        last_num_weights = num_weights
    mtime = os.path.getmtime(line)
    txt = open(line, 'rb')
    example_txt = """
Metric     |     Precision |        Recall |      F1 Score |     AligndAcc
-----------+---------------+---------------+---------------+---------------
Tokens     | 100.000000000 | 100.000000000 | 100.000000000 |
[skipping a few lines]
UAS        |  81.105215109 |  81.105215109 |  81.105215109 |  81.105215109
LAS        |  77.633230629 |  77.633230629 |  77.633230629 |  77.633230629
CLAS       |  72.289582702 |  71.929824561 |  72.109254920 |  71.929824561
MLAS       |  70.475901788 |  70.125169658 |  70.300098269 |  70.125169658
BLEX       |  72.289582702 |  71.929824561 |  72.109254920 |  71.929824561
Sentence        LAS     len(gold)       len(predicted)
0       0.482758621     29      29
1       0.947368421     19      19
2       0.888888889     18      18
3       0.941176471     17      17
4       0.648648649     37      37
"""
    # read standard conll18 eval header
    line = txt.readline()
    if not line.startswith('Metric') or not 'F1 Score' in line:
        raise ValueError, 'wrong file format (1) in %r: %r' %(txt, line)
    las_f1 = None
    while True:
        line = txt.readline()
        if not line:
            break
        if line.startswith('Sentence\t'):
            break
        fields = line.replace('|', ' ').split()
        if fields and fields[0] == 'LAS':
            las_f1 = fields[3]
    # read sentence-level scores
    sent_scores = []
    sent_lengths = []
    index = 0
    while True:
        line = txt.readline()
        if not line:
            break
        fields = line.split()
        if len(fields) != 4 or fields[0] != ('%d' %index):
            raise ValueError, 'wrong file format (2): %r' %fields
        sent_scores.append(fields[1])
        sent_lengths.append(fields[2])
        if fields[2] != fields[3]:
            raise ValueError, 'sentence length mismatch (1): %r' %fields
        index += 1
    # check and write header
    sent_lengths_s = ':'.join(sent_lengths)
    if header_written:
        if last_sent_lengths_s != sent_lengths_s:
            raise ValueError, 'sentence length mismatch (2)'
    else:
        columns = []
        columns.append('Seed')
        for i, tbid in enumerate(setting):
            columns.append('Weight-%d-for-%s' %(i, tbid))
        if opt_project_3 and num_weights == 3:
            columns.append('X')
            columns.append('Y')
            columns.append('Z')
        columns.append('Time')
        columns.append('LAS-F1-total')
        for i, sent_length in enumerate(sent_lengths):
            columns.append('Sent-%s:%s' %(i, sent_length))
        tsv.write('\t'.join(columns))
        tsv.write('\n')
        last_sent_lengths_s = sent_lengths_s
        header_written = True
    # write data
    columns = []
    columns.append(seed)
    for weight in weights:
        columns.append(weight)
    if opt_project_3 and num_weights == 3:
        x, y, z = project_3_weights.project([
            float(weights[0]),
            float(weights[1]),
            float(weights[2]),
        ])
        columns.append('%.9f' %x)
        columns.append('%.9f' %y)
        columns.append('%.9f' %z)
    columns.append('%.2f' %mtime)
    columns.append(las_f1)
    for score in sent_scores:
        columns.append(score)
    tsv.write('\t'.join(columns))
    tsv.write('\n')
    # update median info
    weights_s = ':'.join(weights)
    key = (weights_s, -1)
    if not key in median_info:
        median_info[key] = []
    median_info[key].append(float(las_f1))
    for sent_i, sent_score in enumerate(sent_scores):
        key = (weights_s, sent_i)
        if not key in median_info:
            median_info[key] = []
        median_info[key].append(float(sent_score))
    key = (weights_s, -2)
    if not key in median_info:
        median_info[key] = []
    median_info[key].append(mtime)

if last_setting_dir:
    write_median_table(tsv, median_info)
    tsv.close()

sys.stderr.write(' %.1f%% done\n' %100.0)
