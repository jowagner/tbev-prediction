#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# (C) 2018 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# levenshteinDistance() copied from stackoverflow, see comment below

import os
import hashlib
import h5py
import numpy
import sys
import time

"""
Usage: ./get_elmo_sentence_features.py [options] input-tok-rep.hdf5 output-sent-rep.hdf5
"""

method = 'max'
if sys.argv[1] == '--method':
    method = sys.argv[2]
    del sys.argv[2]
    del sys.argv[1]

opt_elmoformanylanguages = None
if sys.argv[1][:17] == '--elmoformanylang':
    opt_elmoformanylanguages = sys.argv[2]
    del sys.argv[2]
    del sys.argv[1]

if method == 'first':
    method = 'index:0'
if method == 'last':
    method = 'index:-1'

token_rep = h5py.File(sys.argv[1], 'r')
sent_rep  = h5py.File(sys.argv[2], 'w')

# The following function is from
# Salvador Dali's answer on
# https://stackoverflow.com/questions/2460177/edit-distance-in-python

def levenshteinDistance(s1, s2):
    if len(s1) > len(s2):
        s1, s2 = s2, s1
    distances = range(len(s1) + 1)
    for i2, c2 in enumerate(s2):
        distances_ = [i2+1]
        for i1, c1 in enumerate(s1):
            if c1 == c2:
                distances_.append(distances[i1])
            else:
                distances_.append(1 + min((distances[i1], distances[i1 + 1], distances_[-1])))
        distances = distances_
    return distances[-1]

keys = []
if opt_elmoformanylanguages:
    token_column = 1
    head_column  = 6
    sentences = []
    _file = open(opt_elmoformanylanguages, mode='r', encoding='utf-8')
    tokens = []
    while True:
        line = _file.readline()
        if not line:
            if tokens:
                raise ValueError('No end-of-sentence marker at end of file %r' %path)
            break
        if line.isspace():
            # apply same processing as in elmoformanylangs/__main__.py
            sent = '\t'.join(tokens)
            sent = sent.replace('.', '$period$')
            sent = sent.replace('/', '$backslash$')
            sentences.append(sent)
            tokens = []
        elif not line.startswith('#'):
            fields = line.split('\t')
            if fields[head_column] != '_':
                tokens.append(fields[token_column])
    _file.close()
    num_sentences = len(sentences)
    exact_matches = 0
    for i, sentence in enumerate(sentences):
        if sentence in token_rep:
            exact_matches += 1
            best_key = sentence
            best_score = (100.0, '')
        else:
            best_key = None
            best_score = (-1.0, '')
            for key in token_rep.keys():
                score = 100.0/(1.0+levenshteinDistance(key, sentence))
                info = '%d\n%s\n%s' %(i, key, sentence)
                tiebreaker = hashlib.sha256(info.encode('UTF-8')).hexdigest()
                if (score, tiebreaker) > best_score:
                    best_key   = key
                    best_score = (score, tiebreaker)
        #print('[%d]' %i)
        #print('\tsentence =', sentence)
        #print('\tbest_key =', best_key)
        #print('\tscore    =', best_score)
        keys.append(best_key)
    print('%d of %d matches were exact' %(exact_matches, num_sentences))
else:
    # hdf5 format of the `allennlp elmo` command
    num_sentences = len(token_rep) - 1  # -1 due to the extra key 'sentence_to_index'
    for i in range(num_sentences):
        keys.append('%d' %i)

sent_data = None
for i, key in enumerate(keys):
    tokens = token_rep[key]
    if method.startswith('index:'):
        index = int(method.split(':')[1])
        sentence_rep = tokens[index]
    elif method == 'max':
        sentence_rep = numpy.amax(tokens, axis=0)
    elif method == 'avg':
        sentence_rep = numpy.average(tokens, axis=0)
    else:
        raise ValueError('unknown method %r' %method)
    if sent_data is None:
        # first sentence: create output array
        sent_data = sent_rep.create_dataset(
            'sent_rep',
            (num_sentences, sentence_rep.shape[0]),
            dtype = sentence_rep.dtype,
            chunks = True,
            compression = "gzip"
        )
    sent_data[i] = sentence_rep

sent_rep.close()
token_rep.close()

