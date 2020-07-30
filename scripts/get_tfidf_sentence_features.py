#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018, 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# Also used in ape-topic-labels project.

import hashlib
import os
import h5py
import math
import numpy
import sys
import time

def usage():
    print("""
Usage: ././get_tfidf_sentence_features.py [options] input.txt output-sent-rep.hdf5

Warning: without --vocab, it also writes to input.txt.vocab

Options:
    --demo
         Do not write vocab and sentence representation to files. Instead, print numeric token sequences.
    --vocab FILE
         Read vocab from file (default: build vocab and write to input.txt.vocab)
    --truecase
         Keep upper case characters (default: lowercase all input)
    --keep-email
         Keep e-mail addresses (default: replace all tokens with an @-sign that do not start with an @-sign and where the @-sign is not the last character)
    --keep-urls
         Keep URLs (default: truncate all tokens after the first occurrence of '://' if it is preceded by at least 2 characters, e.g. 'tv://'; in character-ngram mode, truncation operates on space separated tokens, not on the n-grams)
    --keep-digits
         Keep digits (default: map digits 0-9 to 5)
    --limit-repetition STRING NUMBER
         For each character in STRING, replace any sequence of more than NUMBER of the character with NUMBER occurrences. This is done after digit replacement. (use empty STRING or NUMBER <= 0 to deactivate; default: _-=+*/.!?5 3)
    --char-ngrams NUMBER_1 NUMBER_2
         Use character n-grams with n in {NUMBER_1, ..., NUMBER_2} as tokens (default: spaces in input separate tokens)
    --unk-fraction NUMBER
         Set fraction of the vector to use for unknown tokens (default: 0.0 == skip unknowns)
    --hashes NUMBER
         Number of times unknown tokens should be hashed (default: 1)
    --vector-size  NUMBER
         Set dimensionality of sentence representation (default: 4096)
    --plus-two
         Ensure idf values cannot be zero or negative by adding 2 to the number of documents so that hits + 1 cannot reach it.
         (Only effective when building a new vocab, i.e. without --vocab.)
    --exit-after-vocab
         Only write vocab.
""")


opt_vector_size  = 4096
opt_unk_fraction = 0.0
opt_vocab = None   # None means build vocab from input and write it to file
opt_exit_after_vocab = False
opt_exit_after_numeric = False
opt_show_numeric = False
opt_demo = False
opt_limit_repetition_chars = '_-=+*/.!?5'
opt_limit_repetition_max_n = 3
opt_truncate_urls = True
opt_replace_email = True
opt_map_digits = True
opt_use_idf = True
opt_char_ngrams = False
opt_ngram_min_n = 5
opt_ngram_max_n = 7
opt_lowercase = True
opt_num_hashes = 1
opt_plus_two = False
opt_verbosity_interval = 1.0
opt_debug   = False
opt_help    = False

while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option == '--keep-digits':
        opt_map_digits = False
    elif option == '--keep-urls':
        opt_truncate_urls = False
    elif option == '--limit-repetition':
        opt_limit_repetition_chars = sys.argv[1]
        opt_limit_repetition_max_n = int(sys.argv[2])
        del sys.argv[2]
        del sys.argv[1]
    elif option == '--truecase':
        opt_lowercase = False
        del sys.argv[1]
    elif option == '--vocab':
        opt_vocab = sys.argv[1]
        del sys.argv[1]
    elif option == '--vector-size':
        opt_vector_size = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--char-ngrams':
        opt_ngram_min_n = int(sys.argv[1])
        opt_ngram_max_n = int(sys.argv[2])
        opt_char_ngrams = True
        del sys.argv[2]
        del sys.argv[1]
    elif option == '--hashes':
        opt_num_hashes = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--unk-fraction', '--unk-percentage'):
        fraction = sys.argv[1]
        while fraction[0] == '0' and len(fraction.split('.')[0]) > 1:
            fraction = fraction[1:]
        fraction = float(fraction)
        if option == '--unk-percentage':
            fraction /= 100.0
        opt_unk_fraction = fraction
        del sys.argv[1]
    elif option == '--exit-after-vocab':
        opt_exit_after_vocab = True
    elif option == '--verbosity-interval':
        opt_verbosity_interval = float(sys.argv[1])
        del sys.argv[1]
    elif option == '--plus-two':
        opt_plus_two = True
    elif option == '--demo':
        opt_demo = True
    elif option == '--debug':
        opt_debug = True
    else:
        print('Unsupported option %s' %option)
        opt_help = True
        break

if len(sys.argv) == 2 and opt_exit_after_vocab:
    pass
elif len(sys.argv) == 3:
    pass
else:
    opt_help = True

if opt_help:
    usage()
    sys.exit(0)

opt_unk_size = int(opt_vector_size * opt_unk_fraction)
opt_voc_size = opt_vector_size - opt_unk_size

if opt_demo:
    opt_exit_after_vocab = True
    opt_exit_after_numeric = True
    opt_show_numeric = True

input_txt = open(sys.argv[1], 'rb')

if not opt_exit_after_vocab:
    sent_rep = h5py.File(sys.argv[2], 'w')

# read txt (tokenised and one sentence per line) and normalise tokens
data = []
lines = input_txt.readlines()
num_lines = float(len(lines))
last_verbose = 0.0
for line_index, line in enumerate(lines):
    if time.time() > last_verbose + opt_verbosity_interval:
        sys.stderr.write('%.1f%% normalised\r' %(100.0*line_index/num_lines))
        last_verbose = time.time()
    if opt_lowercase:
        # do this in unicode mode to not dependent on locale settings
        line = line.decode('UTF-8').lower().encode('UTF-8')
    if opt_replace_email and '@' in line:
        sentence = []
        for token in line.split():
            pos = token.find('@')
            if 0 < pos < len(token)-1:
                token = '==@=='
            sentence.append(token)
        line = ' '.join(sentence)
    if opt_truncate_urls and '://' in line:
        sentence = []
        for token in line.split():
            pos = token.find('://')
            if pos > 1:    # shortest URL scheme on https://www.iana.org/assignments/uri-schemes/uri-schemes.xhtml has 2 characters
                token = '=://='
            sentence.append(token)
        line = ' '.join(sentence)
    if opt_map_digits:
        for c in '012346789':
            line = line.replace(c, '5')
    if opt_limit_repetition_chars and opt_limit_repetition_max_n > 0:
        for c in opt_limit_repetition_chars:
            search_for = (opt_limit_repetition_max_n+1) * c
            replace_with = opt_limit_repetition_max_n * c
            while search_for in line:
                line = line.replace(search_for, replace_with)
    if opt_char_ngrams:
        sentence = []
        for n in range(opt_ngram_min_n, opt_ngram_max_n+1):
            padding = int((n-1)//2) * '_'
            line = line.rstrip()
            line = line.replace(' ', '_')
            padded = padding + line + padding
            padded = padded.decode('UTF-8') # slice n-grams in unicode mode
            for start in range(len(padded)+1-n):
                ngram = padded[start:start+n]
                sentence.append(ngram.encode('UTF-8'))
    else:
        # here we could map unknown tokens
        # TODO: or should we map them after building the vocab?
        #       --> provide option and experiment
        sentence = line.split()
    data.append(sentence)
input_txt.close()
sys.stderr.write('finished normalisation   \r')

# get vocab
vocab = {}
idfs  = []
if opt_vocab:
    # read vocab (and idf values) from file
    lines = open(opt_vocab, 'rb').readlines()
    num_lines = len(lines)
    for index, line in enumerate(lines):
        if time.time() > last_verbose + opt_verbosity_interval:
            sys.stderr.write('%.1f%% of vocab read\r' %(100.0*index/num_lines))
            last_verbose = time.time()
        token, idf = line.rstrip().split('\t')
        # special entries (for the idf values of hashed unknowns)
        # contain spaces
        if not ' ' in token:
            vocab[token] = index
        idfs.append(float(idf))
    sys.stderr.write('finished reading vocab   \r')
    if len(vocab.keys()) != opt_voc_size:
        raise ValueError, 'vocab size does not match configuration'
else:
    # build new vocab
    # (for the moment without idf values)
    token2freq = {}
    num_sentences = len(data)
    for sentence_index, sentence in enumerate(data):
        if time.time() > last_verbose + opt_verbosity_interval:
            sys.stderr.write('%.1f%% tokens counted\r' %(100.0*sentence_index/num_sentences))
            last_verbose = time.time()
        for token in sentence:
            # update token frequency
            if token in token2freq:
                token2freq[token] += 1
            else:
                token2freq[token] = 1
    sys.stderr.write('finished counting tokens   \r')
    nfreq_and_token = []
    keys = token2freq.keys()
    num_keys = float(len(keys))
    for token_index, token in enumerate(token2freq):
        if time.time() > last_verbose + opt_verbosity_interval:
            sys.stderr.write('%.1f%% types indexed\r' %(100.0*token_index/num_keys))
            last_verbose = time.time()
        freq = token2freq[token]
        nfreq_and_token.append((
            -freq,
            hashlib.sha256(token).hexdigest(),   # tie-breaker
            token
        ))
    sys.stderr.write('finished indexing types   \r')
    nfreq_and_token.sort()
    sys.stderr.write('finished sorting tokens counts   \r')
    for index, item in enumerate(nfreq_and_token[:opt_voc_size]):
        vocab[item[-1]] = index
    sys.stderr.write('finished populating vocab with types   \r')
    # address unlikely case that vocab is too small:
    # make up some tokens to fill vocab to the desired size
    # TODO: wouldn't it be better to increase opt_unk_size
    #       in this case?
    vocab_keys = list(vocab.keys())
    next_index = len(vocab_keys)
    missing = opt_voc_size - next_index
    vocab_keys.sort()
    vocab_id = hashlib.sha256('\n'.join(vocab_keys)).hexdigest()
    for i in range(missing):
        if time.time() > last_verbose + opt_verbosity_interval:
            sys.stderr.write('%.1f%% vocab padding created\r' %(100.0*i/missing))
            last_verbose = time.time()
        token = 'PAD-%s-%d' %(vocab_id, missing)
        vocab[token] = next_index
        next_index += 1
        missing -= 1
    sys.stderr.write('finished building vocab   \r')

if opt_debug:
    print 'vocab', vocab

# map tokens to indices
numeric_data = []
num_sentences = len(data)
for i, sentence in enumerate(data):
    if time.time() > last_verbose + opt_verbosity_interval:
        sys.stderr.write('%.1f%% sentences mapped\r' %(100.0*i/num_sentences))
        last_verbose = time.time()
    numeric_sentence = []
    count_tokens = 0
    count_in_vocab = 0
    count_skipped = 0
    for token in sentence:
        count_tokens += 1
        if token in vocab:
            count_in_vocab +=1
            index = vocab[token]
        # add alternative unknown mapping methods here
        elif opt_unk_size:
            if opt_num_hashes > 1:
                for j in range(opt_num_hashes):
                    h = hashlib.sha256('%d:%s' %(j,token)).hexdigest()
                    index = opt_voc_size + int(h, 16) % opt_unk_size
                    numeric_sentence.append(index)
                continue
            h = hashlib.sha256(token).hexdigest()
            index = opt_voc_size + int(h, 16) % opt_unk_size
        else:
            # skip this token
            count_skipped += 1
            continue
        numeric_sentence.append(index)
    numeric_data.append(numeric_sentence)
    if opt_show_numeric:
        numeric_sentence = map(lambda x: '%d' %x, numeric_sentence)
        numeric_sentence = ' '.join(numeric_sentence)
        sys.stdout.write('[%d]: %s\n' %(i, numeric_sentence))
    if opt_debug:
        density = len(set(numeric_sentence)) / float(opt_vector_size)
        sys.stderr.write('[%d]: %d tokens, %d skipped, %d in vocab, density %.3f\n' %(
            i, count_tokens, count_skipped, count_in_vocab, density
        ))
sys.stderr.write('finished mapping sentences   \r')

if opt_exit_after_numeric:
    sys.exit(0)

# switch to using numeric data
data, text_data = numeric_data, data

sys.stderr.write('moved to numeric data   \r')
last_verbose = 0.0

if not opt_vocab:
    sys.stderr.write('getting ready to write vocab...   \r')
    # calculate idf values for numeric tokens
    num_idf_values = opt_voc_size+opt_unk_size
    token2hits = num_idf_values * [0]
    for index, sentence in enumerate(data):
        if time.time() > last_verbose + opt_verbosity_interval:
            sys.stderr.write('%.1f%% hits calculated\r' %(100.0*index/num_sentences))
            last_verbose = time.time()
        counted_so_far = {}
        for token in sentence:
            if token not in counted_so_far:
                token2hits[token] += 1
                # only count one hit for each token type
                counted_so_far[token] = None
    lp2 = len(data)
    if opt_plus_two:
        lp2 += 2.0
    for index in range(num_idf_values):
        if time.time() > last_verbose + opt_verbosity_interval:
            sys.stderr.write('%.1f%% idf values calculated\r' %(100.0*index/num_idf_values))
            last_verbose = time.time()
        hits = token2hits[index]
        # smooth idf (allow for unseen tokens)
        idf = math.log(lp2/(1.0+hits))
        idfs.append(idf)
    sys.stderr.write('finished calculating idf values   \r')
    # write vocab and idfs to file in order of index
    index2token = {}
    for token in vocab:
        index = vocab[token]
        index2token[index] = token
    vfile = open(sys.argv[1]+'.vocab', 'wb')
    for index in range(num_idf_values):
        if time.time() > last_verbose + opt_verbosity_interval:
            sys.stderr.write('%.1f%% vocab lines written\r' %(100.0*index/num_idf_values))
            last_verbose = time.time()
        if index < opt_voc_size:
            token = index2token[index]
        else:
            # use space to mark this entry as special
            token = 'UNK %d' %(index+1-opt_voc_size)
        idf = idfs[index]
        vfile.write('%s\t%.9f\n' %(token, idf))
    vfile.close()
    sys.stderr.write('finished writing vocab   \r')

if opt_exit_after_vocab:
    sys.exit(0)

num_sentences = len(data)

sent_data = None
for i in range(num_sentences):
    if time.time() > last_verbose + opt_verbosity_interval:
        sys.stderr.write('%.1f%% sentence representations written\r' %(100.0*i/num_sentences))
        last_verbose = time.time()
    sentence_rep = numpy.zeros(opt_voc_size + opt_unk_size)
    if not data[i]:
        sys.stderr.write('Warning: all zero vector for sentence %d: %r\n' %(i, text_data[i]))
    for index in data[i]:
        if opt_use_idf:
            sentence_rep[index] += idfs[index]
        else:
            sentence_rep[index] += 1.0
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
sys.stderr.write('finished writing sentence representations    \n')

