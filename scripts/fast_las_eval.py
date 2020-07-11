#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.
#
# Contains code taken from conll18_ud_eval.py by Milan Straka and
# Martin Popel, which is subject to the terms of the Mozilla Public
# License, v. 2.0. A copy of the MPL can be obtain at
# http://mozilla.org/MPL/2.0/.

from __future__ import print_function
import sys

def usage():
    print("""
Usage: ./fast_las_eval.py [options] gold.conllu system.conllu

Apart from only checking for _ labels that indicate lines to be ignored,
the order of gold and test makes no difference to F1, precison and recall.

Options:
    --output | -o
         Write to file instead of stdout
    --sentences | -s
         Show sentence-level LAS scores
    --verbose | -v
         Pretty print table.")
""")

token_column = 1
head_column  = 6
label_column = 7
opt_output = None
opt_sentences = False
opt_pretty_table = False
opt_help     = False

while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option in ('--output', '-o'):
        opt_output = sys.argv[1]
        del sys.argv[1]
    elif option in ('--sentences', '-s'):
        opt_sentences = True
    elif option in ('--verbose', '-v', '--pretty-table'):
        opt_pretty_table = True
    else:
        print('Unsupported option %s' %option)
        opt_help = True
        break

if len(sys.argv) != 3:
    opt_help = True

if opt_help:
    usage()
    sys.exit(0)

def evaluate(gold_ud, system_ud):
    global token_column
    global head_column
    global label_column
    total_correct = 0.0
    total_items   = 0.0
    sentence_info = []
    sentence_index = 0
    for gold_sentence in gold_ud:
        try:
            system_sentence = next(system_ud)
        except StopIteration:
            raise ValueError, 'Too few sentences in system output'
        if len(gold_sentence) != len(system_sentence):
            raise ValueError, 'Sentence length (including comments) does not match'
        sentence_correct = 0.0
        sentence_items   = 0.0
        for index, line in enumerate(gold_sentence):
            system_line = system_sentence[index]
            if line.startswith('#'):
                if not system_line.startswith('#'):
                    raise ValueError, 'System sentence has too few comment lines'
                continue
            if system_line.startswith('#'):
                raise ValueError, 'System sentence has too many comment lines'
            gold_fields = line.split('\t')
            gold_head = gold_fields[head_column]
            if gold_head == '_':
                continue
            system_fields = system_line.split('\t')
            if gold_fields[token_column] != system_fields[token_column]:
                raise ValueError, 'Tokens do not match'
            sentence_items += 1.0
            if gold_head == system_fields[head_column] \
            and gold_fields[label_column].split(':')[0] \
            == system_fields[label_column].split(':')[0]:
                sentence_correct += 1.0
        sentence_info.append((
            sentence_index,
            sentence_correct/sentence_items,
            sentence_items,
            sentence_items,
        ))
        total_correct += sentence_correct
        total_items   += sentence_items
        sentence_index += 1
    # Check that there are not too many predictions
    try:
        next(system_ud)
    except StopIteration:
        pass
    else:
        raise ValueError, 'Too many sentences in system output'
    # Compute the F1-scores
    return {
        "LAS": total_correct/total_items,
        "PerSentenceInfo": sentence_info,
    }

def load_conllu_file(path):
    _file = open(path, mode="r", **({"encoding": "utf-8"} if sys.version_info >= (3, 0) else {}))
    sentence_lines = []
    while True:
        line = _file.readline()
        if not line:
            if sentence_lines:
                raise ValueError, 'No end-of-sentence marker at end of file %r' %path
            break
        if line.isspace():
            yield sentence_lines
            sentence_lines = []
        else:
            sentence_lines.append(line)
    _file.close()

def evaluate_wrapper(args):
    # Load CoNLL-U files
    gold_ud = load_conllu_file(args.gold_file)
    system_ud = load_conllu_file(args.system_file)
    return evaluate(gold_ud, system_ud)

class DotDict(dict):
    # https://stackoverflow.com/questions/2352181/how-to-use-a-dot-to-access-members-of-dictionary
    def __getattr__(self, name):
        return self[name]

if True:   # use same indention level as in conll18_ud_eval.py to easy comparison
    if opt_output:
        # Redirect output
        backup_stdout = sys.stdout
        sys.stdout = open(opt_output, 'w')

    # Evaluate
    args = DotDict({'gold_file': sys.argv[1], 'system_file': sys.argv[2]})
    evaluation = evaluate_wrapper(args)

    # Print the evaluation
    if not opt_pretty_table:
        print("LAS F1 Score: {:.9f}".format(100 * evaluation["LAS"]))
    else:
        if True:
            print("Metric     |     Precision |        Recall |      F1 Score |     AligndAcc")
        print("-----------+---------------+---------------+---------------+---------------")
        for metric in ['LAS',]:
            if True:
                print("{:11}|{:14.9f} |{:14.9f} |{:14.9f} |{}".format(
                    metric,
                    100 * evaluation[metric],
                    100 * evaluation[metric],
                    100 * evaluation[metric],
                    "{:14.9f}".format(100 * evaluation[metric]),
                ))
    if opt_sentences:
        print("Sentence\tLAS\tlen(gold)\tlen(predicted)")
        for sentence_info in evaluation['PerSentenceInfo']:
            print('%d\t%.9f\t%d\t%d' %sentence_info)

    if opt_output:
        # Close file and restore stdout
        sys.stdout.close()
        sys.stdout = backup_stdout

