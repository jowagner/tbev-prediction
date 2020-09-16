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

# note: h5py and scipy.spatial are imported below if not in dry-run mode

# local module; symlink the .py file from m-bist/src
import project_3_weights

def print_usage():
    print 'Usage: ls data-ichec/*.tsv | %s [options]' %(sys.argv[0].split('/')[-1])
    print
    print 'Writes to folders tbweights, te-parse, te-worker and te-combine'
    print """
Options:

    --sent-rep-dir  FOLDER  append FOLDER to the list of sentence representations
                            (if the list is empty the folder 'elmo' will be used)

    --past-log-dir  FOLDER  read log files from FOLDER and exclude completed experiments and scenarios

    --drop-data  NUMBER     skip fraction of the data points and training items and adjust k and max_samples
                            (default: 0.0 = use all data)

    --test-type  STRING     whether to test on the dev or test section of each treebank
                            (default: dev)

    --subsets-of-3          test all subsets of 3 non-pud treebanks as the model tbids
                            (default: test a single model using all non-pud tbids)

    --collection  STRING    append treebank collection STRING to the list of collections;
                            STRING needs to be a space- or colon-separated list of tbids
                            (if the list is empty, the Czech, English and French collections with non-pud
                            treebanks that do not withhold surface tokens will be used)

    --restrict-from  FILE   read scenario restrictions from FILE
                            (default: explore all scenarios)

    --weights  STRING       append weights STRING to the list of weights
                            (if the list is empty, all supported weights are explored)

    --max-k  NUMBER         limit k-NN retrieval to NUMBER items and use at most
                            NUMBER sentences in knnreduce
                            (default: 81)

    --explore-sent-rep      only explore no-neg knnreduce lpc L2 scenarios
                            (default: explore all scenarios)

    --sent-rep-offset NUMBER  add NUMBER to the list of first stage k-NN offsets
                            (if the list is empty 0 will be added)

    --skip-oracle           skip (fast-track) oracle scenarios
                            (default: include oracle runs, data points for test sets must exist)

    --parallel-n  NUMBER
    --parallel-i  NUMBER    Only run scenarios j with j % n == i
                            (default: n=1 and i=0, i.e. no parallelisation)

    --parallel-worker HOST:PORT  Contact task-farming master at HOST:PORT for the next scenario to work on
                            (default: use --parallel-n and --parallel-i above)

    Only for use with --parallel-worker:

    --secret STRING         unique string to make sure you communicate with the
                            right partner; This string will be
                            used to check the integrity of all messages sent over the
                            network using a cryptographic keyed-hash message
                            authentication code. It will not be sent over the network
                            itself. However, since the command line appears in the output of
                            various command like ps, the following option is prefered.

    --secret-from-file FILE
                            like --secret but read string from file (trailing whitespace and
                            additional lines will be ignored)

    --secret-from-variable NAME
                            like --secret but read string from environment variable NAME

    --limit  NUMBER         do not ask for another task if NUMBER hours have passed since
                            the worker has been started
                            (default: unlimited)

    --stop-file FILE        do not ask for a or another task and quit if FILE exists

    --max-tasks NUMBER      do not ask for another task after having run NUMBER tasks'
                            (default 0 = unlimited)

    --once                  equivalent to --max-tasks 1

    --worker-id STRING      send worker ID to master for information;
                            requires a master that supports receiving it

"""

test_type = 'dev'  # test on dev or test data?
opt_subsets_of_3 = False   # True: requires at least 4 treebanks; False: train on all non-pud treebanks
opt_fast_track_oracle = False
parser_start_seed = 300
parser_num_seeds  = 7
num_runs = 7       # how often to repeat each setting
opt_seed = 100
num_workers = 1
script_template_filename = 'test-tbemb-sampling-template.sh'
opt_count_scenarios      = False
opt_parallel_n           = 1
opt_parallel_i           = 0
opt_parallel_worker      = None
opt_secret = ''
opt_limit = None
opt_stop_file = None
opt_max_tasks = 0
opt_worker_id = None
opt_sleep_a = 0.5
opt_sleep_b = 1.0
opt_hash_length   = 768 //4    # must match the master's hash length
opt_nounce_length = 192 //4
opt_drop_data            = 0.0   # to speed up processing during dev
opt_sent_rep_dirs        = []    # a default dir may be added after options parsing
opt_sent_length_dir      = 'length-and-punct'
tbweights_dir            = 'tbweights'
opt_help                 = False
opt_debug                = False
opt_max_parses           = 1
opt_max_k                = 81
opt_knn_k1               = 500
opt_plain_first_stage_knn = True    # do not append length to sent rep vectors
opt_max_seeds            = 1
opt_min_data_size        = 10
epochs                   = 'best'
opt_collections          = []
opt_verbosity_interval   = 60.0
weights_with_priorities = [
    (20,  'corner'),
    (121, 'cplus1'),
    (141, 'cplus3'),
    (161, 'cplus4'),
    (323, 'no-p02'),
    (303, 'no-p01'),
    (282, 'no-neg'),
    (262, 'no-m01'),
    (242, 'no-m02'),
    (222, 'no-m03'),
    (202, 'no-m04'),
    (0,   'any'),
]
opt_prune_main_scenarios = 3 * len(weights_with_priorities)
opt_restrict_from        = None
opt_past_log_dirs        = []
opt_load_previously_completed_experiments = False
opt_load_previously_completed_scenarios   = True
opt_min_duration_for_completed_scenario   = 30.0
opt_maximise_random_sample_distance = True
opt_random_sample_distance_cutoff = 1.0
opt_random_sample_distance_cutoff_decay = 0.8
opt_force_oracle_rerun = False
opt_max_dev_set_size   = 1024000
opt_populate_knn_only_with_oracle_exact_match = True
opt_explore_sent_rep   = False
opt_sent_rep_offsets   = []
opt_weights            = []

# TODO: add options for all variables above
while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option in ('--parallel-n', '--parallel-generators'):
        opt_parallel_n = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--parallel-i', '--parallel-index'):
        opt_parallel_i = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--parallel-worker':
        opt_parallel_worker = sys.argv[1]
        del sys.argv[1]
    elif option == '--stop-file':
        opt_stop_file = sys.argv[1]
        del sys.argv[1]
    elif option == '--max-tasks':
        opt_max_tasks = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--once':
        opt_max_tasks = 1
    elif option == '--secret':
        opt_secret = sys.argv[1]
        del sys.argv[1]
    elif option == '--secret-from-file':
        f = open(sys.argv[1], 'rb')
        opt_secret = f.readline().rstrip()
        f.close()
        del sys.argv[1]
    elif option == '--secret-from-variable':
        opt_secret = os.env(sys.argv[1])
        del sys.argv[1]
    elif option == '--worker-id':
        opt_worker_id = sys.argv[1]
        del sys.argv[1]
    elif option == '--hash-length':
        opt_hash_length = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--nounce-length':
        opt_nounce_length = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--seed-length':
        opt_seed_length = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--limit':
        opt_limit = time.time() + 3600.0 * float(sys.argv[1])
        del sys.argv[1]
    elif option == '--sleep':
        opt_sleep_a = float(sys.argv[1])
        opt_sleep_b = float(sys.argv[2])
        del sys.argv[2]
        del sys.argv[1]
    elif option == '--sent-rep-dir':
        opt_sent_rep_dirs.append(sys.argv[1])
        del sys.argv[1]
    elif option == '--sent-rep-offset':
        opt_sent_rep_offsets.append(int(sys.argv[1]))
        del sys.argv[1]
    elif option == '--past-log-dir':
        opt_past_log_dirs.append(sys.argv[1])
        del sys.argv[1]
    elif option == '--max-k':
        opt_max_k = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--weights':
        opt_weights.append(sys.argv[1])
        del sys.argv[1]
    elif option == '--collection':
        opt_collections.append(sys.argv[1].replace(':', ' ').split())
        del sys.argv[1]
    elif option == '--drop-data':
        opt_drop_data = float(sys.argv[1])
        del sys.argv[1]
    elif option == '--restrict-from':
        opt_restrict_from = sys.argv[1]
        del sys.argv[1]
    elif option == '--test-type':
        test_type = sys.argv[1]
        del sys.argv[1]
    elif option in ('--skip-oracle', '--fast-track-oracle'):
        opt_fast_track_oracle = True
    elif option == '--explore-sent-rep':
        opt_explore_sent_rep = True
    elif option == '--subsets-of-3':
        opt_subsets_of_3 = True
    elif option == '--debug':
        opt_debug = True
    else:
        print 'Unsupported option %s' %option
        opt_help = True
        break

if len(sys.argv) != 1:
    opt_help = True

if opt_help:
    print_usage()
    sys.exit(0)

random.seed(opt_seed)
if opt_count_scenarios:
    num_workers = 0

opt_max_samples = min(24, opt_max_k)

if opt_parallel_worker and not opt_secret:
    sys.stderr.write('Warning: communication not protected\n')

if opt_worker_id and '\t' in opt_worker_id:
    raise ValueError, 'Worker ID must not contain tab characters'

opt_sleep_a, opt_sleep_b = min(opt_sleep_a, opt_sleep_b), max(opt_sleep_a, opt_sleep_b)
if opt_sleep_a < 0.0:
    raise ValueError, 'Cannot sleep negative number of seconds'
opt_sleep_offset = opt_sleep_b - opt_sleep_a

if not opt_sent_rep_offsets:
    opt_sent_rep_offsets.append(0)

if not opt_sent_rep_dirs:
    opt_sent_rep_dirs.append('elmo')

if not opt_collections:
    opt_collections.append(('fr_gsd',   'fr_partut',  'fr_sequoia', 'fr_spoken',))
    opt_collections.append(('en_ewt',   'en_gum',     'en_lines',   'en_partut',))
    opt_collections.append(('cs_cac',   'cs_cltt',    'cs_fictree', 'cs_pdt',))

worker2file = {}
for worker in range(num_workers):
    worker2file[worker] = StringIO.StringIO()
next_worker = 0
next_exp_id = 0

script_template = open(script_template_filename, 'rb').read()

for outdir in ('te-parse', 'te-worker', 'te-combine'):
    if not os.path.exists(outdir):
        os.mkdir(outdir)

if not opt_count_scenarios:
    if opt_parallel_n == 1:
        filename = 'te-combine/master.tsv'
    else:
        filename = 'te-combine/master-parallel-%d-of-%d.tsv' %(
            opt_parallel_i+1, opt_parallel_n
        )
    combine = open(filename, 'wb')

if opt_drop_data:
    opt_max_k       = int(opt_max_k       * (1.0-opt_drop_data))
    opt_max_samples = int(opt_max_samples * (1.0-opt_drop_data))
    sys.stderr.write('max_k and max_samples reduced to %d, %d to account for drop_data\n' %(opt_max_k, opt_max_samples))

if opt_max_k < opt_max_samples:
    raise ValueError, 'max_k must not be less than max_samples'

if tbweights_dir[0] != '/':
    if os.path.islink(tbweights_dir) \
    and os.readlink(tbweights_dir)[0] == '/':
        tbweights_dir = os.readlink(tbweights_dir).rstrip('/')
    else:
        tbweights_dir = os.path.abspath(tbweights_dir)
    sys.stderr.write('tbweights_dir adjusted to %s\n' %tbweights_dir)

if not os.path.exists(tbweights_dir):
    sys.stderr.write('tbweights_dir not found and not creating it as it usually should be a symlink to some other place\n')
    sys.exit(1)


def get_hash(s, m = 0):
    h = hashlib.sha256(s).hexdigest()
    i = int(h, 16)
    if m:
        return i % m
    return i

def hash_to_float(h):
    ''' converts a hash obtained from get_hash() with m = 0 to a
        number 0 <= x < 1 (or including 1.0 if the platform's
        floats have lower precision than bits in the hash)
    '''
    return float(h)/2**256

relevant_tbids = {}
for collection in opt_collections:
    for tbid in collection:
        relevant_tbids[tbid] = None

if opt_debug:
    print 'relevant tbids:', sorted(relevant_tbids.keys())
    sys.stdout.flush()

result_filenames = []
if not opt_count_scenarios:
    rf_index = 0
    while True:
        result_filename = sys.stdin.readline()
        if not result_filename:
            break
        result_filename = result_filename.rstrip()
        #                                      -3 -2       -1
        # ./results/en_sewta:en_sewtr:en_sewtw-on-en_sewte-dev.tsv
        fields = result_filename.split('/')[-1].replace(':', '-').split('-')
        dataset_type = fields[-1].split('.')[0]
        t = fields[-2]
        model_tbids = tuple(fields[:-3])
        is_relevant = t in relevant_tbids
        for tbid in model_tbids:
            if not tbid in relevant_tbids:
                is_relevant = False
                break
        if opt_subsets_of_3 and len(model_tbids) != 3:
            is_relevant = False
        if is_relevant:
            result_filenames.append((
                rf_index, result_filename,
                dataset_type, model_tbids, t,
            ))
            rf_index += 1
    result_filenames.sort()

if result_filenames:
    sys.stderr.write('= Reading available predictions =\n')
    if opt_drop_data:
        sys.stderr.write('Warning: dropping %.1f%% of data.\n' %(100.0*opt_drop_data))

available_predictions = {}
last_verbose = 0.0
item_count = 0
dataset2num_sentences = {}
for rf_index, result_filename, dataset_type, model_tbids, t in result_filenames:
    model_tbids = model_tbids
    f_size = os.path.getsize(result_filename)
    f = open(result_filename, 'rb')
    header = f.readline().split()
    seed_column = header.index('Seed')
    w_columns = []
    for i, m_tbid in enumerate(model_tbids):
        w_columns.append(header.index('Weight-%d-for-%s' %(i, m_tbid)))
    las_column = header.index('LAS-F1-total')
    item_subset_count = 0
    while True:
        if last_verbose < time.time() - opt_verbosity_interval:
            p1 = f.tell() / float(f_size)
            p2 = (rf_index+p1)/float(len(result_filenames))
            sys.stderr.write('%r on %s-%s: %.1f%% (%.1f%% overall), %d items (%d overall)\r' %(
                model_tbids, t, dataset_type, 100.0*p1, 100.0*p2, item_subset_count, item_count
            ))
            last_verbose = time.time()
        line = f.readline()
        if not line:
            p2 = (rf_index+1)/float(len(result_filenames))
            sys.stderr.write('%r on %s-%s: %.1f%% (%.1f%% overall), %d items (%d overall)\n' %(
                model_tbids, t, dataset_type, 100.0, 100.0*p2, item_subset_count, item_count
            ))
            break
        if opt_drop_data and random.random() < opt_drop_data:
            continue
        fields = line.split()
        seed = fields[seed_column]
        if seed.startswith('Median'):
            continue
        item_count += 1
        item_subset_count += 1
        wvec = []
        for w_column in w_columns:
            w_str = fields[w_column]
            # as we optionally filter negative weights somewhere below, we
            # want to treat values that are only negative due to numerical
            # instability as non-negative
            if w_str.startswith('-0.000000'):
                w_str = '0.' + (len(w_str)-3) * '0'
            wvec.append(float(w_str))
        wvec = tuple(wvec)
        scores = []
        for i in range(las_column + 1, len(fields)):
            if header[i].startswith('Sent-'):
                scores.append(float(fields[i]))
        num_sentences = len(scores)
        key = (t, dataset_type)
        if key in dataset2num_sentences:
            if dataset2num_sentences[key] != num_sentences:
                raise ValueError, 'Number of test sentences does not match previous reading'
        else:
            dataset2num_sentences[key] = num_sentences
        # append overall LAS as final element
        # (can be accessed with [-1])
        scores.append(float(fields[las_column]))
        # store LAS list under key and seed
        key  = (model_tbids, t, dataset_type)
        if not key in available_predictions:
            available_predictions[key] = {}
        seed2results = available_predictions[key]
        if not seed in seed2results:
            seed2results[seed] = {}
        if wvec in seed2results[seed]:
            raise ValueError, 'duplicate scores for %r, seed %s, wvec %r' %(key, seed, wvec)
        seed2results[seed][wvec] = scores
    f.close()

completed_experiments = {}
completed_scenarios   = {}
# load list of completed experiments from outputs of previous run(s)
start_loading_logs = time.time()
for log_dir in opt_past_log_dirs:
    print 'Loading past logs from', log_dir
    num_files = 0
    num_scenarios = 0
    num_experiments = 0
    for filename in os.listdir(log_dir):
        if not filename.endswith('.txt'):
            continue
        filename = os.path.join(log_dir, filename)
        f = open(filename, 'rb')
        num_files += 1
        for line in f:
            if opt_load_previously_completed_scenarios \
            and line.startswith('Duration of gen_tb'):
                # 0        1  2                  3   4        5   6
                # Duration of gen_tbemb_sampling for scenario %d: 200.8s'
                fields = line.split()
                if fields[4] == 'scenario':
                    scenario = int(fields[5].split(':')[0])
                    duration = float(fields[6].rstrip('s'))
                    if duration >= opt_min_duration_for_completed_scenario:
                        if scenario in completed_scenarios:
                            raise ValueError, 'Scenario %d reported as completed in both %r and %r' %(
                                    scenario, filename, completed_scenarios[scenario])
                        completed_scenarios[scenario] = filename
                        num_scenarios += 1
            elif opt_load_previously_completed_experiments \
            and  line.startswith('Wrote tbweights and'):
                # 0     1         2   3 4     5         6   7             8
                # Wrote tbweights and 1 parse script(s) for experiment_id 181
                fields = line.split()
                if fields[7] == 'experiment_id':
                    experiment_id = int(fields[8])
                    if experiment_id in completed_experiments:
                        raise ValueError, 'Experiment %d reported as completed in both %r and %r' %(
                                experiment_id, filename, completed_experiments[experiment_id])
                    completed_experiments[experiment_id] = filename
                    num_experiments +=1
        f.close()
    print ' --> %d files, %d completed scenarios, %d completed experiments' %(
            num_files, num_scenarios, num_experiments
    )

allowed_scenarios = {}
if opt_restrict_from:
    f = open(opt_restrict_from, 'rb')
    while True:
        line = f.readline()
        if not line:
            break
        if line.startswith('#'):
            continue
        fields = line.rstrip().split('\t')
        if len(fields) != 7:
            raise ValueError, 'wrong format in %r: %r' %(opt_restrict_from, line)
        for column in (1,2):
            if fields[column].startswith('tr'):
                fields[column] = 'treebank'
            if fields[column].startswith('se'):
                fields[column] = 'sentence'
        if fields[3] == 'corners':
            fields[3] = 'corner'
        if fields[5] == 'SphL2':
            fields[5] = 'SphereL2'
        allowed_scenarios[tuple(fields)] = None
        # also account for oracle runs, which set reranking, metric and
        # sentence representation to 'none'
        fields[4] = 'none'
        fields[5] = 'none'
        fields[6] = 'none'
        allowed_scenarios[tuple(fields)] = None
    f.close()
    if opt_debug:
        print 'allowed scenarios:'
        for i, key in enumerate(sorted(allowed_scenarios.keys())):
            print '[%d]:\t%s' %(i, '\t'.join(key))
        sys.stdout.flush()

if opt_past_log_dirs:
    duration = time.time() - start_loading_logs
    print 'Loaded past logs in %.1fs' %duration

def get_nn_and_distance(points, query_vec):
    min_d2 = 9999.9
    nn = None
    for p in points:
        d2 = 0.0
        for i, qv in enumerate(query_vec):
            d2 += (p[i]-qv)**2
        if d2 < min_d2:
            min_d2 = d2
            nn = p
    return nn, min_d2**0.5

pick3from9 = [
    (0,1,2), (3,4,5), (6,7,8),
    (0,3,6), (1,4,7), (2,5,8),
    (0,4,8), (1,5,6), (2,3,7),
    (0,5,7), (1,3,8), (2,4,6),
]

def incr_counter(d, name, index):
    key = (name, index)
    try:
        count = d[key]
    except KeyError:
        count = 0
    d[key] = count + 1
    key = (name, 'max')
    try:
        max_index = d[key]
    except KeyError:
        max_index = index
    d[key] = max(max_index, index)
    key = (name, 'min')
    try:
        min_index = d[key]
    except KeyError:
        min_index = index
    d[key] = min(min_index, index)
    key = 'names'
    try:
        names = d[key]
    except:
        names = {}
    names[name] = None
    d[key] = names

def print_counters(d):
    for name in sorted(d['names']):
        max_count = 0
        for i in range(d[(name, 'max')] + 1):
            try:
                count = d[(name, i)]
            except KeyError:
                count = 0
            if count > max_count:
                max_count = count
        if max_count < 60:
            max_count = 60
        for i in range(d[(name, 'min')], d[(name, 'max')] + 1):
            try:
                count = d[(name, i)]
            except KeyError:
                count = 0
            bar = int(count*80//max_count) * '*'
            print '%s[%3d] = %6d |%s' %(name, i, count, bar)

# prioritise data-rich and oracle and not no-neg
main_scenarios = []
for priority1, learning in [
    (0,    'data-rich'),
    (10,   'no-learning'),  #  for methods pick*, random etc.
    (1000, 'pedantic'),
    (100,  'oracle'),
]:
    for priority2, scenario, short_scenario in (
        (0,   'out-of-domain', 'oodom'),
        (500, 'in-domain',     'indom'),
    ):
        for priority3, allowed_weights in weights_with_priorities:
            main_scenarios.append((
                priority1 + priority2 + priority3,
                scenario, short_scenario,
                allowed_weights,
                learning
            ))

# sort by priority
main_scenarios.sort()

print
print 'Order of main scenarios:'
for index, scenario in enumerate(main_scenarios):
    skipping = '- will be skipped' if (opt_prune_main_scenarios and index >= opt_prune_main_scenarios) else ''
    print scenario, skipping

collections_and_main_scenarios = []
for collection in opt_collections:
    main_scenario_index = 0
    for _, scenario, short_scenario, allowed_weights, learning in main_scenarios:
        if opt_prune_main_scenarios:
            is_pruned_scenario = main_scenario_index >= opt_prune_main_scenarios
        else:
            is_pruned_scenario = False
        collections_and_main_scenarios.append((
            collection,
            scenario, short_scenario,
            allowed_weights, learning,
            is_pruned_scenario,
        ))
        main_scenario_index += 1

def distance(p1, p2):
    d2 = 0.0
    for i, v in enumerate(p1):
        d2 += (v-p2[i])**2
    return d2**0.5

def can_use_weights(wvec, allowed_weights):
    if allowed_weights == 'any':
        return True
    if allowed_weights == 'no-p02':
        return min(wvec) >= 0.2
    if allowed_weights == 'no-p01':
        return min(wvec) >= 0.1
    if allowed_weights == 'no-neg':
        return min(wvec) >= 0.0
    if allowed_weights == 'no-m01':
        return min(wvec) >= -0.1
    if allowed_weights == 'no-m02':
        return min(wvec) >= -0.2
    if allowed_weights == 'no-m03':
        return min(wvec) >= -0.3
    if allowed_weights == 'no-m04':
        return min(wvec) >= -0.4
    if allowed_weights == 'corner':
        # allow for small numerical errors
        return abs(min(wvec)) < 0.000001 and abs(max(wvec)-1.0) < 0.000001
    if allowed_weights == 'cplus1':
        # check for uniform vector
        if abs(max(wvec) - min(wvec)) < 0.000001:
            return True
        return can_use_weights(wvec, 'corner')
    if allowed_weights == 'cplus3':
        # check for vector that is the average of two base vectors
        if abs(min(wvec)) < 0.000001 and abs(max(wvec)-0.5) < 0.000001:
            return True
        return can_use_weights(wvec, 'corner')
    if allowed_weights == 'cplus4':
        return can_use_weights(wvec, 'cplus1') or can_use_weights(wvec, 'cplus3')
    raise ValueError, 'Unknown value %r for `allowed_weights`' %allowed_weights

def normalise_all_rows(matrix):
    # https://stackoverflow.com/questions/36267936/normalizing-rows-of-a-matrix-python
    # https://stackoverflow.com/questions/14861891/runtimewarning-invalid-value-encountered-in-divide
    from numpy.linalg import norm
    l2norm = norm(matrix, axis=1, ord=2)
    with numpy.errstate(divide='ignore',invalid='ignore'):
        retval = matrix / (l2norm[:,None])
    numpy.nan_to_num(retval, copy=False)
    return retval

def knnreduce_intersect(
    model_tbids, knn_tbid, knn_dataset_type, seed,
    available_predictions, allowed_weights, knn_tr_index,
    k_index, sentence_index, intersection
):
    knn_key = (model_tbids, knn_tbid, knn_dataset_type)
    try:
        wvec2scores = available_predictions[knn_key][seed]
    except:
        wvec2scores = {}
    if not wvec2scores:
        sys.stderr.write('No data points for %s training sentence %d with seed %s in knnreduce for k_index %d of sentence_index %d\n' %(
            knn_tbid, knn_tr_index, seed, k_index, sentence_index
        ))
        return intersection
    invscores_and_wvec = []
    for wvec in wvec2scores:
        if can_use_weights(wvec, allowed_weights):
            # intersect with previous list if there is one
            if intersection and wvec not in intersection:
                continue
            score = wvec2scores[wvec][knn_tr_index]
            invscores_and_wvec.append((-score, wvec))
    intersection = {}
    if not invscores_and_wvec:
        sys.stderr.write('No data points for %s training sentence %d with seed %s in knnreduce for k_index %d of sentence_index %d\n' %(
            knn_tbid, knn_tr_index, seed, k_index, sentence_index
        ))
        return intersection
    invscores_and_wvec.sort()
    top_invscore = invscores_and_wvec[0][0]
    for invscore, wvec in invscores_and_wvec:
        # only add weight vectors that match the top score
        if invscore != top_invscore:
            break
        intersection[wvec] = None
    return intersection

def get_corner_names_and_vec(m = 3):
    if m > 5:
        raise NotImplementedError
    retval = []
    for i, name in enumerate('1st 2nd 3rd 4th 5th'.split()):
        if i == m:
            break
        corner = []
        for j in range(m):
            if i == j:
                corner.append(1.0)
            else:
                corner.append(0.0)
        retval.append((name, tuple(corner)))
    return retval

def get_random_hexstring(n=72):
    global opt_secret
    retval = []
    while n > 0:
        m = hashlib.new('sha256')
        m.update('%d:' %len(opt_secret))
        m.update(opt_secret)
        for i in range(20):
            m.update('%.15f:' %random.random())
        m.update('%.15f' %time.time())
        block = m.hexdigest()[:n]
        n -= len(block)
        retval.append(block)
    return ''.join(retval)

def p_sign(message, nounce = None):
    global opt_secret
    if nounce is None:
        nounce = get_random_hexstring()
    folded_nounce = hashlib.sha256(nounce).hexdigest()
    m = hashlib.new('sha256')
    m.update('%d:' %len(opt_secret))
    m.update(opt_secret)
    m.update(folded_nounce)
    m.update(message)
    hexdigest = m.hexdigest()[:opt_hash_length]
    if opt_debug:
        sys.stderr.write('signing message %r with nounce %r as %r\n' %(
            message, nounce, hexdigest
        ))
    return '%s:%s' %(nounce, hexdigest)

def is_valid_signature(message, signature):
    fields = signature.split(':')
    if len(fields) != 2:
        return False
    new_signature = p_sign(message, fields[0])
    return signature == new_signature

if opt_parallel_worker:
    import base64
    import xmlrpclib
    taskfarming_master = xmlrpclib.Server('http://' + opt_parallel_worker)
    last_challenge, signature = taskfarming_master.get_challenge()
    if not is_valid_signature(last_challenge, signature):
        raise ValueError, 'received invalid signature %r from taskfarming_master.get_challenge() for %r' %(signature, last_challenge)

task_count = 0
taskfarming_finished = False

def get_next_scenario():
    global taskfarming_master
    global last_challenge
    global task_count
    global taskfarming_finished
    if taskfarming_finished:
        return -1
    if opt_limit and time.time() > opt_limit:
        sys.stderr.write('taskfarming: reached time limit\n')
        taskfarming_finished = True
        return -1
    if opt_stop_file and os.path.exists(opt_stop_file):
        sys.stderr.write('taskfarming: found stop file\n')
        taskfarming_finished = True
        return -1
    if opt_max_tasks and (task_count >= opt_max_tasks):
        sys.stderr.write('taskfarming: reached max tasks\n')
        taskfarming_finished = True
        return -1
    response = p_sign(last_challenge, '!')
    if opt_worker_id:
        message = '%s\t%s' %(opt_worker_id, response)
    else:
        message = response
    signature = p_sign(message)
    task = taskfarming_master.get_next_task(message, signature)
    if not task:
        # no more tasks
        sys.stderr.write('taskfarming: no more tasks\n')
        taskfarming_finished = True
        return -1
    task, sig, cs = task
    if not is_valid_signature(task, sig):
        raise ValueError, 'wrong signature %r for task %r' %(sig, task)
    retval = int(task)
    task_count += 1
    last_challenge, signature = cs
    if not is_valid_signature(last_challenge, signature):
        raise ValueError, 'received invalid signature %r from taskfarming_master.get_challenge() for %r' %(signature, last_challenge)
    if opt_sleep_b:
        sleep_amount = opt_sleep_a + opt_sleep_offset * random.random()
        time.sleep(sleep_amount)
    return retval

if opt_parallel_worker:
    next_scenario = get_next_scenario()

scenario_count = 0
main_scenario_count = 0

for collection, scenario, short_scenario, allowed_weights, learning, is_pruned_scenario in collections_and_main_scenarios:
    if opt_parallel_worker and next_scenario < 0:
        break
    main_scenario_count += 1
    print
    print '= Main Scenario Partition %d of %d =' %(main_scenario_count, len(collections_and_main_scenarios))
    print

    sub_scenarios = []
    model_tbids_list = []
    if opt_subsets_of_3:
        k = len(collection)
        for i1 in range(k-2):
            s1 = collection[i1]
            if s1.endswith('_pud'):
                continue
            for i2 in range(i1+1, k-1):
                s2 = collection[i2]
                if s2.endswith('_pud'):
                    continue
                for i3 in range(i2+1, k):
                    s3 = collection[i3]
                    if s3.endswith('_pud'):
                        continue
                    model_tbids_list.append((s1, s2, s3))
    else:
        # use all non-pud tbids for model_tbids:
        model_tbids = []
        for tbid in collection:
            if not tbid.endswith('_pud'):
                model_tbids.append(tbid)
        model_tbids_list.append(tuple(model_tbids))

    for model_tbids in model_tbids_list:
        for sro_index, sent_rep_offset in enumerate(opt_sent_rep_offsets):
            is_first_sent_rep_offset = not sro_index
            # keep old indention
            if True:
                if True:
                    test_sets = []
                    for candidate_test_set in collection:
                        if scenario == 'out-of-domain' \
                        and candidate_test_set not in model_tbids \
                        or scenario == 'in-domain' \
                        and candidate_test_set in model_tbids:
                            test_sets.append(candidate_test_set)
                    if learning == 'no-learning':
                        if not is_first_sent_rep_offset:
                            continue
                        sub_scenarios.append((
                            model_tbids, test_sets,
                            None, None, None, None, None,
                            None,
                        ))
                        continue
                    elif learning == 'oracle':
                        oracle_variants = []
                        # for retrieval of sentences for sentences in oracle
                        # mode, we can choose between only using the exact
                        # match and normal k-NN (augmented with the test set)
                        if opt_populate_knn_only_with_oracle_exact_match:
                            if not is_first_sent_rep_offset:
                                continue
                            sub_scenarios.append((
                                model_tbids, test_sets,
                                None, None, False,
                                'sentence', 'sentence',
                                None,
                            ))
                        else:
                            oracle_variant.append((True, 'sentence', 'sentence'))
                        oracle_variants.append((False, 'treebank', 'sentence'))
                        oracle_variants.append((False, 'treebank', 'treebank'))
                        for use_length, reference_unit, query_unit in oracle_variants:
                            if reference_unit == 'treebank' and not is_first_sent_rep_offset:
                                continue
                            for opt_sent_rep_dir in opt_sent_rep_dirs:
                                sub_scenarios.append((
                                    model_tbids, test_sets,
                                    opt_sent_rep_dir, 'SphereL2',
                                    use_length, reference_unit, query_unit,
                                    sent_rep_offset,
                                ))
                        continue
                    for opt_sent_rep_dir in opt_sent_rep_dirs:
                        for sent_rep_metric in ('L2', 'SphereL2', 'Random'):
                            for use_length in (True, False):
                                sub_scenarios.append((
                                    model_tbids, test_sets,
                                    opt_sent_rep_dir,
                                    sent_rep_metric,
                                    use_length,
                                    'sentence', 'sentence',
                                    sent_rep_offset,
                                ))
                        if not is_first_sent_rep_offset:
                            continue
                        for query_unit in ('sentence', 'treebank'):
                            sub_scenarios.append((
                                model_tbids, test_sets,
                                opt_sent_rep_dir,
                                'SphereL2', False,
                                'treebank', query_unit,
                                sent_rep_offset,
                            ))
    for model_tbids, test_sets, opt_sent_rep_dir, sent_rep_metric, use_length, \
    reference_unit, query_unit, sent_rep_offset in sub_scenarios:
        if opt_parallel_worker and next_scenario < 0:
            break
        model_tbids_str = ':'.join(model_tbids)
        num_corners = len(model_tbids)
        centre = tuple(num_corners * [1.0/num_corners])
        corner_names_and_vec = get_corner_names_and_vec(num_corners)
        if opt_sent_rep_dir:
            sent_rep_name = opt_sent_rep_dir.split('/')[-1]
        else:
            sent_rep_name = 'none'
        for t in test_sets:
            scenario_count += 1
            start_scenario = time.time()
            if not opt_count_scenarios:
                if opt_parallel_worker:
                    is_selected_scenario = False
                    if scenario_count >= next_scenario:
                        if scenario_count == next_scenario:
                            is_selected_scenario = True
                        else:
                            sys.stderr.write(
                                'Warning: Task-farming master gave us scenario '
                                '%d to work on but this scenario does not exist.\n' %next_scenario
                            )
                        next_scenario = get_next_scenario()
                else:
                    rr_parallel_step = scenario_count % opt_parallel_n
                    is_selected_scenario = (rr_parallel_step == opt_parallel_i)
                is_completed_scenario = scenario_count in completed_scenarios
                if opt_force_oracle_rerun and learning == 'oracle':
                    # force re-running of oracle scenarios
                    is_completed_scenario = False
                fast_track_scenario = False
                fast_track_reasons = []
                if is_pruned_scenario:
                    fast_track_scenario = True
                    fast_track_reasons.append('pruned scenario')
                if is_completed_scenario:
                    fast_track_scenario = True
                    fast_track_reasons.append('completed scenario')
                if not is_selected_scenario:
                    fast_track_scenario = True
                    fast_track_reasons.append('not a selected scenario')
                if opt_fast_track_oracle and learning == 'oracle':
                    fast_track_scenario = True
                    fast_track_reasons.append('oracle scenario and --skip-oracle was specified')
                if allowed_scenarios and learning != 'no-learning':
                    # check list of allowed scenarios
                    scenario_is_allowed = False
                    if use_length:
                        reranking = 'lpc'
                    else:
                        reranking = 'none'
                    if not sent_rep_metric:
                        srm_name = 'none'
                    else:
                        srm_name = sent_rep_metric
                    for lcode in ('*', t.split('_')[0]):
                        scenario_key = (
                            lcode, reference_unit, query_unit,
                            allowed_weights,
                            reranking, srm_name, sent_rep_name
                        )
                        if scenario_key in allowed_scenarios:
                            scenario_is_allowed = True
                            break
                    if not scenario_is_allowed:
                        fast_track_scenario = True
                        fast_track_reasons.append('scenario not in list of allowed scenarios (--restrict-from)')
                if opt_explore_sent_rep:
                    # only for no-neg lpc L2
                    if allowed_weights != 'no-neg' \
                    or learning != 'data-rich' \
                    or sent_rep_metric != 'L2' \
                    or not use_length:
                        fast_track_scenario = True
                        fast_track_reasons.append('exploring sentence representations')
                if opt_weights and allowed_weights not in opt_weights:
                    fast_track_scenario = True
                    fast_track_reasons.append('weights not in list provided with --weights')
                # human-readable description of scenario
                setting = []
                setting.append('%s setting' %scenario)
                setting.append('%r on %s' %(model_tbids, t))
                if learning in ('no-learning', 'oracle'):
                    setting.append(learning)
                if learning != 'no-learning':
                    setting.append('retrieving %s for %s' %(reference_unit, query_unit))
                if learning not in ('no-learning', 'oracle'):
                    setting.append('with sentence representation %s' %sent_rep_name)
                    setting.append('and %s metric' %sent_rep_metric)
                    if sent_rep_offset is not None:
                        setting.append('and offset %d' %sent_rep_offset)
                    if use_length:
                        setting.append('re-ranked by length, punctuation and cosine distance')
                setting.append('restricted to %s weights' %allowed_weights)
                setting.append('(scenario %d)' %scenario_count)
                if opt_debug or not fast_track_scenario:
                    print
                    print '== Scenario %d ==' %scenario_count
                    print
                    print '\n    '.join(setting[:-1])
                    print
                    if opt_debug and fast_track_scenario:
                        print 'Fast-tracked. Reasons:', fast_track_reasons
                    else:
                        now = time.time()
                        print 'Time: %.2f %s' %(now, time.ctime(now))
                    print
                    sys.stdout.flush()
                setting = ', '.join(setting)
            if fast_track_scenario or opt_count_scenarios \
            or learning == 'no-learning':
                build_knn = 'none'
            elif learning != 'oracle':
                build_knn = 'full'
            elif reference_unit == 'treebank' or query_unit == 'treebank':
                build_knn = 'full'
            elif opt_populate_knn_only_with_oracle_exact_match:
                build_knn = 'oracle-exact-match-only'
            else:
                build_knn = 'full'
            first_stage_knn = None
            first_stage_rnd = False
            test_data_emb = None
            test_data_length = None
            test_data_punct = None
            knn_training_sets = None
            try:
                num_test_sentences = dataset2num_sentences[(t, test_type)]
            except KeyError:
                if t.endswith('_pud'):
                    num_test_sentences = 1000
                    if not fast_track_scenario:
                        sys.stderr.write('Warning: pud test set size not found; assuming 1000 sentences\n')
                elif fast_track_scenario:
                    num_test_sentences = 0
                else:
                    raise ValueError, 'number of sentences of %s %s not available' %(t, test_type)
            if build_knn == 'full':
                # get test data sentence representations
                import h5py
                import scipy.spatial
                import sklearn.neighbors
                import numpy
                start_loading = time.time()
                sys.stderr.write('Loading test data %s-%s sentence representations...\n' %(t, test_type))
                vecs_file = '%s/%s-ud-%s-sent-rep.hdf5' %(opt_sent_rep_dir, t, test_type)
                test_data_emb = h5py.File(vecs_file, 'r')['sent_rep']
                if num_test_sentences != test_data_emb.shape[0]:
                    raise ValueError, 'Wrong number of sentences in %s' %vecs_file
                if query_unit == 'treebank':
                    # aggregate sentence representations to obtain treebank representation
                    test_data_emb = numpy.array(test_data_emb).sum(
                        axis = 0, keepdims = True
                    )
                    # and duplicate rows so that we do not need to change the query code
                    test_data_emb = numpy.repeat(
                        test_data_emb, num_test_sentences, axis = 0
                    )
                    if use_length:
                        raise ValueError, 'cannot combine use_length with treebank queries'
                    if sent_rep_metric != 'SphereL2':
                        raise ValueError, 'cannot use metric %s with treebank queries' %sent_rep_metric
                if sent_rep_metric == 'SphereL2':
                    test_data_emb = normalise_all_rows(test_data_emb)
            if True:
                # update num_test_sentences the same way in all modes
                if opt_drop_data:
                    num_test_sentences = int(num_test_sentences*(1-opt_drop_data))
            if build_knn == 'full':
                if use_length:
                    vecs_file = '%s/%s-ud-%s-length.hdf5' %(opt_sent_length_dir, t, test_type)
                    test_data_length = h5py.File(vecs_file, 'r')['sent_length']
                    test_data_punct  = h5py.File(vecs_file, 'r')['sent_punct']
                # get training data sentence representations
                knn_training_sets = []
                total_tr_size = 0
                for candidate_training_set in collection:
                    # decide whether to load the candidate set and what
                    # section (train, dev or test)
                    if candidate_training_set.endswith('_pud') \
                    and not (candidate_training_set == t and learning == 'oracle'):
                        # PUD has no k-NN training data
                        continue
                    if candidate_training_set == t and learning != 'oracle':
                        continue
                    if learning == 'pedantic' \
                    and candidate_training_set in model_tbids:
                        continue
                    if candidate_training_set == t and learning == 'oracle':
                        dataset_type = test_type
                        dataset_description = 'oracle'
                        is_oracle_dataset = True
                    else:
                        dataset_type = 'train'
                        dataset_description = 'training'
                        is_oracle_dataset = False
                    sys.stderr.write('Loading %s data %s-%s sentence representations...\n' %(
                        dataset_description, candidate_training_set, dataset_type
                    ))
                    vecs_file = '%s/%s-ud-%s-sent-rep.hdf5' %(
                        opt_sent_rep_dir, candidate_training_set, dataset_type
                    )
                    tr_data_emb = h5py.File(vecs_file, 'r')['sent_rep']
                    if reference_unit == 'treebank':
                        # aggregate sentence representations to obtain treebank representation
                        tr_data_emb = numpy.array(tr_data_emb).sum(
                            axis = 0, keepdims = True
                        )
                        if use_length:
                            raise ValueError, 'cannot combine use_length with full treebanks as reference itmes'
                        if sent_rep_metric != 'SphereL2':
                            raise ValueError, 'cannot use metric %s with full treebanks as reference itmes' %sent_rep_metric
                    if sent_rep_metric == 'SphereL2':
                        tr_data_emb = normalise_all_rows(tr_data_emb)
                    if use_length:
                        vecs_file = '%s/%s-ud-%s-length.hdf5' %(
                            opt_sent_length_dir, candidate_training_set, dataset_type
                        )
                        tr_data_length = h5py.File(vecs_file, 'r')['sent_length']
                        tr_data_punct  = h5py.File(vecs_file, 'r')['sent_punct']
                    else:
                        tr_data_length = None
                        tr_data_punct  = None
                    knn_training_sets.append((
                        candidate_training_set, tr_data_emb,
                        tr_data_length, tr_data_punct,
                        dataset_type,
                        is_oracle_dataset,
                    ))
                    total_tr_size += tr_data_emb.shape[0]
                now = time.time()
                print 'Duration of loading representations: %.1fs' %(now-start_loading)
                sys.stdout.flush()
                if opt_knn_k1 < total_tr_size   \
                and sent_rep_metric == 'Random':
                    first_stage_knn = None
                    first_stage_rnd = True
                elif opt_knn_k1 < total_tr_size:
                    # prepare first-stage k-NN that will be used to
                    # narrow down the candidate set to perform final k-NN
                    # in acceptable time
                    # (1) concatenate arrays and keep track of indices
                    if opt_plain_first_stage_knn or not use_length:
                        blocks_tr = map(lambda x: x[1], knn_training_sets)
                        blocks_te = [test_data_emb]
                    else:
                        blocks_tr = []
                        blocks_te = []
                        for _, tr_data_emb, tr_data_length, tr_data_punct, _, _ in knn_training_sets:
                            length_and_punct = numpy.stack(
                                    [tr_data_length, tr_data_punct], axis=0
                            ).transpose()
                            print 'tr_data_emb.shape', tr_data_emb.shape
                            print 'length_and_punct.shape', length_and_punct.shape
                            blocks_tr.append(numpy.concatenate(
                                    [tr_data_emb, 10*length_and_punct], axis=1
                            ))
                            print '  --> concat.shape', blocks_tr[-1].shape
                            print
                        # same for test data
                        length_and_punct = numpy.stack(
                                [test_data_length, test_data_punct], axis=0
                        ).transpose()
                        print 'test_data_emb.shape', test_data_emb.shape
                        print 'length_and_punct.shape', length_and_punct.shape
                        blocks_te.append(numpy.concatenate(
                                [test_data_emb, 10*length_and_punct], axis=1
                        ))
                        print '  --> concat.shape', blocks_te[-1].shape
                        print
                    knn_first_stage_tr_data = numpy.concatenate(
                        blocks_tr, axis = 0
                    )
                    print 'knn_first_stage_tr_data.shape', knn_first_stage_tr_data.shape
                    knn_first_stage_te_data = numpy.concatenate(
                        blocks_te, axis = 0
                    )
                    print 'knn_first_stage_te_data.shape', knn_first_stage_te_data.shape
                    # (2) build first-stage k-NN model
                    start_fs_knn = time.time()
                    first_stage_knn = sklearn.neighbors.BallTree(
                        knn_first_stage_tr_data, leaf_size = 5
                    )
                    duration = time.time() - start_fs_knn
                    print 'time to build knn model: %.1fs' %duration
                    start_fs_knn = time.time()
                    if sent_rep_offset is None:
                        raise ValueError, 'sent_rep_offset is None where it should be a number'
                    first_stage_knn_results = first_stage_knn.query(
                        knn_first_stage_te_data,
                        k = opt_knn_k1 + sent_rep_offset,
                        return_distance = False,
                        sort_results = False
                    )
                    duration = time.time() - start_fs_knn
                    print 'time to query knn model: %.1fs' %duration
                    print 'first_stage_knn_results.shape', first_stage_knn_results.shape
                    if sent_rep_offset > 0:
                        first_stage_knn_results = first_stage_knn_results[:,sent_rep_offset:]
                        print 'prunned to', first_stage_knn_results.shape
                    print
                    sys.stdout.flush()
                    first_stage_rnd = False
            if build_knn in ('full', 'oracle-exact-match-only'):
                # get k-NN in training data for each test item
                # (or, in oracle mode, populate k-NN table with
                # exact match)
                now = time.time()
                print 'Time: %.2f %s' %(now, time.ctime(now))
                sys.stdout.flush()
                sys.stderr.write('Building k-NN lookup for %s --> %s queries from %s-%s...\n' %(
                    query_unit, reference_unit, t, test_type
                ))
                sentence_knn = []
                last_verbose = 0.0
                last_index_verbose = -1
                num_zero_te_sent_vec = 0
                zero_tr_sent_vecs = {}
                knn_tie_stats = {}
                rnd_stats = {}
                restrict_reranking = False
            if build_knn == 'full':
                if num_test_sentences > opt_max_dev_set_size:
                    # Too many test sentences to perform expensive k-NN
                    # for all.
                    rnd = random.Random()
                    # Must use same random number generator seed in all scenarios
                    # for results to be comparable.
                    rnd.seed(opt_seed)
                    restrict_reranking = set(rnd.sample(
                        xrange(num_test_sentences), opt_max_dev_set_size
                    ))
            if build_knn == 'oracle-exact-match-only':
                # fake tables for fast oracle mode
                import numpy
                test_data_emb = numpy.zeros((num_test_sentences, 4))
                first_stage_knn = None
                first_stage_rnd = False
                knn_training_sets = [(
                    t, test_data_emb,
                    None, None, test_type, True
                )]
            if build_knn in ('full', 'oracle-exact-match-only'):
                import scipy.spatial
                for test_index in range(num_test_sentences):
                    if last_verbose < time.time() - opt_verbosity_interval:
                        if last_index_verbose < 0:
                            eta = ''
                        else:
                            duration = time.time() - last_verbose
                            sentences_in_interval = test_index - last_index_verbose
                            speed = sentences_in_interval / duration
                            remaining = num_test_sentences - test_index
                            eta = 'ETA %s' %time.ctime(time.time() + remaining / speed)
                        sys.stderr.write('%d query sentence(s) done (%.1f%%) %s  \r' %(
                            test_index, 100.0*test_index/num_test_sentences, eta
                        ))
                        last_verbose = time.time()
                        last_index_verbose = test_index
                    test_sent_vec = test_data_emb[test_index]
                    if use_length:
                        test_sent_length = test_data_length[test_index]
                        test_sent_punct  = test_data_punct[test_index]
                    if (sent_rep_metric == 'cosine' or use_length) \
                    and scipy.linalg.norm(test_sent_vec) < 0.0000001:
                        test_sent_vec = None
                        num_zero_te_sent_vec += 1
                    candidate_otr_indices = None
                    if first_stage_knn is not None:
                        # opt_knn_k1 knn items:
                        candidate_otr_indices = first_stage_knn_results[test_index]
                        # convert to dictionary for fast membership test
                        # and keep record of position in k-NN list
                        # (so that it can be used as a preference indicator
                        # when re-ranking the first stage results)
                        candidate_otr_indices = {
                            index: k for k, index in enumerate(candidate_otr_indices)
                        }
                    if first_stage_rnd:
                        candidate_otr_indices = {}
                        # opt_knn_k1 random items:
                        for knn_rnd_index in range(opt_knn_k1):
                            otr_index = get_hash(
                                '%s:%d:%d:%s' %(
                                    sent_rep_name, knn_rnd_index, test_index, t
                                ),
                                total_tr_size
                            )
                            attempts = 1
                            while otr_index in candidate_otr_indices:
                                attempts += 1
                                if attempts >= 100:
                                    break
                                offset = 1 + get_hash(
                                    '%s:%d:%d:%d:%s' %(
                                        sent_rep_name, knn_rnd_index,
                                        test_index, attempts, t
                                ))
                                otr_index = (otr_index + offset) % total_tr_size
                            candidate_otr_indices[otr_index] = knn_rnd_index
                            incr_counter(rnd_stats, 'sampling_attempts', attempts)
                    knn = []
                    overall_tr_index = 0
                    for knn_training_set in knn_training_sets:
                        candidate_training_set, tr_data_emb, \
                            tr_data_length, tr_data_punct,   \
                            dataset_type,                    \
                            is_oracle_dataset = knn_training_set
                        for tr_index in range(tr_data_emb.shape[0]):
                            if reference_unit == 'treebank':
                                if tr_index:
                                    raise ValueError, 'Non-zero tr_index %r for treebank reference %r (1)' %(
                                        tr_index, candidate_training_set
                                    )
                                tr_index = -1
                            if is_oracle_dataset and tr_index == test_index \
                            and reference_unit == 'sentence' \
                            and query_unit == 'sentence' \
                            or is_oracle_dataset \
                            and reference_unit == 'treebank' \
                            and query_unit == 'treebank':
                                # making sure the oracle's exact match
                                # is recorded as the best match
                                if use_length:
                                    s = (-1, -1, -9.9)
                                else:
                                    s = -9.9
                                knn.append((
                                    s,
                                    candidate_training_set,
                                    dataset_type,
                                    tr_index
                                ))
                                overall_tr_index += 1
                                continue
                            if build_knn == 'oracle-exact-match-only':
                                overall_tr_index += 1
                                continue
                            if candidate_otr_indices \
                            and overall_tr_index not in candidate_otr_indices:
                                overall_tr_index += 1
                                continue
                            if opt_drop_data and random.random() < opt_drop_data:
                                overall_tr_index += 1
                                continue
                            if restrict_reranking and test_index not in restrict_reranking:
                                if reference_unit == 'treebank':
                                    raise NotImplementedError
                                # use distance from first stage k-NN as the final ranking
                                # criterion for this test item
                                s = candidate_otr_indices[overall_tr_index]
                                if use_length:
                                    s = (0, 0, s)
                                knn.append((
                                    s,
                                    candidate_training_set,
                                    dataset_type,
                                    tr_index
                                ))
                                overall_tr_index += 1
                                continue
                            tr_sent_vec = tr_data_emb[tr_index]
                            if sent_rep_metric == 'cosine' or use_length:
                                if scipy.linalg.norm(tr_sent_vec) < 0.0000001:
                                    tr_sent_vec = None
                                    tr_sent_vec_key = (tr_index, candidate_training_set, dataset_type)
                                    zero_tr_sent_vecs[tr_sent_vec_key] = None
                                if test_sent_vec is not None and tr_sent_vec is not None:
                                    s = scipy.spatial.distance.cosine(test_sent_vec, tr_sent_vec)
                                else:
                                    # We treat zero vectors as orthogonal as they probably would
                                    # be orthogonal if we had used a larger vocabulary.
                                    # Note that the range of distance.cosine() is 0 to 2.
                                    s = 1.0
                            elif sent_rep_metric in ('L2', 'SphereL2'):
                                s = scipy.spatial.distance.euclidean(test_sent_vec, tr_sent_vec)
                            elif sent_rep_metric == 'Random':
                                s = get_hash('%s:%s:%s:%d:%d' %(
                                    sent_rep_name,
                                    candidate_training_set, dataset_type,
                                    tr_index, test_index
                                ))
                            else:
                                raise ValueError, 'unsupported sent_rep_metric %r' %sent_rep_metric
                            if use_length:
                                if reference_unit == 'treebank':
                                    raise ValueError, 'Cannot use use_length with treebank reference'
                                d_length = abs(test_sent_length - tr_data_length[tr_index])
                                d_punct  = abs(test_sent_punct  - tr_data_punct[tr_index])
                                length_score = int(d_length/5)
                                punct_score  = int(d_punct)
                                s = (length_score, punct_score, s)
                            knn.append((
                                s,
                                candidate_training_set,
                                dataset_type,
                                tr_index
                            ))
                            overall_tr_index += 1
                    knn.sort()
                    if len(knn) > opt_max_k \
                    and knn[opt_max_k-1][0] == knn[opt_max_k][0]:
                        # There is a tie for the k-th place, i.e. we must break
                        # the tie in order to determine which items will be in
                        # the k-NN set.
                        # This should be rare for sentences unless there are
                        # duplicate sentences.
                        new_knn = []
                        len_knn = len(knn)
                        i = 0
                        while i < len_knn:
                            if i < opt_max_k or knn[i][0] == knn[opt_max_k-1][0]:
                                s, candidate_training_set, dataset_type, \
                                    tr_index = knn[i]
                                new_knn.append((
                                    s,
                                    get_hash('%s:%s:%d:%d' %(
                                        candidate_training_set, dataset_type,
                                        tr_index, test_index
                                    )),
                                    candidate_training_set, dataset_type,
                                    tr_index
                                ))
                            else:
                                break
                            i += 1
                        incr_counter(knn_tie_stats, 're-visiting', len(new_knn))
                        knn = new_knn
                        knn.sort()
                    sentence_knn.append(map(lambda x: x[-3:], knn[:opt_max_k]))
                sys.stderr.write('%d sentence(s) done (%.1f%%)   \n' %(num_test_sentences, 100.0))
                if zero_tr_sent_vecs:
                    num_zero_tr_sent_vec = len(zero_tr_sent_vecs.keys())
                    sys.stderr.write('Info: %d training vectors are close to 0.\n' %num_zero_tr_sent_vec)
                if num_zero_te_sent_vec:
                    sys.stderr.write('Info: %d test vectors are close to 0.\n' %num_zero_te_sent_vec)
                if knn_tie_stats:
                    print 'knn_tie_stats:'
                    print_counters(knn_tie_stats)
                    sys.stdout.flush()
                if rnd_stats:
                    print 'rnd_stats:'
                    print_counters(rnd_stats)
                    sys.stdout.flush()

            if not opt_count_scenarios:
                for run in range(num_runs):
                    #for num_seeds in (1,2,3,5,6,10,12,15,18,20,24,30,36,40,50,60):
                    for num_seeds in (1,3,9):
                        if num_seeds > opt_max_seeds:
                            continue
                        if not fast_track_scenario:
                            sys.stderr.write('\n=== Run %d with %d seed(s) ===\n\n' %(run, num_seeds))
                        # The following list could be used in the future to
                        # prefer certain seeds. (Previous code tried to count
                        # the number of usable k-NN data points but only did
                        # this for one treebank and not for training data.)
                        ranking = []
                        for i in range(parser_num_seeds):
                            ranking.append((0, '%d' %(parser_start_seed+i)))
                        num_suitable_seeds = len(ranking)
                        selected_seeds = []
                        if num_seeds == 1:
                            if num_suitable_seeds < num_runs \
                            and not fast_track_scenario:
                                sys.stderr.write('Warning: Fewer suitable seeds than runs (%d < %d); some runs will share the same seed\n' %(
                                    num_suitable_seeds, num_runs
                                ))
                            selected_seeds.append(ranking[run % num_suitable_seeds][-1])
                        elif num_suitable_seeds >= num_seeds * num_runs:
                            # can use non-overlapping sets of seeds
                            for minor_index in range(num_seeds):
                                seed_index = num_seeds * run + minor_index
                                selected_seeds.append(
                                    ranking[seed_index % num_suitable_seeds][-1]
                                )
                        elif num_seeds == 3:
                            # each block of 9 seeds allows us to do 12 runs so that
                            # 2 runs overlap at most in 1 seed
                            if num_suitable_seeds < 9 * (1+int((num_runs-1)/12)) \
                            and not fast_track_scenario:
                                sys.stderr.write('Warning: Not enough suitable seeds to pick 3 seeds with just 1 overlap between runs\n')
                            major_run = int(run/12)
                            minor_run = run % 12
                            for minor_index in pick3from9[minor_run]:
                                seed_index = 9*major_run + minor_index
                                selected_seeds.append(
                                    ranking[seed_index % num_suitable_seeds][-1]
                                )
                        elif num_seeds == num_suitable_seeds:
                            # must use all seeds
                            if not fast_track_scenario:
                                sys.stderr.write('Warning: No or reduced variation in runs as number of required seeds matches number of suitable seeds\n')
                            selected_seeds = map(lambda x: x[-1], ranking)
                        else:
                            if not fast_track_scenario:
                                sys.stderr.write('Warning: Picking seeds based on hash of seed and run\n')
                            hashed_ranking = []
                            for negcount, seed in ranking:
                                hashed_ranking.append((
                                    negcount,
                                    get_hash('%s:%d' %(seed, run)),
                                    seed
                                ))
                            hashed_ranking.sort()
                            selected_seeds = map(lambda x: x[-1], hashed_ranking[:num_seeds])

                        if len(selected_seeds) < num_seeds:
                            if not fast_track_scenario:
                                sys.stderr.write('Only found %d suitable seeds for %r on %s. Not enough seeds to complete %s with %d seeds (run %d)\n' %(
                                    len(selected_seeds), model_tbids, t, setting, num_seeds, run
                                ))
                            continue
                        # for each seed, compile list of weights that can be used in this run
                        # (we previously had code here to only use weights for which we have
                        # all seeds but that left us with hardly any data points)
                        key = (model_tbids, model_tbids[0], 'dev')  # check available data using t=s1
                        seed2results = available_predictions[key]
                        seed2usable_weights = {}
                        for seed in selected_seeds:
                            usable_weights = []
                            for wvec in seed2results[seed]:
                                if can_use_weights(wvec, allowed_weights):
                                    usable_weights.append(wvec)
                            seed2usable_weights[seed] = usable_weights
                            if not fast_track_scenario:
                                sys.stderr.write('Seed %s has %d usable data points for %r on %s\n' %(
                                    seed, len(usable_weights), model_tbids, t
                                ))
                        # tbemb settings to try
                        tbembs = []
                        if learning == 'no-learning':
                            for corner_name, _ in corner_names_and_vec:
                                tbembs.append(('pick-' + corner_name, 1))
                            for tbemb in [
                                ('equal', 1),
                                ('all3', 3)
                            ]:
                                tbembs.append(tbemb)
                        for i in (1, 2, 3, 5, 9, 16, 27, 47, 81):
                            if i > opt_max_samples:
                                continue
                            if learning == 'no-learning':
                                tbembs.append(('random', i))
                                continue
                            if learning == 'oracle':
                                tbembs.append(('knnavg-of-rnd-1', 1))
                                break
                            if reference_unit == 'treebank' and i > 1:
                                if i == 3:
                                    tbembs.append(('knnminidoc-3', 1))
                                continue
                            # tbembs.append(('knnavg-of-avg-%d' %i, 1)) # would require major code changes
                            tbembs.append(('knnavg-of-ctr-%d' %i, 1))
                            if allowed_weights != 'corner':
                                tbembs.append(('knnavg-of-rnd-%d' %i, 1))
                            if i > 1:
                                # no need to do this for i==1 as it should be
                                # the same as knnavg
                                tbembs.append(('knncomb', i))
                                tbembs.append(('knnminidoc-%d' %i, 1))
                            if i == 1:
                                tbembs.append(('knnreduce-plain-avg', 1))
                                tbembs.append(('knnreduce-plain-rnd', 1))
                                if allowed_weights != 'corner':
                                    tbembs.append(('knnreduce-plain-ctr', 1))
                                if reference_unit == 'sentence':
                                    # "intersection with overall LAS" knnreduce methods
                                    tbembs.append(('knnreduce-iwol-avg', 1))
                                    tbembs.append(('knnreduce-iwol-rnd', 1))
                                    if allowed_weights != 'corner':
                                        tbembs.append(('knnreduce-iwol-ctr', 1))
                            #if i > num_corners:
                            #    tbembs.append(('augment', i))
                        for tbemb, num_samples in tbembs:
                            sys.stdout.flush()
                            num_parses = num_samples * num_seeds
                            if num_parses > opt_max_parses:
                                if not fast_track_scenario:
                                    print 'Ignoring experiment with %d parses' %num_parses
                                continue
                            # allocate experiment ID
                            experiment_id = next_exp_id
                            next_exp_id += 1
                            if not fast_track_scenario:
                                print
                                print 'Allocated experiment_id', experiment_id
                            sys.stdout.flush()
                            # skip experiments in fast-tracked scenarios
                            if fast_track_scenario:
                                continue
                            if opt_explore_sent_rep:
                                # to explore effect of vector size, let's use a method
                                # that should depend a lot on sentence similarity
                                if tbemb != 'knnreduce-plain-avg':
                                    continue
                            # skip completed experiments
                            if experiment_id in completed_experiments:
                                print 'Skipping completed experiment'
                                continue
                            # skip any number of seeds or samples not relevant for the naacl paper
                            if num_seeds > 1:
                                print 'Skipping experiment with %d seeds' %num_seeds
                                continue
                            if num_samples not in (1,3):
                                print 'Skipping experiment with %d samples' %num_samples
                                continue
                            if num_parses not in (1,3,9):
                                print 'Skipping experiment with %d parses' %num_parses
                                continue
                            # the experiment path needs to be easy to parse into the
                            # columns of the result table
                            experiment_path = [
                                '%s-%s-weights-%s' %(short_scenario, allowed_weights, learning),
                                '%s-on-%s' %('-'.join(model_tbids), t),
                                'seeds-%d-sampling-%d-with-%s' %(num_seeds, num_samples, tbemb),
                            ]
                            if tbemb[:3] == 'knn':
                                if use_length:
                                    d_extra = 'lpc-after-'
                                else:
                                    d_extra = ''
                                if sent_rep_metric:
                                    metric_name = sent_rep_metric
                                else:
                                    metric_name = 'none'
                                experiment_path.append('distance-%s%s-%s' %(d_extra, metric_name, sent_rep_name))
                                experiment_path.append('retrieving-%s-for-%s' %(reference_unit, query_unit))
                                if sent_rep_offset is not None:
                                    experiment_path.append('with-offset-%04d' %sent_rep_offset)
                            experiment_path.append('run-%d-exp-%d' %(run, experiment_id))
                            experiment_path = '/'.join(experiment_path)
                            print 'Experiment %d: %r' %(experiment_id, experiment_path)
                            knn_stats = {}
                            sentence_parses = []
                            for sentence_index in range(num_test_sentences):
                                parses = []
                                for seed in selected_seeds:
                                    weights = []
                                    all3 = tbemb in ('all3', 'augment')
                                    num_random_samples = 0
                                    for corner_name, corner_vec in corner_names_and_vec:
                                        if tbemb == 'pick-' + corner_name or all3:
                                            weights.append(corner_vec)
                                    if tbemb == 'equal':
                                        weights.append(centre)
                                    if tbemb == 'random':
                                        num_random_samples = num_samples
                                    if tbemb == 'augment':
                                        num_random_samples = num_samples - num_corners
                                    #if num_random_samples:
                                    #    sys.stderr.write('Sampling %d items from:\n' %num_random_samples)
                                    #    for wvec in seed2usable_weights[seed]:
                                    #        sys.stderr.write('\t%r\n' %(wvec,))
                                    while num_random_samples:
                                        candidates = []
                                        cutoff = opt_random_sample_distance_cutoff
                                        for wvec in seed2usable_weights[seed]:
                                            if wvec in weights:
                                                notnew = 1
                                            else:
                                                notnew = 0
                                            if opt_maximise_random_sample_distance:
                                                _, d = get_nn_and_distance(weights, wvec)
                                                if opt_random_sample_distance_cutoff  \
                                                and d > cutoff:
                                                    d = cutoff
                                            else:
                                                d = 0
                                            candidates.append((notnew, -d, random.random(), wvec))
                                        candidates.sort()
                                        _, _, _, wvec = candidates[0]
                                        weights.append(wvec)
                                        cutoff *= opt_random_sample_distance_cutoff_decay
                                        num_random_samples -= 1
                                    # knn variants
                                    if tbemb.startswith('knnreduce'):
                                        knn = sentence_knn[sentence_index]   # all opt_max_k items
                                        intersection = {}
                                        k_index = 0
                                        sentences_used = 0
                                        for (knn_tbid, knn_dataset_type, knn_tr_index) in knn:
                                            intersection = knnreduce_intersect(
                                                model_tbids, knn_tbid,
                                                knn_dataset_type, seed,
                                                available_predictions,
                                                allowed_weights, knn_tr_index,
                                                k_index, sentence_index,
                                                intersection
                                            )
                                            sentences_used += 1
                                            if len(intersection) == 1:
                                                break
                                            k_index += 1
                                        if reference_unit == 'sentence' and '-iwol-' in tbemb:
                                            incr_counter(knn_stats, 'tie size (before treebank LAS)', len(intersection))
                                            if len(intersection) > 1:
                                                # one last intersection with treebank-level LAS
                                                knn_tbid, knn_dataset_type, _ = knn[0]
                                                intersection = knnreduce_intersect(
                                                    model_tbids, knn_tbid,
                                                    knn_dataset_type, seed,
                                                    available_predictions,
                                                    allowed_weights, -1, -1,
                                                    sentence_index,
                                                    intersection
                                                )
                                                incr_counter(knn_stats, 'treebank LAS used', 1)
                                            else:
                                                incr_counter(knn_stats, 'treebank LAS not used', 0)
                                            incr_counter(knn_stats, 'tie size (after treebank LAS)', len(intersection))
                                        elif reference_unit == 'sentence':
                                            incr_counter(knn_stats, 'tie size (querying sentences, no iwol)', len(intersection))
                                        else:
                                            incr_counter(knn_stats, 'tie size (querying treebanks)', len(intersection))
                                        if tbemb.endswith('rnd') or tbemb.endswith('ctr'):
                                            # pick one vector
                                            best = None
                                            for wvec in intersection:
                                                wvec_str = ':'.join(map(lambda x: '%.6f' %x, wvec))
                                                tie_breaker = get_hash(
                                                    '%s:%d:%d:%s' %(
                                                        seed, run, k, wvec_str
                                                    )
                                                )
                                                if tbemb.endswith('ctr'):
                                                    r = distance(wvec, centre)
                                                    # numerical effects can cause slightly different
                                                    # distances for points that should be at the
                                                    # same distance --> add randomness of tie_breaker
                                                    r += 0.000001 * hash_to_float(tie_breaker)
                                                else:
                                                    r = 0
                                                candidate = (r, tie_breaker, wvec)
                                                if best is None or candidate < best:
                                                    best = candidate
                                            weights.append(best[-1])
                                            intersection = []
                                        # old -avg behaviour when intersection not reset above
                                        for wvec in intersection:
                                            weights.append(wvec)
                                            # note: all these wvecs will be averaged further below
                                        # update knn_stats
                                        incr_counter(
                                            knn_stats, reference_unit + 's',
                                            sentences_used
                                        )
                                    elif tbemb.startswith('knnminidoc'):
                                        k = int(tbemb.split('-')[-1])
                                        if k > opt_max_k:
                                            continue
                                        knn = sentence_knn[sentence_index][:k]
                                        # calculate average LAS in the k-NN set for each
                                        # candidate weight vector
                                        wvec2kscores = {}
                                        k = 0
                                        for (knn_tbid, knn_dataset_type, knn_tr_index) in knn:
                                            knn_key = (model_tbids, knn_tbid, knn_dataset_type)
                                            try:
                                                wvec2scores = available_predictions[knn_key][seed]
                                            except:
                                                sys.stderr.write('Warning: no data points for %r and seed %s\n' %(knn_key, seed))
                                                wvec2scores = {}
                                            if not wvec2scores:
                                                continue
                                            k += 1
                                            for wvec in wvec2scores:
                                                if can_use_weights(wvec, allowed_weights):
                                                    score = wvec2scores[wvec][knn_tr_index]
                                                    if wvec not in wvec2kscores:
                                                        wvec2kscores[wvec] = []
                                                    wvec2kscores[wvec].append(score)
                                        # calculate average scores and find highest score
                                        best = None
                                        in_tie = 0
                                        for wvec in wvec2kscores:
                                            scores = wvec2kscores[wvec]
                                            if len(scores) < k:
                                                continue
                                            score = sum(scores) / float(len(scores))
                                            if best is None or score > best[0]:
                                                in_tie = 0
                                            wvec_str = ':'.join(map(lambda x: '%.6f' %x, wvec))
                                            tie_breaker = get_hash(
                                                '%s:%d:%d:%s:%s:%s:%d' %(
                                                    seed, run, k, wvec_str,
                                                    knn_tbid,
                                                    knn_dataset_type,
                                                    knn_tr_index
                                                )
                                            )
                                            # prefer wvecs close to the centre
                                            inv_d = -distance(wvec, centre)
                                            # numerical effects can cause slightly different
                                            # distances for points that should be at the
                                            # same distance --> add randomness of tie_breaker
                                            inv_d += 0.000001 * hash_to_float(tie_breaker)
                                            candidate = (score, inv_d, tie_breaker, wvec)
                                            if best is None or candidate > best:
                                                best = candidate
                                            if score == best[0]:
                                                in_tie += 1
                                        if best is None:
                                            sys.stderr.write('No usable data points for k_index %d of sentence_index %d with seed %s\n' %(
                                                k_index, sentence_index, seed
                                            ))
                                            best_wvec = centre
                                        else:
                                            best_wvec = best[-1]
                                        weights.append(best_wvec)
                                        # update knn_stats
                                        incr_counter(knn_stats, 'tie size', in_tie)
                                    elif tbemb[:3] == 'knn':
                                        if tbemb[:9] == 'knnavg-of':
                                            k = int(tbemb.split('-')[-1])
                                        else:
                                            k = num_samples
                                        if k > opt_max_k:
                                            continue
                                        knn = sentence_knn[sentence_index][:k]
                                        k_index = 0
                                        for (knn_tbid, knn_dataset_type, knn_tr_index) in knn:
                                            # find weights that worked best for the k-NN
                                            # item with the same training data and seed
                                            knn_key = (model_tbids, knn_tbid, knn_dataset_type)
                                            try:
                                                wvec2scores = available_predictions[knn_key][seed]
                                            except:
                                                sys.stderr.write('Warning: no data points for %r and seed %s\n' %(knn_key, seed))
                                                wvec2scores = {}
                                            best = None
                                            in_tie = 0
                                            for wvec in wvec2scores:
                                                if can_use_weights(wvec, allowed_weights):
                                                    score = wvec2scores[wvec][knn_tr_index]
                                                    if best is None or score > best[0]:
                                                        in_tie = 0
                                                    wvec_str = ':'.join(map(lambda x: '%.6f' %x, wvec))
                                                    tie_breaker = get_hash(
                                                        '%s:%d:%d:%s:%s:%s:%d' %(
                                                            seed, run, k, wvec_str,
                                                            knn_tbid,
                                                            knn_dataset_type,
                                                            knn_tr_index
                                                        )
                                                    )
                                                    if '-ctr-' in tbemb:
                                                        # prefer wvecs close to the centre
                                                        inv_d = -distance(wvec, centre)
                                                        # numerical effects can cause slightly different
                                                        # distances for points that should be at the
                                                        # same distance --> add randomness of tie_breaker
                                                        inv_d += 0.000001 * hash_to_float(tie_breaker)
                                                    else:
                                                        inv_d = 0
                                                    candidate = (score, inv_d, tie_breaker, wvec)
                                                    if best is None or candidate > best:
                                                        best = candidate
                                                    if score == best[0]:
                                                        in_tie += 1
                                            if best is None:
                                                sys.stderr.write('No usable data points for %s-%s sentence %d with seed %s. Falling back to uniform weights for k_index %d of sentence_index %d\n' %(
                                                    knn_tbid, knn_dataset_type, knn_tr_index, seed, k_index, sentence_index
                                                ))
                                                best_wvec = centre
                                            else:
                                                best_wvec = best[-1]
                                            weights.append(best_wvec)
                                            k_index += 1
                                            # update knn_stats
                                            incr_counter(knn_stats, 'tie size', in_tie)
                                    if tbemb[:9] in ('knnavg-of', 'knnreduce',) and len(weights) > 1:
                                        avg_wvec = []
                                        for vec_index in range(num_corners):
                                            vec_comp_total = 0.0
                                            for wvec in weights:
                                                vec_comp_total += wvec[vec_index]
                                            avg_wvec.append(vec_comp_total / len(weights))
                                        parses.append((seed, avg_wvec))
                                    elif not weights:
                                        raise ValueError, 'no weights selected'
                                    else:
                                        for wvec in weights:
                                            parses.append((seed, wvec))
                                # end of "for seed ..."
                                sentence_parses.append(parses)
                            # end of "for sentence_index ..."

                            if knn_stats:
                                print 'knn_stats:'
                                print_counters(knn_stats)

                            # create tbweights files and parsing scripts
                            # and record in combiner sheet what needs to be
                            # combined and evaluated
                            if not sentence_parses:
                                print 'No sentences to parse for experiment %d %r' %(experiment_id, experiment_path)
                                continue
                            columns = []
                            columns.append('%d' %experiment_id)
                            columns.append(experiment_path)
                            columns.append('%d' %num_parses)
                            first_sentence_parses = sentence_parses[0]
                            if len(first_sentence_parses) != num_parses:
                                raise ValueError, 'Found %d first_sentence_parses, expected %d' %(
                                    len(first_sentence_parses), num_parses
                                )
                            for parse_id in range(num_parses):
                                seed, _ = first_sentence_parses[parse_id]
                                parse_name = '%d-%s-%d' %(num_parses, seed, parse_id)
                                tbw_exp_dir = '%s/%s' %(tbweights_dir, experiment_path)
                                tbwname = '%s/%s.tbweights' %(
                                    tbw_exp_dir, parse_name
                                )
                                if not os.path.exists(tbw_exp_dir):
                                    os.makedirs(tbw_exp_dir)
                                tbwfile = open(tbwname, 'wb')
                                for index in range(num_test_sentences):
                                    s, wvec = sentence_parses[index][parse_id]
                                    if s != seed:
                                        raise ValueError, 'Implementation tries to use different seeds in a single parser call'
                                    tbids_with_weights = []
                                    for w_index, model_tbid in enumerate(model_tbids):
                                        tbids_with_weights.append('%s:%.6f' %(
                                            model_tbid, wvec[w_index]
                                        ))
                                    tbwfile.write(' '.join(tbids_with_weights))
                                    tbwfile.write('\n')
                                tbwfile.close()
                                fake_tbid = model_tbids[0]
                                expid1000 = int(experiment_id/1000)
                                script_dir  = 'te-parse/t%04d/%d' %(expid1000, experiment_id)
                                if not os.path.exists(script_dir):
                                    os.makedirs(script_dir)
                                script_name = '%s/%d.sh' %(script_dir, parse_id)
                                worker_file = worker2file[next_worker]
                                next_worker = (next_worker+1) % num_workers
                                worker_file.write('%s\n' %script_name)
                                script_file = open(script_name, 'wb')
                                script_file.write(script_template %locals())
                                script_file.close()
                                for epoch in epochs.split():
                                    columns.append('%s_e%s.conllu' %(parse_name, epoch))
                            # end of "for parse_id ..."
                            combine.write('\t'.join(columns))
                            combine.write('\n')
                            print 'Wrote tbweights and %d parse script(s) for experiment_id %d' %(
                                num_parses, experiment_id
                            )
                            sys.stdout.flush()
                        # end of "for for tbemb, num_samples ..."
                    # end of "for num_seed ..."
                    combine.flush()
                # end of "for run ..."
                end_scenario = time.time()
                if not fast_track_scenario:
                    print 'Duration of gen_tbemb_sampling for scenario %d: %.1fs' %(scenario_count, end_scenario-start_scenario)


if opt_count_scenarios:
    print "Scenarios:", scenario_count
else:
    combine.close()

for worker in range(num_workers):
    worker2file[worker].seek(0)
    if opt_parallel_n == 1:
        filename = 'te-worker/worker-%d.sh' %(1000+worker)
    else:
        filename = 'te-worker/worker-%d-parallel-%d-of-%d.sh' %(
            1000+worker, opt_parallel_i+1, opt_parallel_n
        )
    finalfile = open(filename, 'wb')
    finalfile.write(worker2file[worker].read())
    finalfile.close()


print 'Finished'

