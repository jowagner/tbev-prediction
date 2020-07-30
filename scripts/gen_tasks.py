#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

import math
import random
import sys
import StringIO

import project_3_weights

def print_usage():
    print 'Usage:'
    print
    print '%s [options]' %sys.argv[0]
    print
    print '\t--num-workers NUMBER    Split tasks over these many worker files (default: 24)'
    print '\t--worker-dir  FOLDER    Where to write the worker files (default: te-worker)'
    print '\t--script      FILE      Task script (default: ./uuparser-tbemb-test.sh)'
    print '\t--tab-tasks             Use tabs to separate args in task lines (default: write shell commands)'
    print '\t--parser      NAME      Substitute uuparser-tbemb in the script template filename with this name'
    print '\t--skip-indomain-parsing Do no write tasks parsing training data (default: parse all data)'
    print '\t--filter-results        Read results table from stdin and do not generate same points again'
    print '\t--filter-use-training   Also read training results when building the filter (default: dev only)'
    print '\t--box-size    NUMBER    Width and height of box all points must be in (default: 4.8)'
    print '\t--width       NUMBER    Width of box all points must be in (default: 4.8)'
    print '\t--height      NUMBER    Height of box all points must be in (default: 4.8)'
    print '\t--no-box                Do not clip (default: clip at box if and only if number of weights is 3)'
    print '\t--no-projection         Do not project 3d weights to 2d (default: project if and only if number of weights is 3)'
    print '\t--minimum-weight NUMBER  Do not write tasks with a weight below this limit (default: -99.9)'
    print '\t--maximum-centre-distance NUMBER  Do not write tasks with weights further from the centre / uniform weights (default: 0 = no limit)'
    print '\t--maximum-corner-distance NUMBER  Do not write tasks with weights further than this from the nearest corner (default: 0 = no limit)'
    print '\t--decay-density         Reduce point density (default: uniform density)'
    print '\t--decay-strength        Parameter for density decay (default: 100)'
    print '\t--num-points  NUMBER    How many data points to generate (default: 200)'
    print '\t--num-canidates NUMBER  Select each point from these many candidate points (default: 49500)'
    print '\t--median-interpolations NUMBER  How many rounds of median interpolation to perform (default: 21)'
    print '\t--skip        NUMBER    Do not write tasks for the first NUMBER points (default: 0)'
    print '\t--collection  TBIDS     List of TBIDs (default: fr_gsd:fr_partut:fr_sequoia:fr_spoken)'
    print '\t--no-subsets            Use model over all treebanks of the collection (not all sizes supported; default: use all subsets of size 3)'
    print '\t--seed        NUMBER    Random number generator initialisation (default: 0 = system provided)'
    print '\t--start-seed  NUMBER    Start of the parser seed range (default: 300)'
    print '\t--num-seeds   NUMBER    Number of parser seeds (default: 9)'
    print '\t--debug                 Log more detail to stderr'
    print '\t--help                  Show this page'

num_workers = 24
opt_worker_dir = 'te-worker'
script      = './uuparser-tbemb-test.sh'
opt_tab_tasks = False
opt_filter_results = False
opt_only_dev = True
opt_skip_indomain_parsing = False
x_min, x_max = -2.4, 2.4
y_min, y_max = -2.4, 2.4
opt_clip_at_box = True
opt_project_points = True
opt_minimum_weight = -99.9
opt_maximum_corner_distance = 0.0
opt_maximum_centre_distance = 0.0
opt_with_density_decay = False
opt_decay_strength = 100.0
opt_collection = 'fr_gsd:fr_partut:fr_sequoia:fr_spoken'
opt_subsets = True
opt_seed     = 0
opt_start_seed = 300
opt_num_seeds  = 9
opt_num_points = 200
opt_num_candidates = 49500
opt_skip       = 0
opt_discard_above_r = False
opt_help     = False
opt_debug    = False
num_points_step_1 = 21     # number of points to fill in based on interpolation of neighbouring triangle median points with similar rank

while len(sys.argv) >= 2 and sys.argv[1].startswith('-') and len(sys.argv[1]) > 1:
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option == '--debug':
        opt_debug = True
    elif option == '--median-interpolations':
        num_points_step_1 = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--num-candidates':
        opt_num_candidates = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--num-points', '--data-points'):
        opt_num_points = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--skip', '--skip-points'):
        opt_skip = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--num-seeds', '--parser-seeds'):
        opt_num_seeds = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--start-seed', '--first-parser-seed'):
        opt_start_seed = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--seed', '--random-seed'):
        opt_seed = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--collection', '--tbids'):
        opt_collection = sys.argv[1]
        del sys.argv[1]
    elif option == '--num-workers':
        num_workers = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--worker-dir':
        opt_worker_dir = sys.argv[1]
        del sys.argv[1]
    elif option in ('--script-template', '--script'):
        script = sys.argv[1]
        del sys.argv[1]
    elif option == '--no-projection':
        opt_project_points = False
    elif option == '--no-box':
        opt_with_density_decay = False
    elif option == '--no-subsets':
        opt_subsets = False
    elif option == '--tab-tasks':
        opt_tab_tasks = True
    elif option == '--parser':
        name = sys.argv[1]
        script = script.replace('uuparser-tbemb', name)
        del sys.argv[1]
    elif option == '--filter-results':
        opt_filter_results = True
    elif option == '--filter-use-training':
        opt_only_dev = False
    elif option == '--skip-indomain-parsing':
        opt_skip_indomain_parsing = True
    elif option == '--minimum-weight':
        opt_minimum_weight = float(sys.argv[1])
        del sys.argv[1]
    elif option == '--maximum-corner-distance':
        opt_maximum_corner_distance = float(sys.argv[1])
        del sys.argv[1]
    elif option == '--maximum-centre-distance':
        opt_maximum_centre_distance = float(sys.argv[1])
        del sys.argv[1]
    elif option == '--width':
        width = float(sys.argv[1])
        x_min, x_max = -width/2, width/2
        del sys.argv[1]
    elif option == '--height':
        height = float(sys.argv[1])
        y_min, y_max = -height/2, height/2
        del sys.argv[1]
    elif option == '--box-size':
        box_size = float(sys.argv[1])
        x_min, x_max = -box_size/2, box_size/2
        y_min, y_max = -box_size/2, box_size/2
        del sys.argv[1]
    elif option in ('--decay-density', '--with-density-decay'):
        opt_with_density_decay = True
    elif option == '--decay-strength':
        opt_decay_strength = sys.argv[1]
        # remove leading zeros:
        while len(opt_decay_strength) > 1 and opt_decay_strength[0] == '0':
            opt_decay_strength = opt_decay_strength[1:]
        opt_decay_strength = float(opt_decay_strength)
        del sys.argv[1]
    else:
        print 'Unsupported option %s' %option
        opt_help = True
        break

if len(sys.argv) != 1:
    opt_help = True

if opt_help:
    print_usage()
    sys.exit(0)

if opt_debug:
    sys.stderr.write('Debugging output requested\n')

if opt_seed:
    if opt_debug:
        sys.stderr.write('Seeding random number generator with seed %d\n' %opt_seed)
    random.seed(opt_seed)

if opt_decay_strength <= 0.0:
    print 'Switching density decay off as strength is 0 or negative.'
    opt_with_density_decay = False
else:
    decay = 100.0 / opt_decay_strength

scenarios = []
collection = opt_collection.split(':')
k = len(collection)
if opt_subsets:
    if k < 3:
        raise ValueError, 'Need at least 3 TBIDs to form subsets of 3'
    if opt_skip_indomain_parsing and k < 4:
        raise ValueError, 'Need at least 4 TBIDs to skip indomain parsing with subsets of 3'
    for i1 in range(k-2):
      for i2 in range(i1+1, k-1):
        for i3 in range(i2+1, k):
            s1 = collection[i1]
            s2 = collection[i2]
            s3 = collection[i3]
            for t in collection:
                if not opt_skip_indomain_parsing or t not in (s1, s2, s3):
                    scenarios.append(((s1, s2, s3), t))
    model_set_size = 3
elif opt_skip_indomain_parsing:
    raise ValueError, 'Cannot combine --no-subsets with --skip-indomain-parsing'
else:
    if k < 2:
        raise ValueError, 'Need at least 2 TBIDs to form subsets of 3'
    for t in collection:
        scenarios.append((collection, t))
    model_set_size = k

if model_set_size != 3:
    opt_project_points = False
    opt_clip_at_box    = False

if opt_debug:
    sys.stderr.write('Scenarios:\n')
    for i, scenario in enumerate(scenarios):
        sys.stderr.write('[%d]: %r\n' %(i, scenario))

worker2file = {}
for worker in range(num_workers):
    worker2file[worker] = StringIO.StringIO()
next_worker = 0

points_inside_selected_space = []
points_anywhere   = []

skip_triangle_median = False
if len(sys.argv) > 2:
    if model_set_size != 3:
        raise ValueError, 'Cannot use 2d projected points with %d-d weight space' %model_set_size
    tsv = open(sys.argv[2], 'rb')
    count = 0
    while True:
        line = tsv.readline()
        if not line:
            break
        fields = line.split()
        x = float(fields[0])
        y = float(fields[1])
        points_anywhere.append((x, y))
        if (x_min <= x <= x_max) and (y_min <= y <= y_max):
            points_inside_selected_space.append((x, y))
        count += 1
    tsv.close()
    if opt_debug:
        sys.stderr.write('Added %d points from file %s\n' %(count, sys.argv[2]))
    skip_triangle_median = True

ignore = {}

def write_tasks(weights):
    """
    writes tasks for all scenarios and seeds, skipping previously
    written tasks
    """
    global scenarios
    global worker2file
    global next_worker
    global points_inside_selected_space
    global ignore
    for model_tbids, t in scenarios:
        for seed in range(opt_start_seed, opt_start_seed + opt_num_seeds):
                f = worker2file[next_worker]
                next_worker = (next_worker+1) % num_workers
                tbid_and_weights = []
                for i, weight in enumerate(weights):
                    tbid = model_tbids[i]
                    tbid_and_weights.append('%s:%.6f' %(tbid, weight))
                tbid_and_weights = ' '.join(tbid_and_weights)
                tbid_and_weights = tbid_and_weights.replace('-0.000000', '0.000000')
                if not opt_tab_tasks:
                    tbid_and_weights = '"' + tbid_and_weights + '"'
                key = []
                key.append('')
                key.append('%d' %seed)
                key.append(tbid_and_weights)
                key.append(t)
                if opt_tab_tasks:
                    key = '\t'.join(key)
                else:
                    key = ' '.join(key)
                key = key + '\n'
                if not key in ignore:
                    f.write(script + key)

if opt_filter_results:
   count = 0
   while True:
       result_filename = sys.stdin.readline()
       if not result_filename:
           break
       result_filename = result_filename.rstrip()
       #           -6       -5       -4       -3 -2       -1
       # ./results/en_sewta:en_sewtr:en_sewtw-on-en_sewte-dev.tsv
       if opt_only_dev and not result_filename.endswith('-dev.tsv'):
           continue
       fields = result_filename.replace('/', '-').replace(':', '-').split('-')
       t = fields[-2]
       model_tbids = []
       for tb_index in range(model_set_size):
           tbid = fields[tb_index - 3 - model_set_size]
           if not '_' in tbid:
               raise ValueError, 'No underscore in tbid %r in %r' %(tbid, result_filename)
           model_tbids.append(tbid)
       if debug:
           sys.stderr.write('%r parsed into %r on %s' %(result_filename, model_tbids, t))
       f = open(result_filename, 'rb')
       header = f.readline().split()
       seed_column = header.index('Seed')
       w_columns = []
       for tb_index, tbid in enumerate(model_tbids):
           column = header.index('Weight-%d-for-%s' %(tb_index, tbid))
           w_columns.append(column)
       while True:
           line = f.readline()
           if not line:
               break
           fields = line.split()
           seed = fields[seed_column]
           tbid_and_weights = []
           for tb_index, tbid in enumerate(model_tbids):
               weight = fields[w_columns[tb_index]]
               tbid_and_weights.append('%s:%s' %(tbid, weight))
           tbid_and_weights = ' '.join(tbid_and_weights)
           if not opt_tab_tasks:
               tbid_and_weights = '"' + tbid_and_weights + '"'
           key = []
           key.append('')
           key.append('%d' %seed)
           key.append(tbid_and_weights)
           key.append(t)
           if opt_tab_tasks:
               key = '\t'.join(key)
           else:
               key = ' '.join(key)
           key = key + '\n'
           ignore[key] = None
           count += 1
       f.close()
   print 'Added %d lines to filter' %count

def add_point_if_not_too_close_and_write_tasks_if_in_selected_space(weights):
    if opt_project_points:
        x, y, z = project_3_weights.project(weights)
        if abs(z) > 0.001:
            sys.stderr.write('Warning: Large deviation from plane: %.6f\n' %z)
        is_in_selected_space = point_is_inside_selected_space(weights, (x,y))
        d = nn_distance(weights, pp=(x,y))
    else:
        is_in_selected_space = point_is_inside_selected_space(weights)
        d = nn_distance(weights)
    # check point is not too close to existing point
    if d < 0.001:
        if opt_debug:
            sys.stderr.write('Ignoring new point that has NN with d = %.6f\n' %d)
        return
    if is_in_selected_space:
        if opt_project_points:
            points_inside_selected_space.append((x,y))
        else:
            points_inside_selected_space.append(weights)
        # only start writing once opt_skip weight points have been added
        if len(points_inside_selected_space) >= opt_skip:
            write_tasks(weights)
    if opt_project_points:
        points_anywhere.append((x,y))
    else:
        points_anywhere.append(weights)

def nn_distance(weights, reference = None, pp = None):
    global points_anywhere
    if reference is None:
        reference = points_anywhere
        project_points = opt_project_points
    else:
        project_points = False
    min_d2 = 9999.9
    if project_points:
        if pp is None:
            x,y,_ = project_3_weights.project(weights)
        else:
            x,y = pp
        for p in reference:
            d2 = (p[0]-x)**2 + (p[1]-y)**2
            if d2 < min_d2:
                min_d2 = d2
    else:
        for p in reference:
            d2 = 0.0
            for i, v in enumerate(p):
                d2 += (v-weights[i])**2
            if d2 < min_d2:
                min_d2 = d2
    return min_d2**0.5

def distance(p1, p2):
    d2 = 0.0
    for i, v in enumerate(p1):
        d2 += (v-p2[i])**2
    return d2**0.5

uniform_weight = 1.0/float(model_set_size)

centre = []
for i in range(model_set_size):
    centre.append(uniform_weight)

corners = []
for i in range(model_set_size):
    corner = []
    for j in range(model_set_size):
        if i == j:
            corner.append(1.0)
        else:
            corner.append(0.0)
    corners.append(corner)

corners_and_centre = corners + [centre]

def point_is_inside_selected_space(p, pp=None):
    global centre
    global corners
    if opt_clip_at_box:
        if pp is None:
            # check whether projected point is within bounding box
            x, y, _ = project_3_weights.project(p)
        else:
            x, y = pp
        if not ((x_min <= x <= x_max) and (y_min <= y <= y_max)):
            return False
    if min(p) < opt_minimum_weight:
        return False
    if opt_maximum_centre_distance \
    and distance(p, centre) > opt_maximum_centre_distance:
        return False
    if opt_maximum_corner_distance:
        d = nn_distance(p, reference = corners)
        if d > opt_maximum_corner_distance:
            return False
    return True

# add triangle median points

def median_points(a):
    retval = []
    for i in range(model_set_size):
        retval.append(median_point(a, i))
    return retval

def median_point(a, median_idx):
    point = []
    for j in range(model_set_size):
        if (median_idx+j+1) == model_set_size:
            point.append(1 - (model_set_size-1) * a)
        else:
            point.append(a)
    return point

if not skip_triangle_median:
    add_point_if_not_too_close_and_write_tasks_if_in_selected_space(centre)

min_a = -0.25
max_a = 1.00
inc_a = 1.00/12
a = min_a
base_beams = []
beam_names = []
if model_set_size >= 3:
    for i in range(2*model_set_size):
        base_beams.append([])
        beam_names.append('?')

def point_info(p):
    if opt_project_points:
        x,y,_ = project_3_weights.project(p)
        a,b,c = p
        return '(%9.6f, %9.6f, %9.6f) --> (%6.3f, %6.3f)' %(a,b,c,x,y)
    else:
        components = []
        for v in p:
            components.append('%9.6f' %v)
        components = ', '.join(components)
        return '(' + components + ')'

num_beams = len(base_beams)
while a < (max_a + 0.5 * inc_a) \
and not skip_triangle_median:
    for i in range(model_set_size):
        new_point = median_point(a, median_idx = i)
        if opt_debug:
            sys.stderr.write('Point %d for median %d with a = %.3f: %s\n' %(
                1+len(points_inside_selected_space),
                i+1, a, point_info(new_point)
            ))
        add_point_if_not_too_close_and_write_tasks_if_in_selected_space(new_point)
        if num_beams and -0.26 < a < (uniform_weight - 0.001):
            # point on the beam from centre to and beyond a corner
            base_beams[2*i].append(new_point)
            beam_names[2*i] = 'beyond corner %d' %(model_set_size-i)
        if num_beams and (uniform_weight + 0.001) < a < 0.95:
            # point on the beam from centre to and beyond the midpoint
            # of a surface
            base_beams[(2*i+3)%num_beams].append(new_point)
            beam_names[(2*i+3)%num_beams] = 'standing on surface opposite corner %d' %(model_set_size-i)
    a += inc_a

# add base point combinations

beam_combinations = []
if model_set_size == 3:
    for beam1 in range(num_beams):
        beam2 = ((beam1+1) % num_beams)
        beam_combinations.append((beam1, beam2))
elif model_set_size > 3:
    for i in range(model_set_size):
        # from a beam standing on a surface
        beam1 = (2*i+3)%num_beams
        # to all beams extending from a corner
        # (except the opposite corner, which
        # is on the same median as beam1)
        for j in range(model_set_size):
            if i != j:
                beam2 = 2*j
                beam_combinations.append((beam1, beam2))
        # to all other beams standing on neighbouring
        # surfaces
        # TODO: Are all surfaces neighbours for all
        #       model_set_size > 4?
        for j in range(model_set_size):
            if i != j:
                beam2 = (2*j+3)%num_beams
                beam_combinations.append((beam1, beam2))
        # from a beam extending from a corner
        beam1 = 2*i
        # to all other beams extending from a corner
        for j in range(model_set_size):
            if i != j:
                beam2 = 2*j
                beam_combinations.append((beam1, beam2))

if opt_debug:
    sys.stderr.write('Beam combinations:\n')
    for i, beam_combo in enumerate(beam_combinations):
        beam1, beam2 = beam_combo
        name1 = beam_names[beam1]
        name2 = beam_names[beam2]
        sys.stderr.write('[%d]: beam %d (%s) and beam %d (%s)\n' %(
            i, beam1, name1, beam2, name2
        ))
    if not beam_combinations:
        sys.stderr.write('None\n')

def interpolate(points, weights):
    ''' weights must sum to 1 
    '''
    retval = []
    for i in range(model_set_size):
        component = 0.0
        for j, point in enumerate(points):
            weight = weights[j]
            component += weight * point[i]
        retval.append(component)
    return retval

for i in range(num_points_step_1):
    if len(points_inside_selected_space) == opt_num_points:
        if opt_debug:
            sys.stderr.write('Reached number of points before completing all interpolation rounds.\n')
        break
    if opt_debug:
        sys.stderr.write('Median interpolation round %d:\n' %(i+1))
    for beam1, beam2 in beam_combinations:
        if len(points_inside_selected_space) == opt_num_points:
            if opt_debug:
                sys.stderr.write('Reached number of points before completing interpolation round %d.\n' %(i+1))
            break
        best = None
        for wi in range(1,60):
            w = wi / 60.0
            for base_point_1 in base_beams[beam1]:
                for base_point_2 in base_beams[beam2]:
                    # only consider combibations of beam points 
                    # with same distance from centre
                    r1 = distance(base_point_1, centre)
                    r2 = distance(base_point_2, centre)
                    if abs(r1-r2) > 0.02:
                        continue
                    # interpolate points
                    candidate_point = interpolate(
                        [base_point_1, base_point_2],
                        [w, 1.0-w]
                    )
                    # get distance to centre
                    if opt_project_points:
                        x,y,z = project_3_weights.project(candidate_point)
                        r = (x**2+y**2)**0.5
                    else:
                        r = distance(candidate_point, centre)
                    # record candidate with distance to its NN
                    d = nn_distance(candidate_point)
                    if opt_with_density_decay:
                        p = -d/(decay+r)
                    else:
                        # give slight preference to points near the centre to avoid
                        # breaking ties based on numerical noise
                        p = -d/(12.0+r)
                    candidate = (p, r, d, candidate_point)
                    if not best or best > candidate:
                        best = candidate
        p, r, d, candidate_point = best
        print 'Point %d with priority %.6f, distance %.6f and radius %.6f: %s' %(
            1+len(points_inside_selected_space), p, d, r, point_info(candidate_point)
        )
        sys.stdout.flush()
        add_point_if_not_too_close_and_write_tasks_if_in_selected_space(candidate_point)

# scale spanning points so that they are all outside the box

def all_points_outside_selected_space(points):
    for p in points:
        if point_is_inside_selected_space(p):
            # found a point inside the box (or on its border)
            return False
    return True

min_a = 0.0
while True:
    spanning_points_min_a = median_points(min_a)
    if all_points_outside_selected_space(spanning_points_min_a):
        break
    min_a -= 0.1

max_a = 0.5
while True:
    spanning_points_max_a = median_points(max_a)
    if all_points_outside_selected_space(spanning_points_max_a):
        break
    max_a += 0.1

# push out spanning points further as the hexagon otherwise
# does not always cover all parts of the box

min_a = 2*min_a
max_a = 2*max_a

spanning_points = median_points(min_a) + median_points(max_a)

if opt_debug:
    sys.stderr.write('Spanning points:\n')
    for i, point in enumerate(spanning_points):
        sys.stderr.write('[%d]: %s\n' %(i, point_info(point)))

while len(points_inside_selected_space) < opt_num_points:
    best = None
    for j in range(opt_num_candidates):
        sel_spanning_points = random.sample(spanning_points, 3)
        w = []
        for i in range(3):
            w.append(random.random())
        t = sum(w)
        candidate_point = []
        for i in range(model_set_size):
            component = 0.0
            for k in (0,1,2):
                ws = w[k] / t
                component += ws * sel_spanning_points[k][i]
            candidate_point.append(component)
        # interpolate with current best to fine-tune best?
        if best and random.random() < math.tanh(0.0001*j):
            ws = 0.1 / (j+1)**0.5
            while random.random() < 0.5:
                ws *= 0.5
            candidate_point = interpolate(
                [candidate_point, best[-1]],
                [ws, 1.0-ws]
            )
        # check whether projected point is within bounding box
        if not point_is_inside_selected_space(candidate_point):
            continue
        # get minimum distance to centre and triangle corners
        if opt_project_points:
            r = (x**2+y**2)**0.5
            for corner in corners:
                x1,y1,_ = project_3_weights.project(corner)
                r1 = ((x-x1)**2+(y-y1)**2)**0.5
            if r1 < r:
                r = r1
        else:
            r = nn_distance(candidate_point, reference = corners_and_centre)
        # discard points too far away
        if opt_discard_above_r and r > opt_discard_above_r:
            continue
        # record candidate with distance to its NN
        d = nn_distance(candidate_point)
        if opt_with_density_decay:
            p = -d/(decay+r)
        else:
            p = -d
        candidate = (p, r, d, candidate_point)
        if not best or best > candidate:
            best = candidate
    p, r, d, best_point = best
    print 'Point %d with priority %.6f, distance %.6f and radius %.6f: %s' %(
        1+len(points_inside_selected_space), p, d, r, point_info(best_point)
    )
    sys.stdout.flush()
    add_point_if_not_too_close_and_write_tasks_if_in_selected_space(best_point)

for worker in range(num_workers):
    worker2file[worker].seek(0)
    finalfile = open('%s/worker-%d.sh' %(opt_worker_dir, 1000+worker), 'wb')
    finalfile.write(worker2file[worker].read())
    finalfile.close()

print 'Finished'
