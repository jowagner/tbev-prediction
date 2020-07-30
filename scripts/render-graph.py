#!/usr/bin/env python2
# -*- coding: utf-8 -*-

# (C) 2018, 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# based on rgb-voronoi.py and tune-parameters.py
# (programmed by Joachim Wagner outside work)

import math
import os
import random
import sys
import struct
import sys
import time
import zlib

import kdtree
import project_3_weights

def print_usage():
    print 'Usage: %s [options] input.tsv output.png' %(sys.argv[0].split('/')[-1])

opt_help = False
opt_debug = False
zoom     = 100.0
width    = 480
height   = 480
opt_use_supersampling  = False
opt_more_supersamples  = False
num_neighbours         = 5
opt_knn_method         = 'reciprocal squared'
opt_noise              = 2.0**-24.0
opt_drop               = 0.0         # drop (do not use) a fraction of the input data
opt_clip_data          = True        # do not use data points outside the visible area (affects interpolation and colour scale)
opt_seed               = 0           # 0 means use system seed (noise and drop never use system seed)
opt_centre_x           = 0.0
opt_centre_y           = 0.0
max_point_distance     = 0.08        # don't draw score colours when further away from data point than this
show_points            = True
show_point_border      = 0.006       # in weight-space units
show_point_size        = 0.030       # in weight-space units
profile_curve_scale    = 5.0         # how many curves per score; 5 = every 0.2 score
profile_curve_strength = 0.800       # in pixel unit
grid_lines_strength    = 3.200 / zoom
triagle_edge_strength  = 3.875 / zoom
opt_jitter             = True        # move location of samples slightly to prevent aliasing effects
opt_dither             = True        # use dithering to exceed png's 8-bit colour precision
opt_preview            = True        # write preview pictures (unfinished lines are transparent)
opt_num_preview        = 2           # limit the number of preview pictures available at any time
opt_clean_up_preview   = True
tsv_version            = 2           # input file format
tsv_match_row          = 'Median'    # change to '123' to plot scores for seed 123 (also matches seed 1234)
tsv_match_column       = 'LAS-F1-total'  # use 'Sent-123:' to plot scores for sentence 123 only
opt_background_brightness = 0.80
opt_top_colour         = (1,1,1)     # default is to use white for the highest scores
opt_bottom_colours_lightness = 1.0
opt_adjust_colours = True
opt_show_legend    = True
opt_progress_interval  = 1.0

while len(sys.argv) >= 2 and sys.argv[1][:1] == '-':
    option = sys.argv[1]
    del sys.argv[1]
    if option in ('--help', '-h'):
        opt_help = True
        break
    elif option == '--debug':
        opt_debug = True
    elif option in ('--top-colour', '--highest-scores-in'):
        colour = sys.argv[1]
        del sys.argv[1]
        if colour == 'white':
            opt_top_colour = (1,1,1)
        elif colour == 'black':
            opt_top_colour = (0,0,0)
        elif colour == 'purple':
            opt_top_colour = (0.4,0,1)
        elif colour == 'light-purple':
            opt_top_colour = (0.7,0.5,1)
        elif colour.count(':') == 2:
            colour = colour.split(':')
            opt_top_colour = (
                float(colour[0]),
                float(colour[1]),
                float(colour[2]),
            )
        else:
            raise ValueError, 'unsupported colour %r' %colour
    elif option in ('--bottom-colours', '--bottom-scores-in'):
        colours = sys.argv[1]
        del sys.argv[1]
        if colours.startswith('light'):
            opt_bottom_colours_lightness = 0.85
        else:
            raise ValueError, 'unsupported colour scheme %r' %colours
    elif option in ('--background', '--background-brightness'):
        opt_background_brightness = float(sys.argv[1])
        del sys.argv[1]
    elif option == '--sentence':
        tsv_match_column = 'Sent-%s:' %(sys.argv[1])   # colon needed to avoid matching 123 for 1
        del sys.argv[1]
    elif option == '--tsv-version':
        tsv_version = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--tsv-match-column', '--tsv-match-columns'):
        tsv_match_column = sys.argv[1]
        del sys.argv[1]
    elif option in ('--tsv-match-row', '--parser-seed'):
        tsv_match_row = sys.argv[1]
        del sys.argv[1]
    elif option == '--scale':
        scale = float(sys.argv[1])
        width = int(scale*width)
        height = int(scale*height)
        zoom *= scale
        grid_lines_strength /= scale
        triagle_edge_strength /= scale
        del sys.argv[1]
    elif option in ('--keep-previews', '--no-not-clean-up'):
        opt_clean_up_preview = False
    elif option in ('--previews', '--number-of-previews'):
        opt_num_preview = int(sys.argv[1])
        del sys.argv[1]
    elif option in ('--no-preview', '--do-not-write-preview-files'):
        opt_preview = False
    elif option in ('--no-dither', '--no-dithering'):
        opt_dither = False
    elif option in ('--no-jitter', '--use-grid-samples'):
        opt_jitter = False
    elif option in ('--triagle-edge-strength', '--triagle-line-thickness'):
        triagle_edge_strength = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--grid-lines-strength', '--grid-line-thickness'):
        grid_lines_strength = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--profile-curve-strength', '--profile-line-thickness'):
        profile_curve_strength = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--profile-curve-scale', '--curves-per-score'):
        profile_curve_scale = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--show-point-size', '--data-point-size'):
        show_point_size = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--do-not-show-points', '--no-points'):
        show_points = False
    elif option in ('--max-point-distance', '--interpolation-radius'):
        max_point_distance = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--centre', '--xy'):
        opt_centre_x = float(sys.argv[1])
        opt_centre_y = float(sys.argv[2])
        del sys.argv[2]
        del sys.argv[1]
    elif option in ('--seed', '--random-seed'):
        opt_seed = int(sys.argv[1])
        if opt_seed:
            random.seed(opt_seed)
        del sys.argv[1]
    elif option in ('--drop', '--drop-points'):
        opt_drop = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--noise', '--add-noise'):
        opt_noise = float(sys.argv[1])
        del sys.argv[1]
    elif option in ('--neg-log-noise', '--add-negative-log-noise'):
        opt_noise = 2.0**(-float(sys.argv[1]))
        del sys.argv[1]
    elif option == '--knn-method':
        opt_knn_method = sys.argv[1].replace('-', ' ')
        del sys.argv[1]
    elif option in ('--knn-neighbours', '--k'):
        num_neighbours = int(sys.argv[1])
        del sys.argv[1]
    elif option[:12] in ('--supersampl', '--anti-alias'):
        if opt_use_supersampling:
            opt_more_supersamples = True
        opt_use_supersampling = True
    elif option == '--height':
        height = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--width':
        width = int(sys.argv[1])
        del sys.argv[1]
    elif option == '--zoom':
        zoom = float(sys.argv[1])
        del sys.argv[1]
    elif option == '--no-legend':
        opt_show_legend = False
    elif option == '--progress-interval':
        opt_progress_interval = float(sys.argv[1])
        del sys.argv[1]
    else:
        print 'Unsupported option %s' %option
        opt_help = True
        break

if len(sys.argv) != 3:
    opt_help = True

if opt_help:
    print_usage()
    sys.exit(0)

tsv = open(sys.argv[1], 'rb')
outname = sys.argv[2]

show_point_radius = show_point_size / 2.0

if show_points and max_point_distance < show_point_radius:
    sys.stderr.write('Correcting max_point_distance %.6f to show_point_radius %.6f so that the data points are shown.\n' %(
        max_point_distance, show_point_radius
    ))
    max_point_distance = show_point_radius

triangle_edges = []
for a,b,c,d,e,f in [
    (0,0,1,0,1,0),
    (0,1,0,1,0,0),
    (1,0,0,0,0,1),
]:
    x1,y1,_ = project_3_weights.project((a,b,c))
    x2,y2,_ = project_3_weights.project((d,e,f))
    triangle_edges.append(((x1,y1),(x2,y2)))

def get_point_xy_for_pixel_xy(x,y, lzoom = None):
    global zoom
    global width
    global height
    if lzoom is None:
        lzoom = zoom
    return (
        opt_centre_x + (x-width/2.0) / lzoom,
        opt_centre_y + (height/2.0-y) / lzoom
    )

# make sure legend is visible
# TODO: offer options to influence it
bbox_right, bbox_top = get_point_xy_for_pixel_xy(width, 0)
bbox_left, bbox_bottom = get_point_xy_for_pixel_xy(0, height)
legend_x2 = bbox_right # - 0.1
legend_x1 = legend_x2 - 0.7
legend_y2 = bbox_top # - 0.1
legend_y1 = bbox_bottom # + 0.1 # legend_y2 - 1.0
legend_margin = 0.15 # 0.0625
legend_lines_strength = 1.5 * grid_lines_strength * zoom

# png writing code adapted from
# http://stackoverflow.com/questions/902761/saving-a-numpy-array-as-an-image

raw_data = []
transparent = b'\x00\x00\x00\x00'
line = width * transparent
for i in range(height):
    raw_data.append(b'\x00' + line)

def png_pack(png_tag, data):
    chunk_head = png_tag + data
    return (struct.pack("!I", len(data)) +
            chunk_head +
            struct.pack("!I", 0xFFFFFFFF & zlib.crc32(chunk_head)))

def write_png(filename):
    global raw_data
    global width
    global height
    out = open(filename, 'wb')
    out.write(b'\x89PNG\r\n\x1a\n')
    out.write(png_pack(b'IHDR', struct.pack("!2I5B", width, height, 8, 6, 0, 0, 0)))
    png_data = b''.join(raw_data)
    out.write(png_pack(b'IDAT', zlib.compress(png_data, 9)))
    out.write(png_pack(b'IEND', b''))
    out.close()

def replace_png_line(line, row_index):
    global raw_data
    data = b'\x00' + line
    raw_data[row_index] = data

def avg_colour(colours, weights = None):
    retval = []
    for i in range(3):
        sum_c = 0.0
        sum_w = 0.0
        for j, colour in enumerate(colours):
            c = colour[i]
            if weights:
                w = weights[j]
            else:
                w = 1.0
            sum_c += w*c
            sum_w += w
        retval.append(sum_c / sum_w)
    return retval

def filter_colour(c1, c2):
    return (c1[0]*c2[0], c1[1]*c2[1], c1[2]*c2[2])

def dither_and_pack(colour, x, y):
    retval = []
    for cvalue in colour:
        if opt_dither:
            cvalue = int(1020.99 * cvalue)
            bvalue = cvalue // 4
            pattern = cvalue % 4
            if pattern > 0:
                xo = x % 2
                yo = y % 2
                if pattern == 1 and xo == 0 and yo == 0:
                    bvalue += 1
                elif pattern == 2 and xo == yo:
                    bvalue += 1
                elif pattern == 3 and (xo == 1 or yo == 0):
                    bvalue += 1
            cvalue = bvalue
        else:
            cvalue = int(255.99 * cvalue)
        if cvalue < 0 or cvalue > 255:
            cvalue = 96 + int(64*random.random())
        retval.append(chr(cvalue))
    retval.append(chr(255))
    return b''.join(retval)

colour_scale = [
    opt_top_colour,
    ( 0.4 + 0.6*opt_top_colour[0],
      0.0 + 0.2*opt_top_colour[1],
      0.4 + 0.6*opt_top_colour[2]), # magenta or dark magenta
    ( 0.6 + 0.4*opt_top_colour[0],
      0.0 + 0.1*opt_top_colour[1],
      0.0 + 0.1*opt_top_colour[2]), # red or dark red
    avg_colour([
    ( 0.8 + 0.2*opt_top_colour[0],
      0.8 + 0.2*opt_top_colour[1],
      0.0 + 0.1*opt_top_colour[2]), # yellow or dark yellow
    ( 1.0, 1.0, 0.6)],
    [1-opt_bottom_colours_lightness, opt_bottom_colours_lightness]
    ),
    (opt_bottom_colours_lightness,1,opt_bottom_colours_lightness), # green
    (opt_bottom_colours_lightness,1,1), # cyan
    (opt_bottom_colours_lightness,opt_bottom_colours_lightness,1), # blue
]

profile_curve_colour = (0.35, 0.25, 0.00)  # dark brown
grid_line_colour     = (0.00, 0.00, 0.00)  # black
triangle_edge_colour = (0.00, 0.00, 0.35)  # dark blue
white                = (1.00, 1.00, 1.00)
background_colour    = (
    opt_background_brightness,
    opt_background_brightness,
    opt_background_brightness
)
legend_background_colour = white

point2score = {}
point_tree = kdtree.create(dimensions=2)
max_score = 0.0
min_score = 100.0

bbox_left, bbox_bottom = get_point_xy_for_pixel_xy(0, height)

def add_point_with_score(x, y, score):
    global max_score
    global min_score
    if opt_clip_data \
    and x < bbox_left or x > bbox_right or y < bbox_bottom or y > bbox_top:
        return
    point2score[(x,y)] = score
    point_tree.add((x,y))
    if score > max_score:
        max_score = score
    if score < min_score:
        min_score = score

if tsv_version == 1:
    tsv_x_column = 0
    tsv_x_column = 1
    tsv_score_columns = [2]
    tsv_num_scores = 1.0
    tsv_multi_scores = False
elif tsv_version == 2:
    header = tsv.readline().split()
    tsv_x_column = header.index('X')
    tsv_y_column = header.index('Y')
    tsv_score_columns = []
    for i, label in enumerate(header):
        if label.startswith(tsv_match_column):
            tsv_score_columns.append(i)
    if not tsv_score_columns:
        raise ValueError, 'no column found with label %r' %tsv_match_column
    tsv_num_scores = float(len(tsv_score_columns))
    tsv_multi_scores = (tsv_num_scores > 1)
elif tsv_version == 0:
    # read points without score from task file
    tsv_x_column = 0
    tsv_y_column = 1
    tsv_score_columns = [2]
    tsv_num_scores = 1
    tsv_multi_scores = False
    tsv_seen = {}
else:
    raise ValueError, 'unsupported tsv format version %r' %tsv_version

count = 0
rnd = random.Random()
while True:
    line = tsv.readline()
    if not line:
        break
    if tsv_version == 0:
        # ./test-m-bist-multi-en_ewt-subsets-3.sh 303 "fr_partut:0.829167 fr_sequoia:0.581995 fr_spoken:-0.411162" fr_gsd
        # (following code also works for tab-separated args (for task-farming)
        line = line.replace(':', ' ')
        line = line.replace('"', ' ')
        fields = line.split()
        a = fields[-6]
        b = fields[-4]
        c = fields[-2]
        if (a, b, c) in tsv_seen:
            continue
        tsv_seen[(a, b, c)] = None
        a = float(a)
        b = float(b)
        c = float(c)
        x, y, _ = project_3_weights.project((a,b,c))
        fields = (x, y, 85.0-0.01*count)
    if tsv_version == 2 and not line.startswith(tsv_match_row):
        continue
    if tsv_version > 0:
        fields = line.split()
    x = float(fields[tsv_x_column])
    y = float(fields[tsv_y_column])
    s = 0.0
    for i in tsv_score_columns:
        s += float(fields[i])
    if tsv_multi_scores:
        s /= tsv_num_scores
    if tsv_match_column.startswith('Sent'):
        s *= 10.0
    if opt_noise or opt_drop:
        if opt_seed:
            fields.append('%d' %opt_seed)
        used_fields = []
        used_fields.append(str(fields[tsv_x_column]))
        used_fields.append(str(fields[tsv_y_column]))
        for i in tsv_score_columns:
            used_fields.append(str(fields[i]))
        seed = hash('\t'.join(used_fields))
        seed += hash(x) * 2**64
        seed += hash(y) * 2**128
        seed += hash(s) * 2**192
        if opt_seed:
            seed += hash((x,y,s,opt_seed)) * 2**256
            seed += hash(opt_seed) * 2**320
        else:
            seed += hash((x,y,s)) * 2**256
        rnd.seed(seed)
        r1 = rnd.random()
        r2 = rnd.random()
    if opt_drop and r1 < opt_drop:
        continue
    if opt_noise:
        s += opt_noise * r2
    add_point_with_score(x, y, s)
    count += 1
tsv.close()
if tsv_version == 0:
    tsv_seen = None

sys.stdout.write('Added %d points\n' %count)

sys.stdout.write('Score range: %.6f to %.6f\n' %(min_score, max_score))


def distance_func(a, b):
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def get_knn(x,y,k):
    global point2details
    global point_tree
    global width
    global height
    candidates = point_tree.search_knn((x,y), k, distance_func)
    retval = []
    for kdnode, _ in candidates:
        cx = kdnode.data[0]
        cy = kdnode.data[1]
        point = (cx,cy)
        score = point2score[point]
        d2 = (cx-x)**2 + (cy-y)**2
        retval.append((d2, point, score))
    retval.sort()
    return retval

def update_d2(neighbours, newq):
    for i in range(len(neighbours)):
        point = neighbours[i][1]
        score = neighbours[i][2]
        d2 = (newq[0]-point[0])**2 + (newq[1]-point[1])**2
        neighbours[i] = (d2, point, score)
    neighbours.sort()

def get_score_from_neighbours(neighbours, newq = None, k = 0):
    if newq:
        update_d2(neighbours, newq)
    if k:
        neighbours_k = neighbours[:k]
    else:
        neighbours_k = neighbours
    d02 = neighbours_k[0][0]
    if opt_knn_method == 'reciprocal squared' and d02 < 0.000001:
        return neighbours_k[0][2]
    score_sum = 0.0
    weight_sum = 0.0
    for d2, point, score in neighbours_k:
        if opt_knn_method == 'reciprocal squared':
            w = 1.0 / d2
        elif opt_knn_method == 'average':
            w = 1.0
        elif opt_knn_method == 'reciprocal squared plus 1':
            w = 1.0 / (1.0+d2)
        else:
            raise ValueError, 'unknown knn method %r' %opt_knn_method
        score_sum += w*score
        weight_sum += w
    return score_sum/weight_sum

def get_colour_for_score(score):
    global opt_adjust_colours
    global max_score
    d = max_score - score
    if opt_adjust_colours:
        dscore = max_score - min_score
        if dscore > 0.000001:
            d = 2.0 * d / (max_score - min_score)
        else:
            d = 0.0
    for i, interval in enumerate([0.1, 0.4, 0.5, 1.0, 1.0, 2.0]):
        if d < interval:
            return avg_colour(colour_scale[i:i+2], [interval-d, d])
        d -= interval
    return colour_scale[-1]

scale_start = legend_y1 + legend_margin
scale_end   = legend_y2 - legend_margin
legend_m = (min_score-max_score)/(scale_start-scale_end)
legend_b = min_score-legend_m*scale_start
try:
    legend_zoom = zoom*(scale_end-scale_start)/(max_score-min_score)
except ZeroDivisionError:
    sys.stderr.write('Warning: Legend zoom undefined. Setting it to 1.0\n')
    legend_zoom = 1.0

def print_dict_subset(f_out, keys, d):
    rows = []
    for key in sorted(keys):
        rows.append('%s: %r' %(key, d[key]))
    rows.append('')
    f_out.write('\n'.join(rows))

if opt_debug:
    print_dict_subset(sys.stderr, """
width height
bbox_right bbox_top
bbox_left bbox_bottom
legend_x2
legend_x1
legend_y2
legend_y1
legend_margin
legend_lines_strength
min_score max_score
scale_start legend_margin
scale_end
legend_m
legend_b
legend_zoom zoom
    """.split(), locals())

def get_legend_colour(x, y, point_x, point_y):
    if point_x < legend_x1 + legend_margin \
    or point_x > legend_x1 + 3 * legend_margin \
    or point_y < legend_y1 + legend_margin \
    or point_y > legend_y2 - legend_margin:
        return legend_background_colour
    score = legend_m * point_y + legend_b
    if point_x < legend_x1 + 2 * legend_margin:
        return get_colour_for_score(score)
    lines_strength = legend_lines_strength / legend_zoom
    _, y1 = get_point_xy_for_pixel_xy(x, y-0.5, zoom)
    _, y2 = get_point_xy_for_pixel_xy(x, y+0.5, zoom)
    q1 = legend_m * y1 + legend_b
    q2 = legend_m * y2 + legend_b
    grid_fraction, scale = get_grid_fraction_for_axis(q1, q2, lines_strength)
    if scale == -1 \
    or scale == 1  and point_x > legend_x1 + 2.90 * legend_margin \
    or scale == 2  and point_x > legend_x1 + 2.60 * legend_margin \
    or scale == 10 and point_x > legend_x1 + 2.40 * legend_margin \
    or scale == 20 and point_x > legend_x1 + 2.10 * legend_margin:
        return legend_background_colour
    return filter_colour(legend_background_colour, avg_colour(
        [white, grid_line_colour],
        [1.0-grid_fraction, grid_fraction]
    ))

total_subpixel_count = 0
profile_curve_subpixel_count = 0

def get_knn_colour_for_picture_xy(x, y, neighbours):
    global show_points
    global num_neighbours
    global max_point_distance
    global min_score
    global max_score
    global legend_score_0
    global total_subpixel_count
    global profile_curve_subpixel_count
    total_subpixel_count += 1
    point_x, point_y = get_point_xy_for_pixel_xy(x, y)
    update_d2(neighbours, (point_x, point_y))
    neighbours = neighbours[:num_neighbours+2]
    d0 = neighbours[0][0]**0.5
    if d0 < max_point_distance:
        on_curve = False
        if show_points and d0 < show_point_radius:
            score = neighbours[0][2]
        else:
            score  = get_score_from_neighbours(neighbours[:num_neighbours])
            # profile curves
            base_rounded_score = int(profile_curve_scale*score)
            a = profile_curve_strength / 2.0
            for alpha in range(0, 360, 30):
                xo = a * math.sin(2*alpha/math.pi)
                yo = a * math.cos(2*alpha/math.pi)
                point_xo, point_yo = get_point_xy_for_pixel_xy(x+xo, y+yo)
                # to speed up computation, we re-use the neighbours for
                # the centre of the pixel ignoring that the set of neighbours
                # sometimes changes; we keep 2 extra neighbours and
                # re-order the list to catch the most common errors
                score_o = get_score_from_neighbours(neighbours, (point_xo, point_yo), num_neighbours)
                rounded_score = int(profile_curve_scale*score_o)
                if rounded_score != base_rounded_score:
                    on_curve = True
                    break
        if on_curve:
            colour = profile_curve_colour
            profile_curve_subpixel_count += 1
        else:
            colour = get_colour_for_score(score)
            if show_points and d0 > show_point_radius:
                colour = avg_colour(
                    [colour, background_colour],
                    [0.2, 0.8]
                )
            if show_points \
            and show_point_radius-show_point_border < d0 < show_point_radius:
                colour = avg_colour(
                    [colour, (0,0,0)],
                    [0.2, 0.8]
                )
    else:
        colour = background_colour
    return colour

def get_knn_colour_for_pixel(x, y):
    global opt_use_supersampling
    global opt_more_supersamples
    point_x, point_y = get_point_xy_for_pixel_xy(x, y)
    neighbours = get_knn(point_x, point_y, num_neighbours+4)
    if not opt_use_supersampling:
        # just 1 sample per pixel
        return get_knn_colour_for_picture_xy(x, y, neighbours)
    else:
        colours = []
        if not opt_more_supersamples:
            # 9 samples per pixel
            for xob in (-0.333, 0.0, 0.333):
                for yo in (-0.333, 0.0, 0.333):
                    xo = xob
                    if opt_jitter:
                        xo += 0.333*(0.8*random.random()-0.4)
                        yo += 0.333*(0.8*random.random()-0.4)
                    colours.append(get_knn_colour_for_picture_xy(x+xo, y+yo, neighbours))
        else:
            # 25 samples per pixel
            for xob in (-0.4, -0.2, 0.0, 0.2, 0.4):
                for yo in (-0.4, -0.2, 0.0, 0.2, 0.4):
                    xo = xob
                    if opt_jitter:
                        xo += 0.200*(0.8*random.random()-0.4)
                        yo += 0.200*(0.8*random.random()-0.4)
                    colours.append(get_knn_colour_for_picture_xy(x+xo, y+yo, neighbours))
        return avg_colour(colours)

def get_grid_fraction_for_axis(q1, q2, lines_strength = None):
    global grid_lines_strength
    if not lines_strength:
        lines_strength = grid_lines_strength
    q_start = min(q1,q2)
    q_end   = max(q1,q2)
    a = lines_strength/2.0
    q = (q_start+q_end)/2.0
    q_width = q_end-q_start
    for scale in (None, 1, 2, 10, 20):
        if scale is None:
            nearest_line = 0.0
        else:
            nearest_line = round(q*scale)/scale
        nearest_start = nearest_line - a
        nearest_end   = nearest_line + a
        if q_end < nearest_start or q_start > nearest_end:
            a = a / 1.92
            continue
        if nearest_start < q_start and nearest_end > q_end:
            return (1.0, scale)
        if nearest_start < q_start:
            return ((nearest_end - q_start)/q_width, scale)
        if nearest_end > q_end:
            return ((q_end - nearest_start)/q_width, scale)
        return ((nearest_end - nearest_start)/q_width, scale)
    return (0.0, -1)

def get_grid_fraction_for_pixel(x, y):
    topleft_x, topleft_y = get_point_xy_for_pixel_xy(x-0.5, y-0.5)
    bottomright_x, bottomright_y = get_point_xy_for_pixel_xy(x+0.5, y+0.5)
    fraction_x, _ = get_grid_fraction_for_axis(topleft_x, bottomright_x)
    remaining = 1.0 - fraction_x
    fraction_y, _ = get_grid_fraction_for_axis(topleft_y, bottomright_y)
    return fraction_x + remaining * fraction_y

def get_triangle_hit_for_picture_xy(x, y):
    point_x, point_y = get_point_xy_for_pixel_xy(x, y)
    r2 = point_x**2+point_y**2
    if r2 > 1.0:
        return 0.0
    min_d_from_edge = 99.9*triagle_edge_strength
    for edge_start, edge_end in triangle_edges:
        edge_x = edge_end[0] - edge_start[0]
        edge_y = edge_end[1] - edge_start[1]
        tr_point_x = point_x - edge_start[0]
        tr_point_y = point_y - edge_start[1]
        len_edge = (edge_x**2+edge_y**2)**0.5
        d_from_edge = abs(edge_y*tr_point_x-edge_x*tr_point_y)/len_edge
        if d_from_edge < min_d_from_edge:
            min_d_from_edge = d_from_edge
    if min_d_from_edge < triagle_edge_strength / 2.0:
        return 1.0
    else:
        return 0.0

def get_triangle_fraction_for_pixel(x, y):
    global opt_use_supersampling
    global opt_more_supersamples
    # TODO: it probably would be faster to first check against
    #       a list of boxes covering the triangle edges
    hits = []
    # for rendering the triangle, we use higher
    # supersampling than requested by the user for
    # general pixels
    if not opt_use_supersampling:
        # 9 samples per pixel
        for xob in (-0.333, 0.0, 0.333):
            for yo in (-0.333, 0.0, 0.333):
                xo = xob
                if opt_jitter:
                    xo += 0.333*(0.8*random.random()-0.4)
                    yo += 0.333*(0.8*random.random()-0.4)
                hits.append(get_triangle_hit_for_picture_xy(x+xo, y+yo))
    elif not opt_more_supersamples:
        # 25 samples per pixel
        for xob in (-0.4, -0.2, 0.0, 0.2, 0.4):
            for yo in (-0.4, -0.2, 0.0, 0.2, 0.4):
                xo = xob
                if opt_jitter:
                    xo += 0.200*(0.8*random.random()-0.4)
                    yo += 0.200*(0.8*random.random()-0.4)
                hits.append(get_triangle_hit_for_picture_xy(x+xo, y+yo))
    else:
        # 49 samples per pixel
        for xob in (-0.429, -0.286, -0.143, 0.0, 0.143, 0.286, 0.429):
            for yo in (-0.429, -0.286, -0.143, 0.0, 0.143, 0.286, 0.429):
                xo = xob
                if opt_jitter:
                    xo += 0.143*(0.8*random.random()-0.4)
                    yo += 0.143*(0.8*random.random()-0.4)
                hits.append(get_triangle_hit_for_picture_xy(x+xo, y+yo))
    return sum(hits)/len(hits)

def get_colour_for_pixel(x, y):
    point_x, point_y = get_point_xy_for_pixel_xy(x, y)
    if opt_show_legend \
    and legend_x1 < point_x < legend_x2 \
    and legend_y1 < point_y < legend_y2:
        return get_legend_colour(x, y, point_x, point_y)
    colour = get_knn_colour_for_pixel(x, y)
    # grid lines
    grid_fraction = get_grid_fraction_for_pixel(x,y)
    colour = filter_colour(colour, avg_colour(
        [white, grid_line_colour],
        [1.0-grid_fraction, grid_fraction]
    ))
    # triagle edges
    triangle_edge_fraction = get_triangle_fraction_for_pixel(x,y)
    return filter_colour(colour, avg_colour(
        [white, triangle_edge_colour],
        [1.0-triangle_edge_fraction, triangle_edge_fraction]
    ))

startt = time.time()
last_verbose = 0.0
last_preview = 0.0
preview_threshold = 20.0
preview_index = 0
preview_lastnames = []

shuffleindex = list(range(height))
random.shuffle(shuffleindex)
for s_index in range(height):
    y = shuffleindex[s_index]
    line = []
    start_this_line = time.time()
    for x in range(width):
        colour = get_colour_for_pixel(x,y)
        line.append(dither_and_pack(colour, x, y))
    line = b''.join(line)
    replace_png_line(line, y)
    remaining = height - 1 - y
    now = time.time()
    if not remaining:
        eta = now
    else:
        eta = now + (now-start_this_line) * remaining
    now = time.time()
    if now > last_verbose + opt_progress_interval:
        info = []
        percentage = 100.0 * (1.0+s_index) / float(height)
        info.append('Line %d (%.1f%%)' %(s_index+1, percentage))
        elapsed = now - startt
        if elapsed < 100.0:
            info.append('elapsed %.1f seconds' %elapsed)
        elif elapsed < 6000.0:
            info.append('elapsed %.1f minutes' %(elapsed/60.0))
        elif elapsed < 50.0*3600.0:
            info.append('elapsed %.1f hours' %(elapsed/3600.0))
        else:
            info.append('elapsed %.1f days' %(elapsed/3600.0/24.0))
        remaining = height - 1 - s_index
        if not remaining:
            eta = now
        else:
            eta = now + (now-start_this_line) * remaining
        info.append('ETA = %s' %time.ctime(eta))
        sys.stderr.write(', '.join(info))
        sys.stderr.write('  \r')
        last_verbose = now

    if opt_preview and now > last_preview + preview_threshold \
    and s_index < height - 1:
        # write preview picture
        last_slash = outname.rfind('/')
        if last_slash >= 0:
            preview_name = '%s/preview-%s-%03d.png' %(
                outname[:last_slash],
                outname[last_slash+1:-4],
                preview_index
            )
        else:
            preview_name = 'preview-%s-%03d.png' %(outname[:-4], preview_index)
        write_png(preview_name)
        preview_lastnames.append(preview_name)
        while len(preview_lastnames) > opt_num_preview:
            # delete previous preview picture
            if os.path.exists(preview_lastnames[0]):
                os.unlink(preview_lastnames[0])
            del preview_lastnames[0]
        preview_threshold *= 1.414
        last_preview = now
        preview_index += 1

now = time.time()
duration = now - startt
sys.stderr.write('\nPrepared lines in %.1f seconds.\n' %duration)
sys.stdout.write('Profile curve subpixel count: %d of %d (%.4f%%)\n' %(
    profile_curve_subpixel_count,
    total_subpixel_count,
    100.0*profile_curve_subpixel_count/float(total_subpixel_count)
))
sys.stderr.write('\n\nWriting PNG file\n\n')

write_png(outname)

if opt_clean_up_preview:
    # delete preview picture(s)
    for preview_name in preview_lastnames:
        if os.path.exists(preview_name):
            os.unlink(preview_name)

sys.stderr.write('Finished ' + time.ctime(time.time())+'\n')


