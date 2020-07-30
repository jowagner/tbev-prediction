#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

NAME=$1

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/tbemb/tbev-prediction
source ${PRJ_DIR}/config/locations.sh
source ${VENV_TBEMB}/bin/activate

TR_SET=a:b:c
TE_SET=d

SUFFIX="blackdot"

S=${NAME}/worker-1000.sh
O=imgout-fake-graph-${NAME}-${SUFFIX}.png

OPTIONS="--background 1.0"\
" --interpolation-radius 0.00"\
" --highest-scores-in black"\
" --bottom-scores-in light-colours"\
" --data-point-size 0.085"\
" --width 320 --height 272"\
" --scale 2.00"\
" --supersample"\
" --knn-neighbours 1"\
" --knn-method average"\
" --progress-interval 12.0"\
" --no-legend"\
" --tsv-version 0"

${SCRIPTS_DIR}/render-graph.py $OPTIONS $S $O 

