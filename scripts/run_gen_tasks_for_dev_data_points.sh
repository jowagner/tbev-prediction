#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

#SBATCH -p compute          # which partition to run on
#SBATCH -B 2:6:2
#SBATCH -J genp2h    # name for the job
#SBATCH --mem=96000
# -N 1-1

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/tbemb/tbev-prediction
source ${PRJ_DIR}/config/locations.sh

SCRIPT=${SCRIPTS_DIR}/uuparser-tbemb-test.sh


for COLLECTION in \
    fr_gsd:fr_partut:fr_sequoia:fr_spoken \
    en_ewt:en_gum:en_lines:en_partut \
    cs_cac:cs_cltt:cs_fictree:cs_pdt \
; do
    date
    echo ${COLLECTION}
    LCODE=$(echo ${COLLECTION} | cut -d_ -f1)
    NAME=te-worker-dev-${LCODE}
    mkdir ${NAME}
    ${SCRIPTS_DIR}/gen_tasks.py   \
        --width 3.2 --height 2.72  \
        --num-points 200            \
        --seed 100 --num-workers 1   \
        --script-template ${SCRIPT}  \
        --worker-dir ${NAME}         \
        --tab-tasks                  \
        --collection ${COLLECTION}   \
        > dev-samples-${COLLECTION}.log
    mv ${NAME}/worker-1000.sh dev-data-point-parsing-${LCODE}.tfm
    rmdir ${NAME}
done

