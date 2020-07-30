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

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/tbemb/tbev-prediction

source ${PRJ_DIR}/config/locations.sh

cd $RESULT_DIR

SCRIPTS_DIR=${PRJ_DIR}/scripts
SCRIPT=${SCRIPTS_DIR}/uuparser-tbemb-test.sh
COLLECTION=fake_aaa:fake_bbb:fake_ccc:fake_ddd
LCODE=fake

for DS in 000 001 002 003 004 005 006 007 008 010 012 015 020 030 040 050 060 070 080 090 100 110 120 130 140 150 160 180 200 220 250 300 350 400 500 600 700 800 900 ; do
    NAME=te-${LCODE}-w${DS}
    mkdir ${NAME}
    ${SCRIPTS_DIR}/gen_tasks.py  \
        --width 3.2 --height 2.72  \
        --num-points 200           \
        --seed 100 --num-workers 1  \
        --with-density-decay --decay-strength $DS  \
        --worker-dir ${NAME}                       \
        --script-template $SCRIPT --tab-tasks      \
        --collection $COLLECTION &
    sleep 2
done

NAME=te-${LCODE}-nodecay
mkdir ${NAME}
${SCRIPTS_DIR}/gen_tasks.py  \
    --width 3.2 --height 2.72  \
    --num-points 200           \
    --seed 100 --num-workers 1  \
    --worker-dir ${NAME}         \
    --script-template $SCRIPT --tab-tasks      \
    --collection $COLLECTION &

wait

for DS in 000 001 002 003 004 005 006 007 008 010 012 015 020 030 040 050 060 070 080 090 100 110 120 130 140 150 160 180 200 220 250 300 350 400 500 600 700 800 900 ; do
    ${SCRIPTS_DIR}/fake-graph.sh te-${LCODE}-w$DS &
    sleep 2
done

${SCRIPTS_DIR}/fake-graph.sh te-${LCODE}-nodecay &

wait

