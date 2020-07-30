#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner


NUM_CANDIDATES=49500

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/tbemb/tbev-prediction
source ${PRJ_DIR}/config/locations.sh

echo "Collections of size 2"

for COLLECTION in  \
    fi_ftb:fi_tdt     \
    ko_gsd:ko_kaist     \
    pt_bosque:pt_gsd     \
    es_ancora:es_gsd      \
    sv_lines:sv_talbanken  \
; do
    date
    echo ${COLLECTION}
    LCODE=$(echo ${COLLECTION} | cut -d_ -f1)
    NAME=te-worker-pud-${LCODE}
    mkdir -p ${NAME}
    ${SCRIPTS_DIR}/gen_tasks.py   \
            --worker-dir ${NAME}         \
	    --no-subsets                 \
	    --no-box                      \
	    --maximum-centre-distance 1.6  \
	    --minimum-weight -0.4           \
            --collection ${COLLECTION}       \
	    --tab-tasks --num-workers 1       \
	    --num-points 23                    \
            --seed 100                         \
	    --median-interpolations 0          \
            --num-candidates ${NUM_CANDIDATES}  \
	    > pud-samples-2-${COLLECTION}.log
    mv ${NAME}/worker-1000.sh pud-data-point-parsing-2-${COLLECTION}.tfm
    rmdir ${NAME}
done

echo "Collections of size 3"

for COLLECTION in  \
    it_isdt:it_partut:it_postwita  \
    ru_gsd:ru_syntagrus:ru_taiga   \
; do
    date
    echo ${COLLECTION}
    LCODE=$(echo ${COLLECTION} | cut -d_ -f1)
    NAME=te-worker-pud-${LCODE}
    mkdir -p ${NAME}
    ${SCRIPTS_DIR}/gen_tasks.py   \
            --worker-dir ${NAME}         \
	    --no-subsets                \
	    --no-box                      \
	    --maximum-centre-distance 1.6  \
	    --minimum-weight -0.4          \
            --collection ${COLLECTION}      \
	    --tab-tasks --num-workers 1      \
	    --num-points 120                  \
            --seed 100                         \
	    --median-interpolations 21         \
            --num-candidates ${NUM_CANDIDATES}  \
	    > pud-samples-3-${COLLECTION}.log
    mv ${NAME}/worker-1000.sh pud-data-point-parsing-3-${COLLECTION}.tfm
    rmdir ${NAME}
done

echo "Collections of size 4"

for COLLECTION in  \
    cs_cac:cs_cltt:cs_fictree:cs_pdt     \
    en_ewt:en_gum:en_lines:en_partut      \
    fr_gsd:fr_partut:fr_sequoia:fr_spoken  \
; do
    date
    echo ${COLLECTION}
    LCODE=$(echo ${COLLECTION} | cut -d_ -f1)
    NAME=te-worker-pud-${LCODE}
    mkdir -p ${NAME}
    ${SCRIPTS_DIR}/gen_tasks.py   \
            --worker-dir ${NAME}         \
	    --no-subsets                \
	    --no-box                      \
	    --maximum-centre-distance 1.6  \
	    --minimum-weight -0.4          \
            --collection ${COLLECTION}      \
	    --tab-tasks --num-workers 1      \
	    --num-points 320                  \
            --seed 100                         \
	    --median-interpolations 9          \
            --num-candidates ${NUM_CANDIDATES}  \
	    > pud-samples-4-${COLLECTION}.log
    mv ${NAME}/worker-1000.sh pud-data-point-parsing-4-${COLLECTION}.tfm
    rmdir ${NAME}
done

