#!/bin/bash

# (C) 2018, 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

test -z $1 && echo "Missing model seed"
test -z $1 && exit 1
SEED=$1

test -z "$2" && echo "Missing model TBIDs with weights"
test -z "$2" && exit 1
MODEL_WEIGHTS="$2"

# example for MODEL_WEIGHTS: fr_gsd:0.333333 fr_partut:0.333333 fr_sequoia:0.333333

test -z $3 && echo "Missing test TBID"
test -z $3 && exit 1
TEST_TBID=$(echo $3)

if [ -n "$4" ]; then
    WORKER_ID=$4
fi

source ${PRJ_DIR}/config/locations.sh

MODEL_BASE=${RESULT_DIR}/uuparser_multilingual_std/uuparser-tbemb
OUTPUT_BASE=${RESULT_DIR}/data-points
PARSER=src/parser.py
PARSER_DIR=${UUPARSER_DIR}
MEM=256      # initial amount of dynet memory; will be increased automatically by dynet if needed

MODEL_TBIDS=$(echo ${MODEL_WEIGHTS} | tr -d '0123456789.:-' | tr ' ' ':')
MODEL_WEIGHTS_D=$(echo ${MODEL_WEIGHTS} | tr ' ' '+')

echodate()
{
    echo `date +%Y-%m-%dT%H:%M:%S` seed-${SEED} with ${MODEL_WEIGHTS} on ${TEST_TBID}: $*
}

if [ -n "$WORKER_ID" ]; then
    echodate "Started by worker $WORKER_ID"
fi

echodate "Selecting best model"

EPOCH=$(fgrep "Model score after epoch" \
    ${MODEL_BASE}/seed-${SEED}/${MODEL_TBIDS}/stdout.txt | \
    sort -n -k 6 | tail -n 1 | cut -d' ' -f5 | cut -d: -f1)

if [ -z ${EPOCH} ] ; then
    echodate "Error: Cannot find best epoch."
    exit 1
fi

FAKE_TBID=xx_xxx
FAKE_TBNAME=UD_Xxx-xxx

for DATASET in dev train ; do

    WORK_DIR=${OUTPUT_BASE}/workdirs/seed-${SEED}-with-${MODEL_WEIGHTS_D}-on-${TEST_TBID}-${DATASET}
    rm -rf ${WORK_DIR}
    INFILE=${UD_TREEBANK_DIR}/*/${TEST_TBID}-ud-${DATASET}.conllu

    OUTPUT_DIR=${WORK_DIR}/output
    DATA_DIR=${WORK_DIR}/input
    mkdir -p ${OUTPUT_DIR}
    mkdir -p ${DATA_DIR}/${FAKE_TBNAME}
    ln -s ${INFILE} ${DATA_DIR}/${FAKE_TBNAME}/${FAKE_TBID}-ud-dev.conllu
    touch ${DATA_DIR}/${FAKE_TBNAME}/${FAKE_TBID}-ud-train.conllu  # parser complains if missing

    MODEL_DIR=${WORK_DIR}/model
    rm -rf ${MODEL_DIR}
    mkdir ${MODEL_DIR}
    ln -s ${MODEL_BASE}/seed-${SEED}/${MODEL_TBIDS}/params.pickle \
          ${MODEL_DIR}/params.pickle
    ln -s ${MODEL_BASE}/seed-${SEED}/${MODEL_TBIDS}/barchybrid.model${EPOCH} \
          ${MODEL_DIR}/barchybrid.model
    echodate "Parsing ${DATASET} with model from epoch ${EPOCH}"
    cd ${PARSER_DIR}
    python2 ${PARSER}          \
        --dynet-mem ${MEM}      \
        --dynet-seed ${SEED}    \
        ${DYNET_OPTIONS}         \
        --predict                 \
        --outdir ${OUTPUT_DIR}     \
        --include ${FAKE_TBID}      \
        --modeldir ${MODEL_DIR}      \
        --weighted-tbemb              \
        --tb-weights "${MODEL_WEIGHTS}"  \
        --datadir ${DATA_DIR}          \
        --userl                        \
        --k 3                          \
        --disable-pred-eval            \
        --multiling                    \
        2> ${OUTPUT_DIR}/epoch-${EPOCH}-stderr.txt \
        >  ${OUTPUT_DIR}/epoch-${EPOCH}-stdout.txt

    # Move output outside work dir in case we want to keep it
    PREDICT_DIR=${OUTPUT_BASE}/predictions/${MODEL_TBIDS}-on-${TEST_TBID}-${DATASET}/seed-${SEED}
    PREDICTION=${PREDICT_DIR}/${MODEL_WEIGHTS_D}.conllu
    mkdir -p ${PREDICT_DIR}
    mv ${OUTPUT_DIR}/${FAKE_TBID}.conllu ${PREDICTION}

    echodate "Evaluating ${DATASET}"
    EVAL_DIR=${OUTPUT_BASE}/eval/${MODEL_TBIDS}-on-${TEST_TBID}-${DATASET}/seed-${SEED}
    EVALUATION=${EVAL_DIR}/${MODEL_WEIGHTS_D}.txt
    mkdir -p ${EVAL_DIR}

    $PRJ_DIR/scripts/fast_las_eval.py   \
        --sentences                      \
        -v                               \
        ${INFILE}                        \
        ${PREDICTION}                    \
        > ${EVALUATION}

    echodate "Cleaning up ${DATASET}"
    rm -rf ${PREDICTION}
    rm -rf ${WORK_DIR}

done

echodate "Done"

