#!/bin/bash

test -z $1 && echo "Missing include file"
test -z $1 && exit 1
FILE=$1

test -z $2 && echo "Missing seed"
test -z $2 && exit 1
SEED=$2

test -z $3 && echo "Missing number of epochs"
test -z $3 && exit 1
EPOCHS=$3

test -z $4 && echo "Missing deadline"
test -z $4 && exit 1
DEADLINE=$4

if [ -n "$5" ]; then
    WORKER_ID=$5
fi

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/tbemb/tbev-prediction

if ! [ -e ${PRJ_DIR}/config/locations.sh ] ; then
    echo "PRJ_DIR not correct"
    exit 1
fi

source ${PRJ_DIR}/config/locations.sh

if [ -z "$UD_TREEBANK_DIR" ] ; then
    echo "UD_TREEBANK_DIR not set"
    exit 1
fi

if [ -z "$RESULT_DIR" ] ; then
    echo "RESULT_DIR not set"
    exit 1
fi

MEM=256 # initial amount of dynet memory; will be increased automatically by dynet if needed

PARSER_NAME=uuparser-tbemb
PARSER_DIR=${UUPARSER_DIR}
cd ${PARSER_DIR}
PARSER=src/parser.py

OUTDIR=${RESULT_DIR}/uuparser_multilingual_std/${PARSER_NAME}/seed-${SEED}
echo $OUTDIR
mkdir -p ${OUTDIR}/${FILE}

STATS_DIR=${OUTDIR}/stats
mkdir -p ${STATS_DIR}

LOG=${STATS_DIR}/${FILE}.training.start
hostname > $LOG
test -n $WORKER_ID && echo "worker: $WORKER_ID" >> $LOG
echo "now:" $(date --iso=s) >> $LOG
echo "parser: $PARSER" >> $LOG
echo "mem: $MEM" >> $LOG
echo "dynet options: $DYNET_OPTIONS" >> $LOG
echo "file: $FILE" >> $LOG
echo "outdir: $OUTDIR" >> $LOG
echo "ud_treebank_dir: $UD_TREEBANK_DIR" >> $LOG
echo "deadline: $DEADLINE" >> $LOG

python ${PARSER}  \
    --dynet-seed ${SEED}  \
    --dynet-mem ${MEM}  \
    ${DYNET_OPTIONS} \
    --outdir ${OUTDIR}/${FILE}  \
    --modeldir ${OUTDIR}/${FILE} \
    --datadir ${UD_TREEBANK_DIR} \
    --include ${FILE} \
    --epochs ${EPOCHS}  \
    --top-k-epochs 3  \
    --k 3      \
    --userl    \
    --multiling \
    --fingerprint  \
    --deadline ${DEADLINE} \
    2> ${OUTDIR}/${FILE}/stderr.txt \
    >  ${OUTDIR}/${FILE}/stdout.txt

touch ${STATS_DIR}/${FILE}.training.end

