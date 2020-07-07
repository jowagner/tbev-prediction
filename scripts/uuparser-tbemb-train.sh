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
source ${PRJ_DIR}/config/locations.sh

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

hostname > ${STATS_DIR}/${FILE}.training.start
test -n $WORKER_ID && echo $WORKER_ID >> ${STATS_DIR}/${FILE}.training.start

python ${PARSER}  \
    --dynet-seed ${SEED}  \
    --dynet-mem ${MEM}  \
    ${DYNET_OPTIONS} \
    --outdir ${OUTDIR}/${FILE}  \
    --modeldir ${OUTDIR}/${FILE} \
    --datadir ${UD_CROSSMOR_DIR}  \
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

