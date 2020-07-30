#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner


test -z $1 && echo "Missing TBID"
test -z $1 && exit 1
TARGET_TBID=$1

test -z $2 && echo "Missing vocab TBID"
test -z $2 && exit 1
VOCAB_TBID=$2

source ${PRJ_DIR}/config/locations.sh


get_tbname() {
    # declare -n retval=$1  # too new for ichec
    TBID=$2
    TBNAME=$(grep ${TBID}\$ ${PRJ_DIR}/config/tbnames.tsv | cut -f1)
    if [ -z ${TBNAME} ] ; then
        echo "Error: Cannot locate TBNAME for tbid ${TBID}."
        exit 1
    fi
    #retval=$TBNAME   # too new for ichec
    eval "$1=$TBNAME"
}

get_tbname TARGET_TBNAME ${TARGET_TBID}
get_tbname VOCAB_TBNAME  ${VOCAB_TBID}

process_file() {
    TBNAME=$1
    TBID=$2
    DATASET=$3
    TFIDFDIR=$4
    OPTIONS="$5"
    echo "== ${TBID} (${TBNAME}) ${DATASET} =="
    INFILE=${UD_TREEBANK_DIR}/${TBNAME}/${TBID}-ud-${DATASET}.conllu
    TXTFILE=${TFIDFDIR}/${TBID}-ud-${DATASET}.txt
    SENTREPFILE=${TFIDFDIR}/${TBID}-ud-${DATASET}-sent-rep.hdf5
    if [ -e ${INFILE} ] ; then
        if [ -e ${TXTFILE} ] ; then
            echo "Re-using ${TXTFILE}"
        else
            ../elmo_scripts/get_text_for_elmo.py ${INFILE} ${TXTFILE}
        fi
    else
        echo "Input ${INFILE} not found"
    fi
    if [ -e ${TXTFILE} ] ; then
        if [ -e ${SENTREPFILE} ] ; then
            echo "Sentence representations ${SENTREPFILE} exist; skipping; to re-new, delete first"
        else
            ./get_tfidf_sentence_features.py ${OPTIONS} ${TXTFILE} ${SENTREPFILE}
        fi
    else
        echo "Input ${TXTFILE} not found"
    fi
}

for UNK_PERCENTAGE in 000 100 025 050 075 ; do

    echo "= token tfidf with unknown percentage ${UNK_PERCENTAGE} ="

    OUTDIR=${RESULT_DIR}/token-tfidf-with-unk-${UNK_PERCENTAGE}

    mkdir -p ${OUTDIR}

    process_file ${VOCAB_TBNAME} ${VOCAB_TBID} train ${OUTDIR} "--unk-percentage ${UNK_PERCENTAGE} --exit-after-vocab"

    VOCAB_OPTIONS="--vocab ${OUTDIR}/${VOCAB_TBID}-ud-train.txt.vocab --unk-percentage ${UNK_PERCENTAGE}"

    for DATASET in train dev test ; do
        process_file ${TARGET_TBNAME} ${TARGET_TBID} ${DATASET} ${OUTDIR} "${VOCAB_OPTIONS}"
    done


    echo "= token character ${N}-grams tfidf with unknown percentage ${UNK_PERCENTAGE} ="
    OUTDIR=${RESULT_DIR}/char-567-grams-tfidf-with-unk-${UNK_PERCENTAGE}
    mkdir -p ${OUTDIR}
    process_file ${VOCAB_TBNAME} ${VOCAB_TBID} train ${OUTDIR} "--char-ngrams 5 7 --unk-percentage ${UNK_PERCENTAGE} --exit-after-vocab"
    VOCAB_OPTIONS="--vocab ${OUTDIR}/${VOCAB_TBID}-ud-train.txt.vocab --char-ngrams 5 7 --unk-percentage ${UNK_PERCENTAGE}"
    for DATASET in train dev test ; do
        process_file ${TARGET_TBNAME} ${TARGET_TBID} ${DATASET} ${OUTDIR} "${VOCAB_OPTIONS}"
    done

done

