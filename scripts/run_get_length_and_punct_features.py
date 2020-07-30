#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner


test -z $1 && echo "Missing TBID"
test -z $1 && exit 1
TARGET_TBID=$1

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

process_file() {
    TBNAME=$1
    TBID=$2
    DATASET=$3
    OUTDIR=$4
    OPTIONS="$5"
    echo "== ${TBID} (${TBNAME}) ${DATASET} =="
    INFILE=${UD_TREEBANK_DIR}/${TBNAME}/${TBID}-ud-${DATASET}.conllu
    SENTREPFILE=${OUTDIR}/${TBID}-ud-${DATASET}-length.hdf5
    if [ -e ${INFILE} ] ; then
        if [ -e ${SENTREPFILE} ] ; then
            echo "Sentence length ${SENTREPFILE} exist; skipping; to re-new, delete first"
        else
            ./get_length_punct_sentence_features.py ${INFILE} ${SENTREPFILE}
        fi
    else
        echo "Input ${INFILE} not found"
    fi
}

OUTDIR=${RESULT_DIR}/length-and-punct

mkdir -p ${OUTDIR}

for DATASET in train dev test ; do
    process_file ${TARGET_TBNAME} ${TARGET_TBID} ${DATASET} ${OUTDIR}
done

