#!/bin/bash

# (C) 2019, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# See also ../elmo_scripts/run_elmoformanylangs.sh

test -z $1 && echo "Missing TBID"
test -z $1 && exit 1
TBID=$1

source ${PRJ_DIR}/config/locations.sh

OUTDIR=${RESULT_DIR}/elmo

mkdir -p ${OUTDIR}

TBNAME=$(grep ${TBID}\$ ${PRJ_DIR}/config/tbnames.tsv | cut -f1)
if [ -z ${TBNAME} ] ; then
    echo "Error: Cannot locate TBNAME for tbid ${TBID}."
    exit 1
fi

for DATASET in train dev test ; do
    echo "== ${TBID} ${DATASET} =="
    INFILE=${UD_CROSSMOR_DIR}/${TBNAME}/${TBID}-ud-${DATASET}.conllu
    TXTFILE=${OUTDIR}/${TBID}-ud-${DATASET}.txt
    TOKREPFILE=${OUTDIR}/${TBID}-ud-${DATASET}.hdf5
    SENTREPFILE=${OUTDIR}/${TBID}-ud-${DATASET}-sent-rep.hdf5
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
        if [ -e ${TOKREPFILE} ] ; then
            echo "Re-using ${TOKREPFILE}"
        else
            allennlp elmo --average ${TXTFILE} ${TOKREPFILE}
        fi
    else
        echo "Input ${TXTFILE} not found"
    fi
    if [ -e ${TOKREPFILE} ] ; then
        if [ -e ${SENTREPFILE} ] ; then
            echo "Sentence representations ${SENTREPFILE} exist; skipping; to re-new, delete first"
        else
            ./get_elmo_sentence_features.py ${TOKREPFILE} ${SENTREPFILE}
        fi
    else
        echo "Input ${TOKREPFILE} not found"
    fi
done

