# (C) 2018, 2020 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner
# 2018-06-14 training on predicted POS; udocker locations
# 2018-06-12 adjustments for grove cluster

# This file must be used with "source bin/activate" from bash or another
# script.


# Reduce hostname to first 2 characters as we don't want to write a
# configuration for each node on the cluster.
SIMPLEHOST=`echo ${HOSTNAME} | cut -c-2 | tr '23456789' '11111111'`

SETTING=${USER}@${SIMPLEHOST}

test -z ${PRJ_DIR} && PRJ_DIR=${HOME}/tbemb/tbev-prediction

# defaults

UD_TREEBANK_DIR=$HOME/data/ud-treebanks-v2.3
RESULT_DIR=$HOME/tbemb/workdir/results
UUPARSER_DIR=${HOME}/tbemb/uuparser/barchybrid
ELMOFORMANYLANGS_DIR=${HOME}/tbemb/ELMoForManyLangs
ELMO_MODEL_DIR=${HOME}/elmo
VENV_TBEMB=${HOME}/tbemb/dynet-cpu-py27

# deviations from the above defaults for specific users and systems

case "${SETTING}" in
"user1@n0")
    UD_TREEBANK_DIR=/some/other/place/ud-treebanks-v2.3
    ;;
"user2@n1")
    UD_TREEBANK_DIR=${HOME}/ud-treebanks-v2.3
    ;;
root*)
    # inside udocker
    UD_TREEBANK_DIR=/ud-treebanks-v2.3
    GDRIVE=gdrive:Research/ud-parsing-2018
    ;;
*)
    true
    ;;
esac

SCRIPTS_DIR=${PRJ_DIR}/scripts

