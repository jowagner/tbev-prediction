#!/bin/bash

VENVNAME=dynet-cpu-py27
VENVDIR=venv
VENVPATH=$VENVDIR/$VENVNAME

if [ -e $VENVNAME ] ; then
    echo "Error: $VENVNAME already exists"
    exit 1
fi

mkdir -p $VENVDIR

virtualenv -p /usr/bin/python2.7 $VENVPATH

ACTIVATE=$VENVPATH/bin/activate

echo  >> ${ACTIVATE}
echo "# Project-specific settings" >> ${ACTIVATE}
echo "PRJ_DIR=\$HOME/tbemb/tbev-prediction" >> ${ACTIVATE}
echo "export PRJ_DIR" >> ${ACTIVATE}
echo >> ${ACTIVATE}
echo "DYNET_TYPE=cpu" >> ${ACTIVATE}
echo "export DYNET_TYPE" >> ${ACTIVATE}

source ${ACTIVATE}

for I in \
    numpy  \
    pydot \
    pandas \
    wheel \
    sklearn \
    h5py \
    cython \
    dynet \
; do
    echo
    echo == $I ==
    echo
    pip install --upgrade $I
done

