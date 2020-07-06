#!/bin/bash

# (C) 2019 Dublin City University
# All rights reserved. This material may not be
# reproduced, displayed, modified or distributed without the express prior
# written permission of the copyright holder.

# Author: Joachim Wagner

# usage:
# ./assess-pud-situation.sh > pud-situation.txt
# or
# ./assess-pud-situation.sh treebank-folder > pud-situation.txt

if [ -n $1 ] ; then
    cd $1
fi

ls -d UD_*PUD* | grep -o -E "UD_[^-]*" | cut -c4- > pud_languages.txt

for L in `cat pud_languages.txt` ; do

    echo == $L ==
    ls -ld UD_${L}*/* | fgrep PUD | fgrep -i readme
    ls -ld UD_${L}*/* | fgrep PUD | fgrep -i test.conllu
    echo
    ls -ld UD_${L}*/* | fgrep -v PUD | fgrep train.conllu
    echo
    ls -ld UD_${L}*/* | fgrep -v PUD | fgrep dev.conllu
    echo
    echo

done

