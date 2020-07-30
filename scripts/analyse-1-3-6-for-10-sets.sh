#!/bin/bash

INPUT_TSV="$1"
FILTER="$2"
FILTER2="$3"
FILTER3="$4"
CATEGORIES="$5"
UNTIDY=$6
LOG=$7

TARGET_COLUMN=17   # median

# example usage:
# ./analyse-1-3-6-for-10-sets.sh results-with-both-combiners.tsv "-v no-neg"

#for SET in \
#fr_gsd$'\t'fr_partut+fr_sequoia+fr_spoken \
#fr_partut$'\t'fr_gsd+fr_sequoia+fr_spoken \
#fr_sequoia$'\t'fr_gsd+fr_partut+fr_spoken \
#fr_spoken$'\t'fr_gsd+fr_partut+fr_sequoia \
# ; do

#for SET in \
#en_ewt$'\t'en_gum+en_lines+en_partut \
#en_gum$'\t'en_ewt+en_lines+en_partut \
#en_lines$'\t'en_ewt+en_gum+en_partut \
#en_partut$'\t'en_ewt+en_gum+en_lines \
# ; do

#for TE in fr_gsd fr_partut fr_sequoia fr_spoken ; do
#  for TR in fr_partut+fr_sequoia+fr_spoken fr_gsd+fr_sequoia+fr_spoken fr_gsd+fr_partut+fr_spoken fr_gsd+fr_partut+fr_sequoia ; do

for TE in en_ewt en_gum en_lines en_partut ; do
  for TR in en_gum+en_lines+en_partut en_ewt+en_lines+en_partut en_ewt+en_gum+en_partut en_ewt+en_gum+en_lines ; do

SET=$TE$'\t'$TR


    echo
    echo "= set (test tab train): $SET =="
    echo
    for NUM_SAMPLES in 1 3 ; do
        for SEEDS in 1 3 ; do
            echo "== samples: $NUM_SAMPLES seeds: $SEEDS =="
            for CATEGORY in $CATEGORIES ; do
                echo $CATEGORY $SEEDS `cat $INPUT_TSV | fgrep "$SET" | fgrep $FILTER | fgrep $FILTER2 | fgrep "$FILTER3" | fgrep decay | fgrep hashed | grep -E $'(^|\t)'$CATEGORY | grep -E $'\t'$NUM_SAMPLES$'\t'$SEEDS$'\t'"[0-9]"$'\t'"[5-9]"$'\t' | cut -f$TARGET_COLUMN | average.py `

#done ; done ; done ; done ; done | tee $UNTIDY | ./tidy-up-analysis.py | fgrep -v -- ----
done ; done ; done ; done ; done | ./tidy-up-analysis.py | fgrep -v -- ----

#echo $FILTER >> $LOG
#echo $FILTER2 >> $LOG
#echo $FILTER3 >> $LOG
#echo $CATEGORIES >> $LOG
