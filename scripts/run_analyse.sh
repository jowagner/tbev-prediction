#!/bin/bash

RUN=en-33a

ALLMETHODS="pick-1st pick-2nd pick-3rd all3 equal random knnreduce knnavg-of-1 knnavg-of-3 knnavg-of-9 knncomb"

DYNAMICMETHODS="random knnreduce knnavg-of-1 knnavg-of-3 knnavg-of-9 knncomb"

KNNMETHODS="knnreduce knnavg-of-1 knnavg-of-3 knnavg-of-9 knncomb"

DISTANCES="cosine-char-567-grams-tfidf-with-unk-000 cosine-char-567-grams-tfidf-with-unk-025 cosine-char-567-grams-tfidf-with-unk-050 cosine-char-567-grams-tfidf-with-unk-075 cosine-char-567-grams-tfidf-with-unk-100 cosine-elmoformanylangs cosine-token-tfidf-with-unk-000 cosine-token-tfidf-with-unk-025 cosine-token-tfidf-with-unk-050 cosine-token-tfidf-with-unk-075 cosine-token-tfidf-with-unk-100 L2-char-567-grams-tfidf-with-unk-000 L2-char-567-grams-tfidf-with-unk-025 L2-char-567-grams-tfidf-with-unk-050 L2-char-567-grams-tfidf-with-unk-075 L2-char-567-grams-tfidf-with-unk-100 L2-elmoformanylangs L2-token-tfidf-with-unk-000 L2-token-tfidf-with-unk-025 L2-token-tfidf-with-unk-050 L2-token-tfidf-with-unk-075 L2-token-tfidf-with-unk-100 length-punct-and-cosine-char-567-grams-tfidf-with-unk-000 length-punct-and-cosine-char-567-grams-tfidf-with-unk-025 length-punct-and-cosine-char-567-grams-tfidf-with-unk-050 length-punct-and-cosine-char-567-grams-tfidf-with-unk-075 length-punct-and-cosine-char-567-grams-tfidf-with-unk-100 length-punct-and-cosine-elmoformanylangs length-punct-and-cosine-token-tfidf-with-unk-000 length-punct-and-cosine-token-tfidf-with-unk-025 length-punct-and-cosine-token-tfidf-with-unk-050 length-punct-and-cosine-token-tfidf-with-unk-075 length-punct-and-cosine-token-tfidf-with-unk-100 length-punct-and-L2-char-567-grams-tfidf-with-unk-000 length-punct-and-L2-char-567-grams-tfidf-with-unk-025 length-punct-and-L2-char-567-grams-tfidf-with-unk-050 length-punct-and-L2-char-567-grams-tfidf-with-unk-075 length-punct-and-L2-char-567-grams-tfidf-with-unk-100 length-punct-and-L2-elmoformanylangs length-punct-and-L2-token-tfidf-with-unk-000 length-punct-and-L2-token-tfidf-with-unk-025 length-punct-and-L2-token-tfidf-with-unk-050 length-punct-and-L2-token-tfidf-with-unk-075 length-punct-and-L2-token-tfidf-with-unk-100"

for FILTER1 in use-all-data oracle ; do
    for FILTER2 in any-weights corners no-negative ; do

        DIR="analysis/by-method"
        FILE=$(echo "restricted-to-${FILTER1}-${FILTER2}" | tr -d $'\t' )
	NAME=$DIR/$FILE
	mkdir -p $DIR
        echo "== $NAME =="
        ./analyse-1-3-6-for-10-sets.sh results-${RUN}.tsv $FILTER1 $FILTER2 . "$ALLMETHODS" ${NAME}.txt > ${NAME}.tsv
	wc -l ${NAME}.tsv

        for FILTER3 in $DISTANCES ; do
            DIR="analysis/by-knn-method"
            FILE=$(echo "restricted-to-${FILTER1}-${FILTER2}-${FILTER3}" | tr -d $'\t' )
	    NAME=$DIR/$FILE
	    mkdir -p $DIR
            echo "== $NAME =="
            ./analyse-1-3-6-for-10-sets.sh results-${RUN}.tsv $FILTER1 $FILTER2 $'\t'$FILTER3 "$KNNMETHODS" > ${NAME}.tsv
	    wc -l ${NAME}.tsv
        done

        for FILTER3 in $KNNMETHODS ; do
            DIR="analysis/by-distance"
            FILE=$(echo "restricted-to-${FILTER1}-${FILTER2}-${FILTER3}" | tr -d $'\t' )
	    NAME=$DIR/$FILE
	    mkdir -p $DIR
            echo "== $NAME =="
            ./analyse-1-3-6-for-10-sets.sh results-${RUN}.tsv $FILTER1 $FILTER2 $FILTER3 "$DISTANCES" > ${NAME}.tsv
	    wc -l ${NAME}.tsv
        done

    done
done

for FILTER1 in $KNNMETHODS ; do
    for FILTER2 in use-all-data oracle ; do
	for FILTER3 in length-punct-and-cosine-char-567-grams-tfidf-with-unk-000 length-punct-and-cosine-char-567-grams-tfidf-with-unk-050 ; do
            DIR="analysis/by-weights"
            FILE=$(echo "restricted-to-${FILTER1}-${FILTER2}-${FILTER3}" | tr -d $'\t' )
	    NAME=$DIR/$FILE
	    mkdir -p $DIR
            echo "== $NAME =="
            ./analyse-1-3-6-for-10-sets.sh results-${RUN}.tsv $FILTER1 $FILTER2 $FILTER3 "any-weights corners no-negative" > ${NAME}.tsv
	    wc -l ${NAME}.tsv
        done
    done
done

for FILTER1 in random ; do
    for FILTER2 in use-all-data ; do
	for FILTER3 in "?" ; do
            DIR="analysis/by-weights"
            FILE=$(echo "restricted-to-${FILTER1}" | tr -d $'\t' )
	    NAME=$DIR/$FILE
	    mkdir -p $DIR
            echo "== $NAME =="
            ./analyse-1-3-6-for-10-sets.sh results-${RUN}.tsv $FILTER1 $FILTER2 $FILTER3 "any-weights corners no-negative" > ${NAME}.tsv
	    wc -l ${NAME}.tsv
        done
    done
done


for FILTER1 in $KNNMETHODS ; do
    for FILTER2 in any-weights corners no-negative ; do
	for FILTER3 in length-punct-and-cosine-char-567-grams-tfidf-with-unk-000 length-punct-and-cosine-char-567-grams-tfidf-with-unk-050 ; do
            DIR="analysis/oracle-vs-knn"
            FILE=$(echo "restricted-to-${FILTER1}-${FILTER2}-${FILTER3}" | tr -d $'\t' )
	    NAME=$DIR/$FILE
	    mkdir -p $DIR
            echo "== $NAME =="
            ./analyse-1-3-6-for-10-sets.sh results-${RUN}.tsv $FILTER1 $FILTER2 $FILTER3 "use-all-data oracle" > ${NAME}.tsv
	    wc -l ${NAME}.tsv
        done

    done
done

