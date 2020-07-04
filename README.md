# Treebank embedding vector prediction for out-of-domain dependency parsing

Over the coming weeks, before ACL 2020, we add cleaned-up code here to reproduce experiments of

Joachim Wagner, James Barry and Jennifer Foster. 2020. Treebank Embedding Vectors for Out-of-domain Dependency Parsing. In Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL 2020).
https://www.aclweb.org/anthology/2020.acl-main.778/

## Dependencies (and Installation Suggestions)

### UUParser with multi-treebank extension

https://github.com/jowagner/uuparser/tree/tbemb

```
git clone ...
cd uuparser
git checkout tbemb
```

### ElmoForManyLangs

### KD-Tree

https://github.com/stefankoegl/kdtree
You can install this as a python module or simply place it in our
scripts folder:

```
cd tbev-prediction/scripts
git clone ...
```


## Prepare Treebanks

* Split EWT into genres: `split-en-ewt.py`
* Add new treebank names and codes to `config/tbnames.tsv`

## Obtaining Data Points (tbweights --> LAS)

1. Train grammars for all treebank combinations of interest and for the number of seeds
needed for the k-NN experiments.
In the ACL 2020 paper, we use 9 seeds and explore all
treebank combinations with 3 of the 4 usable UD v2.3
treebanks for Czech, English and French in development.
For final testing, we use the combination of all usable
non-PUD treebanks.
The choice of seeds is not critical as an
exact reproduction of parser training is not possible
because parallel training on GPUs randomises the
order of numeric operations.
(Models for the same seed tend to make the same
predictions for a few epochs but then start to
diverge noticeable.)
`gen_train_multi-subset-3.py`,
`../gen_pud_training.py`,
`grove-worker-train-parser-t12.job`,
`grove-train-multi-en_ewt-subsets-3.job`,
`ichec-train-multi-en_ewt-subsets-3.job`,
`train-multi-en_ewt-subsets-3.sh`

2. Choose tbemb weights to try and generate parsing task list: `gen_tasks.py`
(check `--help`), `../pick_candidate_wvec.py`, `ichec-gen-test.job`, `grove-gen-tasks-t12.job`

For experiments with the k-NN method, do not use `--skip-indomain-parsing`
of `gen_tasks.py` as k-NN needs parse results for the in-domain treebanks.

3. Parse both training data and dev data with the selected tbemb weights:
`ichec-test-all.job` runs workers in te-worker/.
The workers call `test-m-bist-multi-en_ewt-subsets-3.sh` or `test-uuparser-multi-en_ewt-subsets-3.sh`.
(These scripts have an alternative line `for DATASET in dev ; do` that excludes parsing
of training data. This is not suitable for the k-NN experiment that requires
parse results for the training data.)
If you want to parse with a different parser or with a different parser setting,
e.g. elmo, please create a new wrapper script and update variable `script` in `gen_tasks.py`
(or manually update the task file(s)).

On grove, we can use xmlrpc-based taskfarming:
`grove-worker-parsing-for-data-points-t12.job`
(there is a comment how to start the master)

4. Collect LAS summary table: `collect_ewt_results.py`.
This can be done in parallel for each data set, see `ichec-collect-results.job`.

5. Create graphs to verify that results make sense: `render-graph.py`, `update-graphs.sh`

```shell
ln -s $HOME/tbemb/data-points/ data-points
```

## Prepare Sentence Representations for Similiarity Measure

`../concat-treebanks.sh`
`../run_concat.sh`
`get_elmo_sentence_features.py`
`get_similarity_matrix.py`
`run_get_elmo_sentence_features.sh`
`get_tfidf_sentence_features.py`
`run_get_tfidf_sentence_features.sh`
`run_get_length_and_punct_features.py`

```shell
ln -s $HOME/tbemb/sent-rep/length-and-punct/ length-and-punct
```

## Predict Weights and Generate Parsing Tasks (on Dev)

`gen_tbemb_sampling.py`
`test-tbemb-sampling-template.sh`
`preproc-parse-sampling-task.py`
`preproc-parse-sampling-uuparser-task.py`
`grove-gen-tbemb-sampling-1-of-6.job`
`grove-gen-tbemb-sampling-2.job`
`grove-gen-tbemb-sampling-3.job`
`grove-gen-tbemb-sampling-5.job`
`grove-gen-tbemb-sampling-4.job`
`grove-gen-tbemb-sampling-6.job`

Checking progress:
```shell
for I in tb*samp*txt ; do
    echo ; echo ; echo == $I ==
    fgrep "== Scenario" -A 7 $I | tail -n 9
    fgrep "Duration of gen_tbemb_sampling for scenario" $I | tail -n 1
done
```

For PUD and other final testing, use `--test-type test` with `gen_tbemb_sampling.py`
and add the test treebanks to the collection. Treebanks ending in `_pud` are
automatically excluded from model tbid triplets and from k-NN learning.

## Parse 

Compile task farming file. (Ok to make preliminary runs while
`gen_tbemb_sampling.py` is still running
but must be repeated with new task farming file when finished.
Example shows how to filter the task farming file by the tasks
of a previous run.)
```shell
find te-parse/ -type f | ./filter_tfm.py all-te-parse-31a-part-0923.tfm | sort > all-te-parse-31a-part-2359.tfm
cat all-te-parse-31a-part-31a-3001.tfm | xargs -d'\n' chmod 755
```

Task farming master:
```shell
xmlrpc_master.py --port 8743 --secret-from-file secret.txt --show-task all-te-parse-31a-part-2359.tfm
```

Task farming workers:
`grove-parse-sampling-uuparser-worker-t12.job`
and variants (not all up to date):
`grove-parse-sampling-worker-t12.job`
`grove-parse-sampling-single.job`
`grove-parse-sampling-worker-n0128d.job`
`grove-parse-sampling-worker-t10.job`
`grove-parse-sampling-worker-gpu.job`
`grove-parse-sampling-worker-t12.job`
`grove-parse-sampling-manual-t6.sh`

The component parsers are run with the `parse_*.sh` scripts.

## Combine and Evaluate

`gen_tbemb_combine.py`:

Compile task farming file. (Ok to make preliminary runs while
parsing is still running or unfinished
but must be repeated with new task farming file when finished.
The script automatically detects finished combine and eval
tasks of previous runs.)
```shell
source ~/tbemb/dynet-cpu-py27/bin/activate
find tbemb-sampling/ | \
    ./gen_tbemb_combine.py  \
        --treebank-dir /data/ud-treebanks-v2.3 \
        dev  \
        > combine-31a-001.tfm
```

Replace `dev` with `test` when testing on test data, e.g. PUD.

For the second run, if still also parsing, it's a good idea to
shuffle the tasks to spread the I/O load more evenly between
combining and evaluating:
```shell
find tbemb-sampling/ | \
    ./gen_tbemb_combine.py \
        --treebank-dir /data/ud-treebanks-v2.3  \
        dev  | \
    shuf > combine-31a-002.tfm
```

Task farming master:
```shell
xmlrpc_master.py --port 8544 \
    --secret-from-file secret.txt --show-task \
    combine-31a-001.tfm
```

Task farming workers:
`grove-combine-and-eval-worker-t12.job`

Repeat until task file is empty (at step 003 if not overlapping with parsing).

## Summary of Results

```Shell
find tbemb-sampling/ | ./collect_sampling_results.py > results.tsv
```

Condense:
`analyse-1-3-6-for-10-sets.sh`
`run_analyse.sh`


LaTeX table:
```Shell
find tbemb-sampling/ | \
    ./select_model.py --median  \
        > selected-median-models-with-dev-results.tsv
find tbemb-sampling/ | \
    ./filter_test_results.py  \
        selected-median-models-with-dev-results.tsv  \
        > dev-results-by-model.tsv
./latex_test_table.py --dev dev-results-by-model.tsv
```

Test results:
```Shell
./run_select-seed.sh > pud-best-of-7-seeds.tsv
find tbemb-sampling/ | ./collect_sampling_results.py \
    --model-seeds-from pud-best-of-7-seeds.tsv       \
    > pud-results-with-best-dev-seed.tsv
```

## Model Selection and Testing

```Shell
rm tbemb-sampling
ln -s /data/results/indom-dev/ \
    tbemb-sampling
find tbemb-sampling/ | \
    ./select_model.py  \
        > selected-best-models-with-dev-results.tsv
```

```Shell
rm tbemb-sampling
ln -s /data/results/indom-test/ \
    tbemb-sampling
find tbemb-sampling/ | \
    ./filter_test_results.py  \
        selected-best-models-with-dev-results.tsv  \
        > test-results-by-model.tsv
```

## LaTeX Table

```Shell
./latex_test_table.py test-results-by-model.tsv
```


## Acknowledgements

This research was funded by the ADAPT Centre for Digital Content Technology
under the SFI Research Centres Programme (Grant 13/RC/2106) and the European
Regional Development Fund.

