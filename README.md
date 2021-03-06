# Treebank Embedding Vector Prediction for Out-of-Domain Dependency Parsing

Thanks for interest in our code for
[Treebank Embedding Vectors for Out-of-Domain Dependency Parsing](https://www.aclweb.org/anthology/2020.acl-main.778/).
The code for the first step, training all required multi-treebank parsing models, is ready. 
If you want to replicate the work asap you can start training these models now.
This is
independent of the remaining code and it easily takes a week.
We aim to complete cleaning and making the code more easy to use by
 the end of August 2020.

For questions, please open an issue in this repository.

If you use this code please cite the paper linked above.


## Dependencies (and Installation Suggestions)

The scripts in this repository currently assume the following:
* This repository is located in `~/tbemb/tbev-prediction`.
  We added code to support setting `PRJ_DIR` to an alternative location but
  this has not been tested. Please let us know if you used this
  variable to run in a different location successfully or if
  you encounter problems.
* `python2`, `python3` and `python` executables are in `PATH` and `python` is Python 2. If necessary, create symlinks or wrapper scripts with these names in a new folder and point to this folder at the start of `PATH`.
* ELMoForManyLangs is in `~/tbemb/ELMoForManyLangs`.
  An alternative location can be configured in `tbev-prediction/config/locations.sh`.
* UUParser with our multi-treebank extension is in `~/tbemb/uuparser`.
  An alternative location can be configured in `tbev-prediction/config/locations.sh`.

TODO:
* re-construct what Python environments are needed and when they need to be activated


### UUParser with Our Multi-Treebank Extension

https://github.com/jowagner/uuparser/tree/tbemb

```
cd ~/tbemb
git clone ...
cd uuparser
git checkout tbemb
```

The script `uuparser-tbemb-create-virtualenv.sh` creates a `virtualenv` Python environment for
running this parser. A list of Python dependencies can be found in it.

TODO: The list of Python packages
seems to be bigger than needed. From memory, uuparser-tbemb only needs
`numpy`, `dynet` and `cython`.

### ELMoForManyLangs

To replicate all
development results of the paper, ELMo-derived sentence representation are needed.
For the winning model, this dependency can be skipped.

### Linear Tree Combiner

https://github.com/jowagner/ud-combination

TODO: add instructions how to put / symlink this into the expected location

Alternatively, a dummy script that uncompresses an input `.conllu.gz` file to
a `.conllu` file using the same command line as our combiner should also work
as we here only explore scenarios with 1 system output.


### KD-Tree

Only needed for visualising the candidate vectors and LAS in weight space
as in Figures 1 and 2 in the paper.
Can be skipped if not using `render-graph.py`.

https://github.com/stefankoegl/kdtree
You can install this as a python module or simply place it in our
scripts folder:

```
cd ~/tbemb
git clone git@github.com:stefankoegl/kdtree.git
cd tbev-prediction/scripts
ln -s ~/tbemb/kdtree/kdtree.py
```

In our ACL 2020 paper, we used version 0.15 of kdtree, which further
requires the file `bounded_priority_queue.py`.


## Prepare Treebanks

It is recommended to place the UD folder in the project folder or any
other folder from which you plan to run the experiment,
e.g. via a symlink:
```
mkdir workdir
cd workdir
ln -s $HOME/data/ud-treebanks-v2.3/
```

The name of the symlink must not contain whitespace as we use split() to parse
intermediate outputs.

Below, we run scripts with `./<scriptname>` but they can also be called
from other locations such as `workdir`.
It is recommended to create a symlink to the `scripts` folder or to
add it to the `PATH` variable.

If replicating preliminary experiments with the 5 genres of the English Web Treebank:
* Split EWT into genres: `split-en-ewt.py`

If using a newer version than UD v2.3:
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
    * `gen_train_multi-subset-3.py`:
      Writes a `.tfm` task-farming file with one command per line
      training all multi-treebank models needed for development,
      i.e. training on each combination of three treebanks of the 
      four treebanks of each development language.
      Add option `--epochs 20` or lower if you are pressed for time
      (the Czech models involving `cs_pdt` take quite long)
      and are ok with less accurate models
      (default is to train for 30 epochs).
      UUParser picks the best model from all trained epochs
      according to development data.
      Typical usage:
        ```
        gen_train_multi-subset-3.py
        ```
    * Run the `.tfm` file generated by the above command with task-farming
      or job arrays (check your cluster documentation), or just with `bash`
      (you may then want to append ` &` to each line and insert `wait`
      every `n` lines and at the end of the file to keep `n` CPU cores
      busy) in a suitable Python environment, see above.
      (If you do not have access to strong server CPUs but have a GPU
      you may want to change the wrapper script to set dynet to use
      your GPU.)
    * You can append the `.tfm` file for training parsing models needed for
      testing later (see below) now as the choice of parsing models does
      not change during development.

2. Choose weight vectors to try and generate the task list for parsing:
    * `run_gen_tasks_for_dev_data_points.sh`: Chooses the candidate
      treebank embedding vectors as weighted averages of the fixed vectors
      and writes taskfarming files for each development language.
      The weight space is restricted as in the ACL 2020 paper.
      Remove option `--tab-tasks` from the script's call to `gen_tasks.py`
      to obtain shell commands in the `.tfm` file, rather than
      tab-separated lists of command arguments.
      Note that the option `--seed` of `gen_tasks.py`, which is called in
      this script, was not used in the
      ACL 2020 experiments, making small deviations in the candidate
      set of treebank vectors unavoidable.
      See `gen_tasks.py --help` and `explore-candidate-point-decay.sh`
      for options to change the vector sampling.
      The option `--skip-indomain-parsing` may sound right for
      out-of-domain experiment but this option was **not** used in the
      ACL 2020 experiment as in-domain results were used as training data
      for the k-NN models.
      TODO: produce more clear log output: log all points and clearly mark
      rejected points

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

## Testing on Parallel UD Treebanks (PUD)

1. Train multi-treebanks models:
    * `gen_pud_training.py`:
      Writes a `.tfm` task-farming file with one command per line training
      all multi-treebank models needed for PUD testing, i.e. training on
      the combination of all treebanks with training data for each test
      language.
      Supports the same options as `gen_train_multi-subset-3.py`.
      Expects as input the output of `assess-pud-situation.sh`.
      Typical usage:
        ```
        assess-pud-situation.sh ud-treebanks-v2.3 > pud-situation.txt
        gen_pud_training.py --treebank-folder ud-treebanks-v2.3 <pud-situation.txt >pud-training.tsv
        ```

    * See development steps for running the `.tfm` file.
      If running in parallel with fewer workers than tasks consider moving
      Czech and Russian tasks to the top of the file to avoid long idle
      times of some of the workers near the end of the job.

    * `run_gen_tasks_for_pud_data_points.sh`: calls `gen_tasks.py` with
      options needed for PUD test set experiments, e.g. adjusting the
      number of samples to the dimensionality of the treebank vector
      weight space and switching off the box clipping used in development.
      For the highest dimensionality 4, each language takes about half an
      hour.


## Acknowledgements

This research was funded by the ADAPT Centre for Digital Content Technology
under the SFI Research Centres Programme (Grant 13/RC/2106) and the European
Regional Development Fund.

