# indiv-annotation-learning

models.py contains model code for predicting annotator ratings; target_model.py contains code for predicting the target group of a statement.
older_models.py is not in use; contains code for older prediction models based on the architectures from Davani et al.

## Quickstart:
Clone this repo. Download the "data" and "tokenized_toxjson" folders from Google Drive and place them in the "indiv_annotation_learning" directory. 

"tokenized_toxjson" contains pre-tokenized data for `--model_type='multitask-demographic'` and `--model_type='multitask' --survey_info='text-sep'`, the most commonly run models. To run the model with different inputs or different input loading, include the `--reload_dataset` parameter. This will also automatically save the tokenized data to a folder for quick reloading.

Requirements: Python 3.10.4 or above

Install required packages: `pip install -r requirements.txt`


## To train:
`python train.py [args]`
where [args] includes:

`--model type`: Choose from 'multitask-annotator' (ID or survey info only), 'multitask-demographic' (demographic info only), or 'multitask' (both)

`--survey_info`: Choose from 'id-sep' (prepends annotator ID), 'text-sep' (prepends survey responses), or 'both-sep' (prepends both)

`--reload_dataset`: Whether to reload the data (recommended only if you've changed anything before data tokenization, otherwise loading the pre-tokenized data directly saves some time)

`--cap`: caps the amount of training data used (helpful for debugging)

`--n_epochs`: Number of training epochs to run

`--train_batch_size`: Training batch size. 32 recommended unless you face memory issues

`--eval_batch_size`: Dev/test batch size; same as above.

`--save_model_to`: Path to which the model should be saved.

`--from_saved_model`: Continue training from an existing model.

`--n_gpu`: Number of GPUs to train on (defaults to 1).

Recommended settings for input `survey_responses [SEP] demographics [SEP] text_to_rate`: 

`python3 train.py --model_type='multitask' --survey_info='text-sep' --n_epochs=6 --train_batch_size=32 --eval_batch_size=32 --save_model_to='path_to_save_your_model'`

Recommended settings for input `demographics [SEP] text_to_rate`:

`python3 train.py --model_type='multitask-demographic' --n_epochs=6 --train_batch_size=32 --eval_batch_size=32 --save_model_to='path_to_save_your_model'`

Evaluation occurs automatically every 500 steps during training. 

To train the target model:

`python target_model.py [args]`
where [args] includes:

`--cap`: caps the amount of training data used (helpful for debugging)

`--n_epochs`: Number of training epochs to run

`--train_batch_size`: Training batch size. 32 recommended unless you face memory issues

`--eval_batch_size`: Dev/test batch size; same as above.

`--save_model_to`: Path to which the model should be saved.

`--from_saved_model`: Continue training from an existing model.

`--n_gpu`: Number of GPUs to train on (defaults to 1).

Recommended settings:

`python3 target_model.py --n_epochs=8 --save_model_to='saved_models/target_11_08_20'`


## To evaluate:
To properly load a model for evaluation, keyword args must match those used during training (e.g., if `--model_type=multitask` was used during training, `--model_type=multitask` must be a keyword arg when evaluating).

Evaluate the annotator rating model:
`python train.py --no_train [args]`

Relevant args:

`--no_train`: Don't train before running evaluation.

`--from_saved_model`: Load an existing annotator rating model.

E.g., to evaluate the annotator rating model with input `survey_responses [SEP] demographics [SEP] text_to_rate`:

`python train.py --no_train --model_type='multitask' --survey_info='text-sep' --train_batch_size=32 --eval_batch_size=32 --from_saved_model='path_to_saved_annotator_rating_model'`

Evaluate the target model and annotator rating model jointly:

Include these params.
`--joint_eval`: Evaluate annotator rating and target group models jointly.

`--from_saved_target_model`: Load an existing target group model.

E.g., to evaluate the models jointly with input `survey_responses [SEP] demographics [SEP] text_to_rate`:

`python train.py --no_train --joint_eval --model_type='multitask' --survey_info='text-sep' --train_batch_size=32 --eval_batch_size=32 --from_saved_model='path_to_saved_annotator_rating_model' --from_saved_target_model='path_to_saved_target_model'`

Evaluate the target model alone:
python target_model.py --no_train --from_saved_model='path_to_saved_target_model'


# Attributions

This codebase was started by Eve Fleisig, and was later contributed to by Kashyap Coimbatore Murali and Harbani Jaggi. This was then published in EMNLP 2024 under the title "Accurate and Data-Efficient Toxicity Prediction when Annotators Disagree" if you cite this work please use the following citation:

```
@misc{jaggi2024accuratedataefficienttoxicityprediction,
      title={Accurate and Data-Efficient Toxicity Prediction when Annotators Disagree}, 
      author={Harbani Jaggi and Kashyap Murali and Eve Fleisig and Erdem Bıyık},
      year={2024},
      eprint={2410.12217},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.12217}, 
}

```