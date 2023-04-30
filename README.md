# Prismer
[![arXiv](https://img.shields.io/badge/arXiv-2303.02506-b31b1b.svg)](https://arxiv.org/abs/2303.02506)
 [![Hugginface Space](https://huggingface.co/datasets/huggingface/badges/raw/main/open-in-hf-spaces-sm-dark.svg)](https://huggingface.co/spaces/shikunl/prismer)

This directory is based off of Prismer which can be found [here](https://github.com/NVlabs/prismer). Helpful links to the paper and their huggingface page above.

It contains the Prismer code with updates made to support finetuning for classification on the Hateful Memes dataset based on the Facebook challenge.

## How to Run
1. Download the dataset from [here](https://www.kaggle.com/datasets/tianyating02/hatefulmeme). Note that the jsonl and images within `hm_data` are the actual data needed.
    - Take note of where you download the files as `configs/experts.yaml` and `configs/classifcation.yaml` will need to be updated with the location.
    - The unique combination of the `_seen` and `_unseen` datasets for dev and test can be created using `hm_data/combine_sets.py`. The train set is used for training and, the dev set is used for validation. The test set is used for the final metrics saved in the report, and is not touched until the very end.
2. Install all package dependencies by running `bash pip install -r requirements.txt`
3. Follow up setup steps in the [original repo](https://github.com/NVlabs/prismer) to download setup accelerate config, download expert pre-trained models, and download Prismer pre-trained models. 
4. Run `accelerate launch experts/generate_{EXPERT_NAME}.py` for each expert to generate the transformed images for Prismer.
5. Fine-tuning can be run with `accelerate launch train_classification.py  --exp_name {pre-trained model}`
    - If using PrismerZ update configs/classification.yaml to not use any of the experts. Otherwise populate all experts.
    - Outputs of training are saved under `logging`. Some sample runs are saved there, but this is not exhaustive of all runs ran.
6. Validation can be run with `accelerate launch demo_caption.py --exp_name {trained model} --from_checkpoint`.