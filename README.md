# SCR

Hi! This is the repository for the EMNLP 2023 oral paper: [Continual Event Extraction with Semantic Confusion Rectification](https://aclanthology.org/2023.emnlp-main.732.pdf).

## Requirements

Please make sure you have installed the packages in [./environment.yml](https://github.com/nju-websoft/SCR/blob/main/environment.yml).

## Data Preprocessing

We use the ACE, ERE and MAVEN datasets for evaluation. Due to that the ACE and ERE datasets are not released publicly, we can't provide the dataset after processing. You can obtain the MAVEN datasets after processing through this [link]( https://drive.google.com/file/d/1-Zud2K_X0cmffwXAMBZ_WwNd9u88vHEE/view?usp=drive_link).

For ACE and ERE datasets, please first follow [OneIE](https://github.com/GerlinGreen/OneIE) to process the dataset. Then you should process the data format to be like ./data/{DATASET_NAME}+/toy.json and name them "train.json", "valid.json", "test.json", respectively.

## Pretrained Models

Download pretrained language model from [huggingface](https://huggingface.co/bert-base-uncased) and put it into the [./pertrain_model directory](https://github.com/nju-websoft/SCR/tree/main/pretrain_model).

## Training and Testing

To run this project, first install the dependencies from `environment.yml`, prepare the dataset files under `./data/{DATASET_NAME}/`, and place the pretrained BERT model under `./pretrain_model/`.

Then train and test the SCR model with one of the dataset scripts:

```cmd
sh run_ace.sh
sh run_ere.sh
sh run_maven.sh
```

Each script runs `main.py` with the matching config file in `./config/`. You can also run a config directly:

```cmd
python main.py --config ./config/{DATASET_NAME}.ini
```

You can modify the hyperparameters in ./config/{DATASET_NAME}.ini 

Note that {DATASET_NAME} is one of the the dataset names include ace, ere and maven. 

## Citation

The base code was used from the following paper:

```
@inproceedings{wang-etal-2023-continual,
    title = "Continual Event Extraction with Semantic Confusion Rectification",
    author = "Wang, Zitao  and Wang, Xinyi and Hu, Wei",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.732",
    doi = "10.18653/v1/2023.emnlp-main.732",
    pages = "11945--11955",
}
```
