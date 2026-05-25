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

This code was further used for the following paper:

```
@inproceedings{al-monsur-etal-2026-event,
    title = "Event Detection with a Context-Aware Encoder and {L}o{RA} for Improved Performance on Long-Tailed Classes",
    author = "Al Monsur, Abdullah  and
      Bommisetty, Nitesh Vamshi  and
      Kim, Gene Louis",
    editor = "Demberg, Vera  and
      Inui, Kentaro  and
      Marquez, Llu{\'i}s",
    booktitle = "Findings of the {A}ssociation for {C}omputational {L}inguistics: {EACL} 2026",
    month = mar,
    year = "2026",
    address = "Rabat, Morocco",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2026.findings-eacl.314/",
    doi = "10.18653/v1/2026.findings-eacl.314",
    pages = "5985--6003",
    ISBN = "979-8-89176-386-9",
    abstract = "The current state of event detection research has two notable re-occurring limitations that we investigate in this study. First, the unidirectional nature of decoder-only LLMs presents a fundamental architectural bottleneck for natural language understanding tasks that depend on rich, bidirectional context. Second, we confront the conventional reliance on Micro-F1 scores in event detection literature, which systematically inflates performance by favoring majority classes. Instead, we focus on Macro-F1 as a more representative measure of a model{'}s ability across the long-tail of event types. Our experiments demonstrate that models enhanced with sentence context achieve superior performance over canonical decoder-only baselines. Using Low-Rank Adaptation (LoRA) during finetuning provides a substantial boost in Macro-F1 scores in particular, especially for the decoder-only models, showing that LoRA can be an effective tool to enhance LLMs' performance on long-tailed event classes."
}
```

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
