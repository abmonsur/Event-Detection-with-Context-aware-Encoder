
Hi! This is the repository for the EACL 2026 paper: [Event Detection with a Context-Aware Encoder and LoRA for Improved Performance on Long-Tailed Classes](https://aclanthology.org/2026.findings-eacl.314/)
## Requirements

Please make sure you have installed the packages in [./environment.yml](https://github.com/nju-websoft/SCR/blob/main/environment.yml).

## Datasets

We use the RAMS and MAVEN datasets for evaluation.

## Pretrained Models

Download pretrained language BERT model from [huggingface](https://huggingface.co/bert-base-uncased) and put it into the pretrain model directory. You can use any encoder model. For our Llama model, download and put a llama model in that directory.

## Training and Testing

To run this project, first install the dependencies from `environment.yml`, prepare the dataset files under `./data/{DATASET_NAME}/`, and place the pretrained encoder model under `./pretrain_model/`.

Then train and test the model with one of the dataset scripts:

```cmd
sh run_any.sh
sh run_maven.sh
```

Each script runs `main.py` with the matching config file in `./config/`. You can also run a config directly:

```cmd
python main.py --config ./config/{DATASET_NAME}.ini
```

You can modify the hyperparameters in the config file

Note that {DATASET_NAME} is one of the the dataset names include rams and maven. 

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
