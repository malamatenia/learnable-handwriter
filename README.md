

# An Interpretable Deep Learning Approach for Morphological Script Type Analysis <br><sub>An adaptation of the [Learnable Typewriter](https://github.com/ysig/learnable-typewriter) for Morphological Script Type Analysis</sub>
Github repository of the [An Interpretable Deep Learning Approach for Morphological Script Type Analysis](add arXiv version of the paper).  
Authors: Malamatenia Vlachou Efstathiou, [Yannis Siglidis](https://imagine.enpc.fr/~siglidii/), [Dominique Stutzmann](https://cv.hal.science/dominique-stutzmann), [Mathieu Aubry](http://imagine.enpc.fr/~aubrym/).  
Research Institute: [IRHT], (https://www.irht.cnrs.fr/), _Institut de Recherche et d'Histoire des Textes, CNRS_, [Imagine](https://imagine.enpc.fr/), _LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, Marne-la-Vallée, France_

![LTW_graph.png](./.media/LTW_graph.png)


### Datasets and Models :inbox_tray: for Southern and Northern _Textualis_ 📜
Download & extract [datasets.zip](https://www.dropbox.com/scl/fi/tfz79kwxoe4vp5e4npmxa/datasets.zip?rlkey=2820mu0bddpnax6alx04bglzu&st=caxfyfsp&dl=0) and [runs.zip](https://www.dropbox.com/scl/fi/4zc24m63hxhkh04y5xdi8/runs.zip?rlkey=6fr598xdiyh8a2yiiydxr7hw5&st=1svl5gpn&dl=0) in the parent folder.

### Inference 
For minimal inference from pre-trained and finetuned models and plotting, we provide a standalone notebook. 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11_CGvoXvpulKNEDsRN9MdBS35NvNz5l7?usp=sharing)



### Paper figures :bar_chart:
A notebook demo.ipynb is also provided to reproduce the paper results and graphs with adjustable code for custom data.

## Install :rocket:
```shell
conda create --name ltw pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda activate ltw
python -m pip install -r requirements.txt
```


### Custom Dataset :books:
Trying the Learnable Scriber on a new dataset, assuming that it consists of a parent dataset folder with or without subfolders per document : 

First create a first config file for the dataset:

```
configs/dataset/<DATASET_ID>.yaml
...

DATASET-TAG:
  path: <DATASET-NAME>/
  sep: ''                    # How the character separator is denoted in the annotation. 
  space: ' '                 # How the space is denoted in the annotation.
```

then a second one setting the hyperparameters: 

```
configs/<DATASET_ID>.yaml
...

For its structure, see the config file provided for our experiment with additional information as it's possible to exclude or restrict the train documents.

```

Then create the dataset folder:

```
datasets/<DATASET-NAME>
├── annotation.json
└── images
  ├── <image_id>.jpg or .png 
  └── ...
```


The annotation.json file should be a dictionary with entries of the form:
```
    "<image_id>": {
        "split": "train",                            # {"train", "val", "test"} - "val" is ignored in the unsupervised case.
        "label": "A beautiful calico cat."           # The text that corresponds to this line.
        "script": "Times_New_Roman"                  # Corresponds to the script type of the image (optional)
    },
```

You can completely ignore the annotation.json file in the case of unsupervised training without evaluation.

### Filter/Normalize transcriptions :soap:
We implement [choco-mufin](https://github.com/PonteIneptique/choco-mufin) when loading the dataset, using a disambiguation-table.csv to normalize or exclude characters from our annotations. This creates a consistent set of characters for analysis regardless of the annotation source. The current normalization suppresses allographs and edition signs (e.g., modern punctuation) for a graphetic approach. For more details see [the associated article](https://openhumanitiesdata.metajnl.com/articles/10.5334/johd.97). You can modify this per your needs.

Additionally, the data loader performs an [NFD normalization](https://fr.wikipedia.org/wiki/Normalisation_Unicode#NFD) when selecting the character vocabulary. This ensures that modifier characters, such as abbreviation tildes, are separated from the base letter and considered as separate characters when creating the prototypes.


## Training  :seedling: and Finetuning :herb:
Training and model configure is performed though hydra.
We supply the corresponding config files for our experiment.

```python
python scripts/train.py <CONFIG_NAME>.yaml
```

and finetune on a group/script level with:

```python

python scripts/finetune_scripts.py -i runs/<MODEL_PATH> -o <OUTPUT_PATH> --mode g_theta --max_steps <int> --invert_sprites --script '<SCRIPT_NAME>' -a <DATASET_PATH>/annotation.json -d <DATASET_PATH> --split <train or all>
```

and individual documents with: 

```python
python scripts/finetune_docs.py -i runs/<MODEL_PATH> -o <OUTPUT_PATH> --mode g_theta --max_steps <int> --invert_sprites -a <DATASET_PATH>/annotation.json -d <DATASET_PATH> --split <train or all>
```

> To all of the above experiment config files, additional command line overrides could be applied to further modify them using the [hydra syntax](https://hydra.cc/docs/advanced/override_grammar/basic/).


### Logging :chart_with_downwards_trend:
Logging is done through wandb; the link to visualization is provided directly upon training.

Alternatively, to visualize results with tensorboard run:

```bash
tensorboard --logdir ./<run_dir>/
```

### Citing :dizzy:

```bibtex
@misc{interpretable-script-analysis-iwcp2024,
	title = {An Interpretable Deep Learning Approach for Morphological Script Type Analysis},
	author = {Vlachou-Efstathiou, Malamatenia and Siglidis, Ioannis and Stutzmann, Dominique, and Aubry, Mathieu},
	publisher = {arXiv},
	year = {2024},
	url = {},
	keywords = {Computer Vision and Pattern Recognition (cs.CV), Digital Palaeography, Document Analysis},
	doi = {},
	copyright = {Creative Commons Attribution 4.0 International}
}
```

## Acknowledgements :sparkles:
This study was supported by the CNRS through MITI and the 80|Prime program (CrEMe Caractérisation des écritures médiévales) , and by the European Research Council (ERC project DISCOVER, number 101076028). We thank Ségolène Albouy, Raphaël Baena, Sonat Baltacı, Syrine Kalleli, and Elliot Vincent for valuable feedback on the paper.
