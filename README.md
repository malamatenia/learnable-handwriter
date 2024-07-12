# <p align="center">[An Interpretable Deep Learning Approach for Morphological Script Type Analysis](https://imagine.enpc.fr/~m.vlachou-efstathiou/learnable-scriber/) <p> <p align="center"><sub> An adaptation of the Learnable Typewriter</sub> </p>
Authors: Malamatenia Vlachou Efstathiou, [Ioannis Siglidis](https://imagine.enpc.fr/~siglidii/), [Dominique Stutzmann](https://www.irht.cnrs.fr/fr/annuaire/stutzmann-dominique) and [Mathieu Aubry](https://imagine.enpc.fr/~aubrym/)
Research Institutes: [Imagine](https://imagine-lab.enpc.fr/), LIGM, Ecole des Ponts, Univ Gustave Eiffel, CNRS, and [IRHT](https://www.irht.cnrs.fr/fr/recherche/les-themes-et-sections/paleographie-latine), CNRS, Paris, Île-de-France, France

![LTW_graph.png](./.media/Derolez_table.png)


- For minimal inference on pre-trained and finetuned models without having to install, we provide a [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11_CGvoXvpulKNEDsRN9MdBS35NvNz5l7?usp=sharing) notebook, available also as [inference.ipynb](https://github.com/malamatenia/learnable-scriber/blob/8aba90fd46f350d3c0ebfd14b4c7229428efb56c/inference.ipynb).

- A [demo.ipynb](https://github.com/malamatenia/learnable-scriber/blob/580ec0695ab427601e2b7808d8901ca5e6dbd740/demo.ipynb) notebook is provided to reproduce the paper results and graphs. You'll need to download & extract [datasets.zip](https://www.dropbox.com/scl/fi/tfz79kwxoe4vp5e4npmxa/datasets.zip?rlkey=2820mu0bddpnax6alx04bglzu&st=caxfyfsp&dl=0) and [runs.zip](https://www.dropbox.com/scl/fi/4zc24m63hxhkh04y5xdi8/runs.zip?rlkey=6fr598xdiyh8a2yiiydxr7hw5&st=1svl5gpn&dl=0) in the base folder first.

## Install
```shell
conda create --name ltw pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda activate ltw
python -m pip install -r requirements.txt
```

## Run it from scratch on our dataset 

In this case you'll need to download & extract only the [datasets.zip](https://www.dropbox.com/scl/fi/tfz79kwxoe4vp5e4npmxa/datasets.zip?rlkey=2820mu0bddpnax6alx04bglzu&st=caxfyfsp&dl=0).

### Train our reference model with:
```python
 python scripts/train.py iwcp_south_north.yaml 
```

### Finetune 

1. Our Northern and Southern _Textualis_ models with: 
```python
python scripts/finetune_scripts.py -i runs/iwcp_south_north/train/ -o runs/iwcp_south_north/finetune/ --mode g_theta --max_steps 2500 --invert_sprites --script Northern_Textualis Southern Textualis -a datasets/iwcp_south_north/annotation.json -d datasets/iwcp_south_north/ --split train
```

2. Our document models with: 
```python
python scripts/finetune_docs.py -i runs/iwcp_south_north/train/ -o runs/iwcp_south_north/finetune/ --mode g_theta --max_steps 2500 --invert_sprites -a datasets/iwcp_south_north/annotation.json -d datasets/iwcp_south_north/ --split all
```

## Logging

To visualize results with tensorboard run:

```bash
tensorboard --logdir ./<run_dir>/
```

## Run it on your data 

1. Create a config file for the dataset:
```
configs/dataset/<DATASET_ID>.yaml
...

DATASET-TAG:                 
  path: <DATASET-NAME>/      
  sep: ''                    # How the character separator is denoted in the annotation. 
  space: ' '                 # How the space is denoted in the annotation.
```

2. then a second one setting the hyperparameters: 
```
configs/<DATASET_ID>.yaml
...

For its structure, see the config file provided for our experiment.

```

3. Create the dataset folder:
```
datasets/<DATASET-NAME>
├── annotation.json
└── images
  ├── <image_id>.png 
  └── ...
```


The annotation.json file should be a dictionary with entries of the form:
```
    "<image_id>": {
        "split": "train",                            # {"train", "val", "test"} - "val" is ignored in the unsupervised case.
        "label": "A beautiful calico cat."           # The text that corresponds to this line.
        "script": "Times_New_Roman"                  # (optional) Corresponds to the script type of the image
    },
```

You can completely ignore the annotation.json file in the case of unsupervised training without evaluation.

4. Train with:
   
   ```python
python scripts/train.py <CONFIG_NAME>.yaml
```
   
6. Finetune
 - On a group of documents defined by their "script" type with:
```python

python scripts/finetune_scripts.py -i runs/<MODEL_PATH> -o <OUTPUT_PATH> --mode g_theta --max_steps <int> --invert_sprites --script '<SCRIPT_NAME>' -a <DATASET_PATH>/annotation.json -d <DATASET_PATH> --split <train or all>
```
- On individual documents with:
  ```python
python scripts/finetune_docs.py -i runs/<MODEL_PATH> -o <OUTPUT_PATH> --mode g_theta --max_steps <int> --invert_sprites -a <DATASET_PATH>/annotation.json -d <DATASET_PATH> --split <train or all>
``` 
   
> [!NOTE]
> To ensure a consistent set of characters regardless of the annotation source for our analysis, we implement internally [choco-mufin](https://github.com/PonteIneptique/choco-mufin), using a disambiguation-table.csv to normalize or exclude characters from the annotations. The current configuration suppresses allographs and edition signs (e.g., modern punctuation) for a graphetic result.

### Cite us

```bibtex
@misc{vlachou2024interpretable,
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

Check out also: [Siglidis, I., Gonthier, N., Gaubil, J., Monnier, T., & Aubry, M. (2023). The Learnable Typewriter: A Generative Approach to Text Analysis.](https://imagine.enpc.fr/~siglidii/learnable-typewriter/)


## Acknowledgements
This study was supported by the CNRS through MITI and the 80|Prime program (CrEMe Caractérisation des écritures médiévales) , and by the European Research Council (ERC project DISCOVER, number 101076028). We thank Ségolène Albouy, Raphaël Baena, Sonat Baltacı, Syrine Kalleli, and Elliot Vincent for valuable feedback on the paper.
