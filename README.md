# An Interpretable Deep Learning Approach <br> for Morphological Script Type Analysis 
<sub> An adaptation of the [Learnable Typewriter](https://github.com/ysig/learnable-typewriter)</sub>
Github repository of the [An Interpretable Deep Learning Approach for Morphological Script Type Analysis](https://imagine.enpc.fr/~m.vlachou-efstathiou/learnable-scriber/). 

![LTW_graph.png](./.media/LTW_graph.png)


## Install
```shell
conda create --name ltw pytorch==1.9.1 torchvision==0.10.1 cudatoolkit=11.3 -c pytorch -c conda-forge
conda activate ltw
python -m pip install -r requirements.txt
```

### Our Datasets and Models 
Download & extract [datasets.zip](https://www.dropbox.com/scl/fi/tfz79kwxoe4vp5e4npmxa/datasets.zip?rlkey=2820mu0bddpnax6alx04bglzu&st=caxfyfsp&dl=0) and [runs.zip](https://www.dropbox.com/scl/fi/4zc24m63hxhkh04y5xdi8/runs.zip?rlkey=6fr598xdiyh8a2yiiydxr7hw5&st=1svl5gpn&dl=0) in the parent folder.

For minimal inference from pre-trained and finetuned models and plotting, we provide a standalone notebook. 
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/11_CGvoXvpulKNEDsRN9MdBS35NvNz5l7?usp=sharing)

### Paper figures
A notebook demo.ipynb is also provided to reproduce the paper results and graphs with adjustable code for custom data.

## Try it yourself 

To test the Learnable Scriber on a new dataset: 

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

Note: To ensure a consistent set of characters regardless of the annotation source for our analysis, we implement internally [choco-mufin](https://github.com/PonteIneptique/choco-mufin), using a disambiguation-table.csv to normalize or exclude characters from the annotations. The current configuration suppresses allographs and edition signs (e.g., modern punctuation) for a graphetic result.


## How to run: 

### Training a model
```python
python scripts/train.py <CONFIG_NAME>.yaml
```

### Finetuning a model

- A group of documents defined by the "script" type:
```python

python scripts/finetune_scripts.py -i runs/<MODEL_PATH> -o <OUTPUT_PATH> --mode g_theta --max_steps <int> --invert_sprites --script '<SCRIPT_NAME>' -a <DATASET_PATH>/annotation.json -d <DATASET_PATH> --split <train or all>
```

- individual documents in a dataset with: 
```python
python scripts/finetune_docs.py -i runs/<MODEL_PATH> -o <OUTPUT_PATH> --mode g_theta --max_steps <int> --invert_sprites -a <DATASET_PATH>/annotation.json -d <DATASET_PATH> --split <train or all>
```


### Logging

To visualize results with tensorboard run:

```bash
tensorboard --logdir ./<run_dir>/
```

### Citing

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

## Acknowledgements
This study was supported by the CNRS through MITI and the 80|Prime program (CrEMe Caractérisation des écritures médiévales) , and by the European Research Council (ERC project DISCOVER, number 101076028). We thank Ségolène Albouy, Raphaël Baena, Sonat Baltacı, Syrine Kalleli, and Elliot Vincent for valuable feedback on the paper.
