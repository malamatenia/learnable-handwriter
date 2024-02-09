import json
import csv
from typing import List, Union
import os
from os import listdir
from os.path import join, dirname
from PIL import Image
from torch.utils.data.dataset import Dataset
from typing import Union
from torch.nn.functional import interpolate
from torchvision.transforms.functional import to_tensor

import unicodedata
import albumentations
from dataclasses import dataclass
import numpy as np


@dataclass
class UniDataset(Dataset): #inherits the torch Class 
    alias: str #these lines serve as __init__ thanks to @dataclass
    path: str
    height: str
    split: str
    crop_width: int = None
    # N_min: int = 0
    # W_max: int = float('inf')
    transcribe: dict = None
    sep: str = ''
    space: str = ' '
    starts_with: Union[str, List[str]] = None #when you want to append more than one doc
    exclude: str = None
    supervised: bool = False
    padding: Union[int, tuple] = None
    n_channels: int = 4
    p: float = 0.0
    script: str = None
    filter_by_name: str = None
    disambiguation_mapping = join(dirname(__file__), 'disambiguation_table.csv')

    def __post_init__(self):
        self.image_dirs = listdir(join(self.path, 'images'))
        self.channels = {3: 'RGB', 1: 'L', 4: 'RGBA'}[self.n_channels]
        self.load_disambiguation_mapping()
        self.extract_post_init()
        self.make_alphabet()
        self.transcribe_post_init()
        self.padding_post_init()
        self.split_it()
        self.count_character_occurrences()

    def disambiguate_line(self, v):
        v['label'] = self.disambiguate_label(v['label'])
        return v
            
    def extract_post_init(self):

        self.data = []
        annotation = {k: self.disambiguate_line(v) for k, v in self.annotation.items()}
        
        prefixes = self.starts_with 
        if isinstance(self.starts_with, str):
            prefixes = [prefixes]

        def starts_with_condition(key):
            return any(key.startswith(prefix) for prefix in prefixes) 

        for k, v in annotation.items():
            if self.script is not None and v['script'] != self.script:
                continue
            if self.filter_by_name is not None and self.filter_by_name not in k:
                continue
            if self.starts_with is None or starts_with_condition(k):
                self.data.append((k, self.split_combining_(v), v['split']))
            if self.exclude is not None and k.startswith(self.exclude):
                self.data.remove((k, self.split_combining_(v), v['split'])) #exclude selected data

    def load_disambiguation_mapping(self):
        with open(self.disambiguation_mapping, 'r') as file:
                reader = csv.DictReader(file)
                disambiguation_mapping = {row['char']: row['replacement'] for row in reader}

        self.disambiguation_mapping = disambiguation_mapping            
            
    def disambiguate_label(self, label):
        return ''.join([self.disambiguation_mapping.get(c, c) for c in unicodedata.normalize('NFC', label)])

    def split_combining_(self, v):
        label = self.process_transcription(v['label'])
        return [c for c in unicodedata.normalize('NFC', label)] #search why there is inconsistency in identifying combining characters 

    @property
    def indexes(self):
        mapping = {a: list(unicodedata.normalize('NFD', a)) for a in self.alphabet}
        unique_combining = list(sorted(set(v for vs in mapping.values() for v in vs)))
        indexes = {c: i for i, c in enumerate(unique_combining)}
        #print(indexes)
        return indexes

    @property
    def factoring(self,): #creates a matrix with the alphabet and the combining characters
        mapping = {a: list(unicodedata.normalize('NFD', a)) for a in self.alphabet}
        unique_combining = list(sorted(set(v for vs in mapping.values() for v in vs)))
        indexes = {c: i for i, c in enumerate(unique_combining)}
        factoring = np.zeros((len(self.alphabet), len(unique_combining)))
        for k, vs in mapping.items():
            for v in vs:
                factoring[self.matching[k], indexes[v]] = 1
        return factoring, unique_combining

    def count_character_occurrences(self):
        # Count occurrences of each character after disambiguation
        self.character_counts = {}

        for _, sentence in self.data:
            for char in sentence:
                for c in list(unicodedata.normalize('NFD', char)):
                    if c in self.character_counts:
                        self.character_counts[c] += 1
                    else:
                        self.character_counts[c] = 1

        # Create a new dictionary with keys as integers from matching and values as occurrences
        self.character_occurrences = {self.indexes[char]: count for char, count in self.character_counts.items()}
        #print("Character Counts:", sorted(self.character_counts.items(), key=lambda x: x[1], reverse=True))

    def make_alphabet(self):
        if self.transcribe is not None:
           self.alphabet = set(self.transcribe.values())
        else:
           self.alphabet = set(l for _, sentence , _ in self.data for l in sentence)
           
           #print("Alphabet:", self.alphabet)

    def transcribe_post_init(self):
        if self.transcribe is None:
            self.transcribe = dict(enumerate(sorted(self.alphabet))) #dictionary with keys as idx and values as sorted chars

        self.matching = {char: num for num, char in self.transcribe.items()} #puts chars in keys and num in index

    def split_it(self):
        base_path = os.path.join(self.path, 'images')  # Base path where all the folders are
        self.data = [(self.get_path(base_path, path), label) for (path, label, split) in self.data if split == self.split]

    def get_path(self, base_path, path):
        filename = os.path.basename(path)  # Get the filename from the provided path

        # Recursively search for the .png file in all subfolders of the base_path
        for root, dirs, files in os.walk(base_path):
            if filename in files:
                return os.path.join(root, filename)

        raise FileNotFoundError(f"File not found: {filename}")

    def padding_post_init(self,):
        if self.padding is not None:
            if isinstance(self.padding, tuple) and all(((not isinstance(p, int)) and p < 1) for p in self.padding):
                self.padding = tuple(int(p*255) for p in self.padding)
            elif self.padding < 1 and isinstance(self.padding, float):
                self.padding = int(self.padding * 255)

        self.build_transform()
        assert not (self.supervised and not self.has_labels), "If dataset is used in supervised mode it should contain labels."

    def build_transform(self): #optional, in case you want to apply transformations
        transform = []
        if self.cropped:
            transform.append(albumentations.RandomCrop((self.height, self.crop_width), pad_if_needed=True, fill=self.padding_value, padding_mode='constant'))
        if self.split == 'train' and self.p != 0:
            print(f'RandomBrightnessContrast with p={self.p}')
            transform.append(albumentations.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=self.p))
        self.transform = albumentations.Compose(transform)

    def __len__(self): #mandatory 
        return len(self.data)

    def is_combining(self, v):
        return unicodedata.category(v) in ['Mn', 'Lm']

    def convert_label(self, vs):
       return [self.matching[v] for v in vs] #gets every character from the NFC and converts it to integers

    def __getitem__(self, i):
        path, label = self.data[i]
        #print(path, self.channels)
        x = Image.open(path).convert(self.channels)        
        x = interpolate(to_tensor(x).unsqueeze(0), size=(self.height, int(self.height * x.size[0] / x.size[1])), mode='bilinear')
        x = (255*x.squeeze(0).permute(1, 2, 0).numpy()).astype(np.uint8)
        x = Image.fromarray(x, mode=self.channels)
        return x, self.convert_label(label) #function that gets the label and applies transformation (returns a tuple)

    @property
    def has_labels(self):
        return self.annotation is not None

    @property
    def annotation_path(self):
        return join(self.path, 'annotation.json')

    def process_transcription(self, raw_transcription):
        transcription = raw_transcription.replace(self.space, '')
        if self.sep != '':
            transcription = transcription.split(self.sep)
        return transcription

    @property
    def annotation(self):
        with open(self.annotation_path) as f:
            return json.load(f)
        
    @property
    def cropped(self):
        return self.crop_width is not None
