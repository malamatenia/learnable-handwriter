import json
import random
from os import listdir
from os.path import join, isfile
from functools import partial
from PIL import Image

import numpy as np
import cv2
from skimage import exposure
from torch.utils.data.dataset import Dataset
from torchvision.transforms import Compose, RandomCrop

import unicodedata
from dataclasses import dataclass


def group_unicode_chars(text):
    combining, base = [], []
    current_char = {'combining': None, 'base': None}

    for c in unicodedata.normalize('NFD', text):
        if unicodedata.category(c) in ['Mn', 'Mc', 'Lm']:
            if current_char['base'] is None:
                current_char['base'] = ' '
            if current_char['combining'] is None:
                current_char['combining'] = c
        else:
            if current_char['base'] is not None:
                combining.append(current_char['combining'])
                base.append(current_char['base'])
            current_char = {'combining': None, 'base': c}

    if current_char['base'] is not None:
        combining.append(current_char['combining'])
        base.append(current_char['base'])

    return combining, base


@dataclass
class UniDataset(Dataset):
    alias: str
    starts_with: str = None
    split_combining: bool = False
    path: str
    height: str
    split: str
    alias: str
    crop_width: int = None
    # N_min: int = 0
    # W_max: int = float('inf')
    transcribe: dict = None
    sep: str = ''
    space: str = ' '
    supervised: bool = False
    padding: Optional[int, tuple] = None
    padding_value: Optional[int, tuple] = None
    n_channels: int = 4

    def __post_init__(self):
        self.image_dirs = os.listdir(os.path.join(path, 'images'))
        self.channels = {1: 'RGB', 3: 'L', 4: 'RGBA'}[self.n_channels]
        self.make_alphabet()
        self.extract_post_init()
        self.transcribe_post_init()
        self.padding_post_init()

    def extract_post_init(self):
        self.data = []
        for k, v in self.annotation.items():
            self.data.append((k, self.split_combining_(v), v['split']))

    def split_combining_(self, v):
        label = self.process_transcription(v['label'])
        if self.split_combining:
            combining, base = group_unicode_chars(unicodedata.normalize('NFD', label))
            return {'combining': combining, 'base': base}
        else:
            return {'base': group_unicode_chars(unicodedata.normalize('NFC', label))}

    def make_alphabet(self):
        if self.transcribe is not None:
            if self.split_combining:
                self.alphabet = {k: set(v) for k, v in self.transcribe.items()}
            else:
                self.alphabet = set(transcribe.values())
        else:
            def get_set(k):
                return set(l for _, d, _ in self.data for l in d[k])

            if self.split_combining:
                self.alphabet = {k: get_set(k) for k in ['base', 'combining']}
            else:
                self.alphabet = get_set('base')

    def transcribe_post_init(self):
        if self.split_combining:
            if transcribe is None:
                self.transcribe = {k: dict(enumerate(sorted(v))) for k, v in self.alphabet.items()}

            self.matching = {k: {num: char for num, char in v.items()} for k, v in self.transcribe.items()}
        else:
            if transcribe is None:
                self.transcribe = dict(enumerate(sorted(self.alphabet)))

            self.matching = {char: num for num, char in self.transcribe.items()}

    def find_dir(self,path):
         for d in self.paths:
            if path.startswith(d):
                return os.path.join(self.path, 'images', d, path)

    def split_it(self,):
        self.data = [(self.get_path(path), label) for (path, label, split) in self.data if split == self.split]

    def to_unicode_base(self, x):
        return [self.matching[b] for b in x['base']]

    def to_unicode_combining(self, x):
        output = []
        for b, c in zip(x['base'], x['combining']):
            if c is not None:
                c = combine(self.matching['base'][b], self.matching['combining'][c])
            else:
                c = self.matching['base'][b]
            output.append(c)
        return output

    def padding_post_init(self, padding_value)
        if isinstance(self.padding_value, tuple) and all(((not isinstance(p, int)) and p < 1) for p in self.padding_value):
            self.padding_value = tuple(int(p*255) for p in self.padding_value)
        elif self.padding_value < 1 and isinstance(self.padding_value, float):
            self.padding_value = int(self.padding_value*255)

        self.build_transform()
        assert not (self.supervised and not self.has_labels), "If dataset is used in supervised mode it should contain labels."

    def build_transform(self):
        transform = []
        if self.cropped:
            transform.append(RandomCrop((self.height, self.crop_width), pad_if_needed=True, fill=self.padding_value, padding_mode='constant'))
        self.transform = Compose(transform)

    def __len__(self):
        return len(self.data)

    def transcribe_(self, vs):
        if self.split_combining:
            return {k: [self.transcribe[l] for l in v[k]] for k, v in vs.items()}
        else:
            return {'base': [self.transcribe[l] for v in vs['base']]}

    def __getitem__(self, i):
        path, label = self.data[i]
        x = Image.open(path).convert(self.channels)
        x = x.resize((int(self.height * x.size[0] / x.size[1]), self.height))
        x = self.transform(x)
        return x, self.transcribe_(label)

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
