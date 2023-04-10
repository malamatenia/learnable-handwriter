import json
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from typing import Union
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

    assert all(b is not None for b in base)
    return combining, base


@dataclass
class UniDataset(Dataset):
    alias: str
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
    starts_with: str = None
    split_combining: bool = False
    supervised: bool = False
    padding: Union[int, tuple] = None
    padding_value: Union[int, tuple] = None
    n_channels: int = 4

    def __post_init__(self):
        self.image_dirs = listdir(join(self.path, 'images'))
        self.channels = {1: 'RGB', 3: 'L', 4: 'RGBA'}[self.n_channels]
        self.extract_post_init()
        self.make_alphabet()
        self.transcribe_post_init()
        self.padding_post_init()
        self.split_it()

    def extract_post_init(self):
        self.data = []
        for k, v in self.annotation.items():
            if self.starts_with is None or k.startswith(self.starts_with):
                self.data.append((k, self.split_combining_(v), v['split']))

    def split_combining_(self, v):
        label = self.process_transcription(v['label'])
        if self.split_combining:
            combining, base = group_unicode_chars(unicodedata.normalize('NFD', label))
            return {'combining': combining, 'base': base}
        else:
            return {'base': unicodedata.normalize('NFC', label)}

    def make_alphabet(self):
        if self.transcribe is not None:
            if self.split_combining:
                self.alphabet = {k: set(v) for k, v in self.transcribe.items()}
            else:
                self.alphabet = set(self.transcribe.values())
        else:
            def get_set(k):
                return set(l for _, d, _ in self.data for sentence in d[k] for l in sentence)

            if self.split_combining:
                self.alphabet = {k: get_set(k) for k in ['base', 'combining']}
            else:
                self.alphabet = get_set('base')

    def transcribe_post_init(self):
        if self.split_combining:
            if self.transcribe is None:
                self.transcribe = {k: dict(enumerate(sorted(v))) for k, v in self.alphabet.items()}

            self.matching = {k: {num: char for num, char in v.items()} for k, v in self.transcribe.items()}
        else:
            if self.transcribe is None:
                self.transcribe = dict(enumerate(sorted(self.alphabet)))

            self.matching = {char: num for num, char in self.transcribe.items()}

    def get_path(self,path):
         for d in self.image_dirs:
            if path.startswith(d):
                return join(self.path, 'images', d, path)

    def split_it(self,):
        self.data = [(self.get_path(path), label) for (path, label, split) in self.data if split == self.split]

    def to_unicode_base(self, x):
        return [self.matching[b] for b in x['base']]

    def to_unicode_combining(self, x):
        output = []
        for b, c in zip(x['base'], x['combining']):
            if c is not None:
                c = self.matching['base'][b] + self.matching['combining'][c]
            else:
                c = self.matching['base'][b]
            output.append(c)
        return output

    def padding_post_init(self,):
        if self.padding_value is not None:
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

    def convert_label(self, vs):
        if self.split_combining:
            return {k: [self.matching[l] for l in v[k]] for k, v in vs.items()}
        else:
            return {'base': [self.matching[v] for v in vs['base']]}

    def __getitem__(self, i):
        path, label = self.data[i]
        x = Image.open(path).convert(self.channels)
        x = x.resize((int(self.height * x.size[0] / x.size[1]), self.height))
        x = self.transform(x)
        return x, self.convert_label(label)

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
