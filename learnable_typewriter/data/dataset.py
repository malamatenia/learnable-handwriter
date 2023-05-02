import json
from os import listdir
from os.path import join
from PIL import Image
from torch.utils.data.dataset import Dataset
from typing import Union
from torchvision.transforms import Compose, RandomCrop

import unicodedata
from dataclasses import dataclass



@dataclass
class UniDataset(Dataset): #inherits the torch Class 
    alias: str #these lines serve as __init__ thanks to @dataclass
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
        return [c for c in unicodedata.normalize('NFC', label)] #search why there is inconsistency in identifying combining characters 

    @property
    def factoring(self,):
        mapping = {a: list(unicodedata.normalize('NFD', a)) for a in self.alphabet}
        unique_combining = set(v for v in mapping.values())
        indexes = {c: i for i, c in enumerate(sorted(unique_combining))}
        mapping = np.zeros((len(self.alphabet), len(unique_combining)))
        for k, vs in mapping.items():
            for v in vs:
                mapping[self.matching[k], indexes[v]] = 1
        return mapping

    def make_alphabet(self):
        if self.transcribe is not None:
            self.alphabet = set(self.transcribe.values())
        else:
            self.alphabet = set(l for _, sentence , _ in self.data for l in sentence)

    def transcribe_post_init(self):
        if self.transcribe is None:
            self.transcribe = dict(enumerate(sorted(self.alphabet))) #dictionary with keys as int and values as sorted chars

        self.matching = {char: num for num, char in self.transcribe.items()} #puts chars in keys and num in index

    def get_path(self,path):
         for d in self.image_dirs:
            if path.startswith(d):
                return join(self.path, 'images', d, path)

    def split_it(self,): #choses split and converts path into absolute path for the directory concerned
        self.data = [(self.get_path(path), label) for (path, label, split) in self.data if split == self.split]

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

    def __len__(self): #mandatory 
        return len(self.data)

    def is_combining(self, v):
        return unicodedata.category(v) in ['Mn', 'Lm']

    def convert_label(self, vs):
        return [self.matching[v] for v in vs] #gets every character from the NFC and converts it to integers

    def __getitem__(self, i): #mandatory/realises the indexing of the Class instance so we can iterate through
        path, label = self.data[i]
        x = Image.open(path).convert(self.channels)
        x = x.resize((int(self.height * x.size[0] / x.size[1]), self.height)) #keeps constant aspect ratio
        x = self.transform(x)
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
