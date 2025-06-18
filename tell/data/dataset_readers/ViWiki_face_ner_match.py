import logging
import os
import random
import json
from typing import Dict, List, Any

import numpy as np
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import ArrayField, MetadataField, TextField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer
from allennlp.data.tokenizers import Tokenizer
from overrides import overrides
from PIL import Image
from torchvision.transforms import Compose, Normalize, ToTensor
from tell.data.fields import ImageField, ListTextField
from tqdm import tqdm

logger = logging.getLogger(__name__)


@DatasetReader.register('ViWiki_face_ner')
class ViWikiFaceNERJSONReader(DatasetReader):
    """
    Read from the ViWiki dataset stored as JSON files.

    Expects these files under `json_dir`:
      • splits.json
      • articles.json
    Optionally (if use_objects=True):
      • objects.json

    And images in `image_dir` named `<sample_id>.jpg`.
    """

    def __init__(
        self,
        tokenizer: Tokenizer,
        token_indexers: Dict[str, TokenIndexer],
        image_dir: str,
        json_dir: str,
        eval_limit: int = 1000,
        use_caption_names: bool = True,
        use_objects: bool = False,
        n_faces: int = None,
        lazy: bool = True,
    ) -> None:
        super().__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers
        self.image_dir = image_dir

        # Load splits
        splits_path = os.path.join(json_dir, 'splits.json')
        with open(splits_path, 'r', encoding='utf-8') as f:
            self.splits: List[Dict[str, Any]] = json.load(f)

        # Load articles and index by _id for fast lookup
        articles_path = os.path.join(json_dir, 'articles.json')
        with open(articles_path, 'r', encoding='utf-8') as f:
            arts = json.load(f)
        self.articles: Dict[str, Dict[str, Any]] = {a['_id']: a for a in arts}

        # Optionally load object features
        self.use_objects = use_objects
        if use_objects:
            objects_path = os.path.join(json_dir, 'objects.json')
            with open(objects_path, 'r', encoding='utf-8') as f:
                objs = json.load(f)
            self.objects: Dict[str, Dict[str, Any]] = {o['_id']: o for o in objs}

        self.preprocess = Compose([
            # Resize(256), CenterCrop(224),
            ToTensor(),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        
        self.eval_limit = eval_limit
        self.use_caption_names = use_caption_names
        self.n_faces = n_faces
        random.seed(1234)
        self.rs = np.random.RandomState(1234)

    @overrides
    def _read(self, split: str):
        if split not in ['train', 'val', 'test']:
            raise ValueError(f'Unknown split: {split}')

        # Filter samples for this split
        samples = [s for s in self.splits if s['split'] == split]
        if split == 'val':
            samples = samples[: self.eval_limit]

        # Randomize order
        self.rs.shuffle(samples)

        for sample in tqdm(samples):
            ###### RETRIEVE ARTICLE AND IMAGEPATH ##########
            # Lookup the corresponding article
            article = self.articles[sample['article_id']]

            # Load the image file
            image_path = os.path.join(self.image_dir, f"{sample['_id']}.jpg")
            ############ DONE -> CLEER THIS ################
            try:
                image = Image.open(image_path)
            except (FileNotFoundError, OSError):
                continue

            # Named entities from context
            
            ###### CODE get named_entities ##########
            named_entities = sorted({
                ner['text']
                for ner in article.get('context_ner', [])
                if ner['label'] in ('PERSON', 'ORG', 'GPE')
            })
            ############ DONE -> CLEER THIS ################

            # Determine how many faces to use
            if self.n_faces is not None:
                n_persons = self.n_faces
            elif self.use_caption_names:
                ###### CODE get persons ##########
                # pick from caption_ner for this image index
                ners = article.get('caption_ner', [])
                persons = {
                    ner['text']
                    for ner in ners[sample['image_index']]
                    if ner['label'] == 'PERSON'
                }
                n_persons = len(persons)
            ############ DONE -> CLEER THIS ################
            else:
                n_persons = 4

            # Face embeddings
            facenet = sample.get('facenet_details', {})
            face_embeds = np.array(facenet.get('embeddings', [])[:n_persons]) \
                          if facenet else np.array([[]])

            # Object features (optional)
            obj_feats = None
            if self.use_objects:
                obj = self.objects.get(sample['_id'], {})
                feats = obj.get('object_features', [])
                obj_feats = np.array(feats) if feats else np.array([[]])

            yield self.article_to_instance(
                article, named_entities, face_embeds,
                image, sample['image_index'], image_path, obj_feats
            )

    def article_to_instance(
        self, article, named_entities, face_embeds, image,
        image_index, image_path, obj_feats
    ) -> Instance:
        # limit context length
        context = ' '.join(article['context'].split()[:500])
        caption = article['images'][image_index].strip()

        # tokenize
        context_tokens = self._tokenizer.tokenize(context)
        caption_tokens = self._tokenizer.tokenize(caption)

        # names field
        name_token_list = [self._tokenizer.tokenize(n) for n in named_entities]
        if name_token_list:
            name_field = [TextField(ts, self._token_indexers)
                          for ts in name_token_list]
        else:
            # fallback to caption tokens
            stub = ListTextField([TextField(caption_tokens, self._token_indexers)])
            name_field = stub.empty_field()

        fields = {
            'context': TextField(context_tokens, self._token_indexers),
            'names': ListTextField(name_field),
            'image': ImageField(image, self.preprocess),
            'caption': TextField(caption_tokens, self._token_indexers),
            'face_embeds': ArrayField(face_embeds, padding_value=np.nan),
        }
        if obj_feats is not None:
            fields['obj_embeds'] = ArrayField(obj_feats, padding_value=np.nan)
        fields['metadata'] = MetadataField({
            'context': context,
            'caption': caption,
            'names': named_entities,
            'web_url': article['web_url'],
            'image_path': image_path
        })
        return Instance(fields)
"""
// splits.json: list of samples
[
  {
    "_id": "sample123",
    "article_id": "article456",
    "split": "train",            // "train" | "val" | "test"
    "image_index": 0,            // index into article.images
    "facenet_details": {         // optional
      "embeddings": [
        [0.123, 0.456, ...],     // one face feature vector
        ...
      ]
    }
    // if use_objects=True, you can either include object_features here:
    "object_features": [
      [0.789, 0.012, ...],       // one object feature vector
      ...
    ]
  },
  ...
]
// articles.json: list of articles
[
  {
    "_id": "article456",
    "context": "Full article text ...",
    "images": [
      "Caption for image #0",
      "Caption for image #1",
      ...
    ],
    "web_url": "https://news.example.com/...",
    "caption_ner": [              // one list per image
      [
        { "label": "PERSON", "text": "Alice", /* ... other fields ... */ },
        ...
      ],
      ...
    ],
    "context_ner": [              // named entities in the context
      { "label": "ORG", "text": "Company X", /* ... */ },
      ...
    ]
  },
  ...
]
// objects.json (optional, if use_objects=True):
[
  {
    "_id": "sample123",            // must match a splits.json _id
    "object_features": [
      [0.111, 0.222, ...],
      ...
    ]
  },
  ...
]
"""
