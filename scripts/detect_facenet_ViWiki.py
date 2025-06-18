'''
Detect faces from local images listed in splits.json and enrich each record with facenet embeddings.

Expected splits.json structure (list of objects):
[
  {
    "_id": "sample123",        # unique identifier, used to name image file sample123.jpg
    // other metadata fields as needed
  },
  ...
]

Each image should be located at `<image_dir>/<_id>.jpg`.
Results are saved back to JSON with a new `facenet_details` key on each sample:
{
  "n_faces": int,
  "embeddings": [[...], ...],  # list of face embeddings (floats)
  "detect_probs": [...]         # list of detection probabilities
}
'''
import os
import json
import logging
from docopt import docopt
from tqdm import tqdm
from PIL import Image
import torch
from tell.facenet import MTCNN, InceptionResnetV1

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

USAGE = '''
Usage:
    detect_facenet_json.py --splits PATH [--image-dir DIR] [--face-dir DIR] [--output PATH]

Options:
    --splits PATH     Path to splits.json file.
    --image-dir DIR   Directory containing images named <_id>.jpg [default: ./data/images].
    --face-dir DIR    Directory to save face crops [default: ./data/faces].
    --output PATH     Path to write updated JSON [default: ./data/splits_with_faces.json].
'''

def main():
    args = docopt(USAGE)
    splits_path = args['--splits']
    image_dir = args['--image-dir']
    face_dir = args['--face-dir']
    output_path = args['--output']

    os.makedirs(face_dir, exist_ok=True)

    # Load splits.json
    with open(splits_path, 'r', encoding='utf-8') as f:
        samples = json.load(f)

    # Initialize MTCNN & Resnet
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f'Using device: {device}')
    mtcnn = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

    for sample in tqdm(samples, desc='Processing samples'):
        sid = sample.get('_id')
        if not sid:
            logger.warning('Skipping entry without `_id`')
            continue
        # Skip already-processed
        if 'facenet_details' in sample:
            continue

        img_path = os.path.join(image_dir, f"{sid}.jpg")
        if not os.path.exists(img_path):
            logger.warning(f'Image not found: {img_path}')
            continue

        try:
            img = Image.open(img_path).convert('RGB')
        except OSError:
            logger.warning(f'Cannot open image: {img_path}')
            continue

        # Detect faces and get probabilities
        with torch.no_grad():
            try:
                faces, probs = mtcnn(img, return_prob=True)
            except Exception as e:
                logger.warning(f'Face detection error on {sid}: {e}')
                continue

        if faces is None or len(faces) == 0:
            logger.info(f'No faces for sample {sid}')
            sample['facenet_details'] = { 'n_faces': 0, 'embeddings': [], 'detect_probs': [] }
            continue

        # Limit to first 10 faces
        faces = faces[:10].to(device)
        probs = probs[:10]

        # Compute embeddings
        embeddings, _ = resnet(faces)
        embeddings = embeddings.cpu().tolist()
        probs = probs.tolist()

        # Save face crops optionally
        for i, face in enumerate(faces.cpu()):
            crop_path = os.path.join(face_dir, f"{sid}_face{i:02}.jpg")
            face_permuted = face.permute(1, 2, 0).numpy()
            Image.fromarray((face_permuted * 255).astype('uint8')).save(crop_path)

        # Attach details
        sample['facenet_details'] = {
            'n_faces': len(embeddings),
            'embeddings': embeddings,
            'detect_probs': probs,
        }

    # Write updated JSON
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(samples, f, ensure_ascii=False, indent=2)

    logger.info(f'Wrote updated splits to {output_path}')

if __name__ == '__main__':
    main()
