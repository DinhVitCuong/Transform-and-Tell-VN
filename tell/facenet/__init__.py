"""Wrapper to provide face detection and recognition models.

The original project included custom implementations of MTCNN and
InceptionResnetV1.  These are now provided by the ``facenet_pytorch``
package which is actively maintained.  Fallback to the local versions if the
library is not available.
"""

try:
    from facenet_pytorch import MTCNN, InceptionResnetV1
except Exception:  # pragma: no cover - only used when facenet_pytorch missing
    from .inception_resnet_v1 import InceptionResnetV1
    from .mtcnn import MTCNN
