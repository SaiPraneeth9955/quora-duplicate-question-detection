"""Model implementations.

Every model exposes ``score(questions1, questions2) -> np.ndarray`` on raw
question text, where a higher score means the pair is more likely a duplicate,
plus ``save(directory)`` and ``load(directory)``.
"""

from qqp.models.sbert import SbertSimilarity
from qqp.models.siamese import SiameseClassifier
from qqp.models.tfidf import TfidfSimilarity

__all__ = ["SbertSimilarity", "SiameseClassifier", "TfidfSimilarity"]
