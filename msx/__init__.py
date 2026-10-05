"""MSX - the MEDSCAN analyser engine.

A human-in-the-loop chest X-ray analyser built around *adaptive analysis
routing*: every scan is checked for quality, screened once, and only the
specialist modules the screen calls for are run. Every finding carries a
probability, a confidence, the evidence behind it and its limitations, and the
doctor accepts, corrects, rejects or questions it.

The stages, in order (see :mod:`msx.pipeline`):

1. :mod:`msx.imaging`     - load PNG/JPEG/DICOM, strip identifiers, normalise
2. :mod:`msx.quality`     - is this scan fit to analyse? if not, stop and ask a human
3. :mod:`msx.screening`   - fast whole-image screen (deep model or built-in engine)
4. :mod:`msx.router`      - decide which specialist modules this scan needs
5. :mod:`msx.specialists` - region-specific analysis: cardiac, pleural, parenchyma, nodule
6. :mod:`msx.uncertainty` - calibration, test-time augmentation, abstention
7. :mod:`msx.explain`     - evidence-grounded explanation at the right depth (RAG)
8. :mod:`msx.datastore` / :mod:`msx.audit` - doctor decisions, hash-chained trail

Nothing here is a medical device. It is a research prototype for TECHgium.
"""

from .version import __version__

__all__ = ["__version__"]
