import sys
import collections
import collections.abc
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

if sys.version_info >= (3, 10):
    for _name in (
        "Callable", "Iterable", "Iterator", "Generator",
        "Mapping", "MutableMapping", "MutableSequence",
        "Sequence", "Set", "MutableSet",
    ):
        if not hasattr(collections, _name):
            setattr(collections, _name, getattr(collections.abc, _name))

import numpy as _np
for _alias, _builtin in (
    ("float", float), ("int", int), ("complex", complex),
    ("bool", bool), ("object", object), ("str", str),
):
    if not hasattr(_np, _alias):
        setattr(_np, _alias, _builtin)
del _np, _alias, _builtin
