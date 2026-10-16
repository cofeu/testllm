"""Geriye donuk uyumluluk shim'i.

Kanonik implementasyon artik `src/py/normal/` paketinde yasar (config,
tokenizer, prompt, model, corpus, dataset, train, generate). Bu dosya eski
import yollarini (`from src.pt.decoder_model import ...`) korumak icin tum
genel sembolleri yeniden disa aktarir.

Calistirma:
    python -m src.pt.decoder_model   # -> src.py.normal.__main__:main
"""

from src.py.normal import *  # noqa: F401,F403
from src.py.normal import __all__ as _normal_all

__all__ = list(_normal_all)


if __name__ == "__main__":
    from src.py.normal.__main__ import main

    main()
