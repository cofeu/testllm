"""Kanonik giris noktasi:  python -m src.py.normal

Egitim CLI'sine (train.main) delege eder; boylece tek bir akis, tum veri
dosyalari ve mod secenekleri kullanilir. Eski sabit kodlu akis (chatml dosyasini
her calismada yeniden yazip generate/verify yapan) kaldirildi.
"""

from .train import main as _train_main


def main(argv=None):
    return _train_main(argv)


if __name__ == "__main__":
    main()
