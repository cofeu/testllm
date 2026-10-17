"""Muhakeme (reasoning) verisi ureteci.

Tohumlanmis, deterministik Turkce ornekler uretir: kelime problemleri (adim adim
cozumlu), yuzde/indirim/ortalama, birim cevirme, mantik, yapilandirilmis JSON
cikarimi, kod cikti tahmini ve belirsizlik/reddetme. Cikti `turns` formatinda
JSONL'dir:

    {"turns": [["kullanici sorusu", "Adim 1: ...\nSonuc: ..."]]}

Amac: modele adim adim dusunme (chain-of-thought), net sonuc yazma,
yapilandirilmis cikti ve "bilmiyorum/emin degilim" davranisini ogretmek.

Kullanim:
    python scripts/make_reasoning_data.py --out data/reasoning.jsonl --per-category 120
"""

import argparse
import json
import random
from pathlib import Path

WEEKDAYS = ["Pazartesi", "Salı", "Çarşamba", "Perşembe", "Cuma", "Cumartesi", "Pazar"]
NAMES = ["Ali", "Ayşe", "Mehmet", "Zeynep", "Can", "Elif", "Selin", "Burak", "Deniz", "Ece"]
OBJECTS = ["elma", "kitap", "kalem", "bilye", "kurabiye", "portakal", "çikolata"]


def _steps(*lines, sonuc=None, giris=None):
    out = []
    if giris:
        out.append(giris)
    for i, line in enumerate(lines, start=1):
        out.append(f"Adım {i}: {line}")
    if sonuc is not None:
        out.append(f"Sonuç: {sonuc}")
    return "\n".join(out)


# ---------------------------------------------------------------------------
# Kategoriler: her biri rng alir, (soru, cevap) doner.
# ---------------------------------------------------------------------------
def gen_add(r):
    a, b = r.randint(4, 80), r.randint(4, 80)
    name, obj = r.choice(NAMES), r.choice(OBJECTS)
    q = f"{name} {a} {obj} topladı, sonra {b} {obj} daha topladı. {name} toplam kaç {obj} toplamıştır?"
    a_ = _steps(
        f"Başlangıçtaki miktar {a} {obj}.",
        f"Eklenen miktar {b} {obj}.",
        f"Toplam = {a} + {b} = {a + b}.",
        sonuc=f"{a + b} {obj}",
    )
    return q, a_


def gen_sub(r):
    a = r.randint(30, 120)
    b = r.randint(3, a - 1)
    obj = r.choice(OBJECTS)
    q = f"Bir kutuda {a} {obj} vardı. {b} tanesi alındı. Kutuda kaç {obj} kaldı?"
    a_ = _steps(
        f"Başlangıç {a}.",
        f"Alınan {b}.",
        f"Kalan = {a} - {b} = {a - b}.",
        sonuc=f"{a - b} {obj}",
    )
    return q, a_


def gen_mul(r):
    a, b = r.randint(3, 12), r.randint(3, 12)
    obj = r.choice(OBJECTS)
    q = f"Her birinde {a} {obj} olan {b} kutu var. Toplam kaç {obj} vardır?"
    a_ = _steps(
        f"Bir kutuda {a} {obj} var.",
        f"Kutu sayısı {b}.",
        f"Toplam = {a} × {b} = {a * b}.",
        sonuc=f"{a * b} {obj}",
    )
    return q, a_


def gen_div(r):
    b = r.randint(3, 12)
    c = r.randint(3, 12)
    a = b * c
    obj = r.choice(OBJECTS)
    q = f"{a} {obj} {b} kişiye eşit paylaştırılıyor. Her kişiye kaç {obj} düşer?"
    a_ = _steps(
        f"Toplam {a} {obj}.",
        f"Kişi sayısı {b}.",
        f"Kişi başı = {a} ÷ {b} = {c}.",
        sonuc=f"{c} {obj}",
    )
    return q, a_


def gen_percent(r):
    base = r.choice([100, 200, 300, 400, 500, 800, 1000])
    pct = r.choice([5, 10, 15, 20, 25, 40, 50])
    extra = base * pct // 100
    q = f"Fiyatı {base} TL olan bir ürüne %{pct} zam yapıldı. Yeni fiyat kaç TL'dir?"
    a_ = _steps(
        f"Zam miktarı = {base} × {pct}/100 = {extra} TL.",
        f"Yeni fiyat = {base} + {extra} = {base + extra} TL.",
        sonuc=f"{base + extra} TL",
    )
    return q, a_


def gen_discount(r):
    base = r.choice([100, 200, 400, 500, 1000, 2000])
    pct = r.choice([10, 20, 25, 30, 50])
    cut = base * pct // 100
    q = f"{base} TL'lik ürüne %{pct} indirim yapıldı. İndirimli fiyat kaç TL'dir?"
    a_ = _steps(
        f"İndirim = {base} × {pct}/100 = {cut} TL.",
        f"Yeni fiyat = {base} - {cut} = {base - cut} TL.",
        sonuc=f"{base - cut} TL",
    )
    return q, a_


def gen_average(r):
    n = r.randint(3, 5)
    nums = [r.randint(1, 40) for _ in range(n)]
    total = sum(nums)
    mean = total / n
    mean_txt = str(int(mean)) if float(mean).is_integer() else f"{mean:.1f}".replace(".", ",")
    joined = ", ".join(str(x) for x in nums)
    q = f"{joined} sayılarının ortalaması kaçtır?"
    a_ = _steps(
        f"Toplam = {' + '.join(str(x) for x in nums)} = {total}.",
        f"Eleman sayısı {n}.",
        f"Ortalama = {total} ÷ {n} = {mean_txt}.",
        sonuc=mean_txt,
    )
    return q, a_


def gen_unit(r):
    kind = r.choice(["saat_dk", "dk_sn", "km_m", "kg_g", "l_ml", "gun_saat"])
    v = r.randint(2, 12)
    if kind == "saat_dk":
        q, unit, factor = f"{v} saat kaç dakikadır?", "dakika", 60
    elif kind == "dk_sn":
        q, unit, factor = f"{v} dakika kaç saniyedir?", "saniye", 60
    elif kind == "km_m":
        q, unit, factor = f"{v} kilometre kaç metredir?", "metre", 1000
    elif kind == "kg_g":
        q, unit, factor = f"{v} kilogram kaç gramdır?", "gram", 1000
    elif kind == "l_ml":
        q, unit, factor = f"{v} litre kaç mililitredir?", "mililitre", 1000
    else:
        q, unit, factor = f"{v} gün kaç saattir?", "saat", 24
    a_ = _steps(
        f"1 birim = {factor} {unit}.",
        f"{v} × {factor} = {v * factor}.",
        sonuc=f"{v * factor} {unit}",
    )
    return q, a_


def gen_age(r):
    a = r.randint(15, 40)
    d = r.randint(1, 10)
    q = f"Ali {a} yaşında. Kardeşi Ali'den {d} yaş küçük. İkisinin yaşları toplamı kaçtır?"
    a_ = _steps(
        f"Kardeşin yaşı = {a} - {d} = {a - d}.",
        f"Toplam = {a} + {a - d} = {2 * a - d}.",
        sonuc=f"{2 * a - d}",
    )
    return q, a_


def gen_shopping(r):
    price = r.choice([5, 10, 15, 20, 25])
    count = r.randint(2, 9)
    paid = r.choice([50, 100, 200])
    total = price * count
    if paid <= total:
        paid = total + r.choice([10, 20, 50])
    q = f"Tanesi {price} TL olan ürünlerden {count} adet alındı ve {paid} TL ödendi. Para üstü kaç TL'dir?"
    a_ = _steps(
        f"Toplam tutar = {price} × {count} = {total} TL.",
        f"Para üstü = {paid} - {total} = {paid - total} TL.",
        sonuc=f"{paid - total} TL",
    )
    return q, a_


def gen_speed(r):
    speed = r.choice([30, 40, 50, 60, 80, 100])
    hours = r.randint(2, 6)
    q = f"Saatte {speed} km hızla giden bir araç {hours} saatte kaç km yol alır?"
    a_ = _steps(
        f"Yol = hız × zaman.",
        f"Yol = {speed} × {hours} = {speed * hours}.",
        sonuc=f"{speed * hours} km",
    )
    return q, a_


def gen_interest(r):
    principal = r.choice([1000, 2000, 5000, 10000])
    rate = r.choice([5, 10, 20])
    years = r.randint(1, 4)
    interest = principal * rate * years // 100
    q = f"{principal} TL para yıllık %{rate} faizle {years} yıl bankada kalıyor. Toplam faiz kaç TL'dir?"
    a_ = _steps(
        f"Faiz = anapara × oran × süre / 100.",
        f"Faiz = {principal} × {rate} × {years} / 100 = {interest} TL.",
        sonuc=f"{interest} TL",
    )
    return q, a_


def gen_compare(r):
    a, b = r.randint(3, 20), r.randint(3, 20)
    ca, cb = r.randint(2, 6), r.randint(2, 6)
    va, vb = a * ca, b * cb
    first = "Ali" if va >= vb else "Veli"
    q = f"Ali {ca} kutuda {a} bilye, Veli {cb} kutuda {b} bilye topladı. Hangisi daha çok bilye topladı?"
    a_ = _steps(
        f"Ali'nin bilyeleri = {ca} × {a} = {va}.",
        f"Veli'nin bilyeleri = {cb} × {b} = {vb}.",
        f"{va} ile {vb} karşılaştırılır; büyük olan {max(va, vb)}.",
        sonuc=f"{first}, {max(va, vb)} bilye ile daha fazla topladı.",
    )
    return q, a_


def gen_sequence(r):
    start = r.randint(1, 9)
    step = r.randint(2, 7)
    seq = [start + step * i for i in range(4)]
    nxt = start + step * 4
    q = f"{', '.join(str(x) for x in seq)} dizisinde bir sonraki sayı kaçtır?"
    a_ = _steps(
        f"Ardışık terimler arasındaki fark sabit: {seq[1]} - {seq[0]} = {step}.",
        f"Son terim {seq[-1]}, kural gereği {step} eklenir.",
        f"{seq[-1]} + {step} = {nxt}.",
        sonuc=f"{nxt}",
    )
    return q, a_


def gen_logic(r):
    q = "Tüm kediler memelidir. Pamuk bir kedidir. Pamuk memeli midir?"
    a_ = _steps(
        "Kural: her kedi memelidir.",
        "Pamuk bir kedi olduğuna göre bu kural Pamuk için de geçerlidir.",
        sonuc="Evet, Pamuk memelidir.",
    )
    return q, a_


def gen_day(r):
    start = r.randrange(len(WEEKDAYS))
    offset = r.randint(1, 6)
    result = (start + offset) % 7
    q = f"Bugün {WEEKDAYS[start]}. {offset} gün sonra hangi gün olur?"
    a_ = _steps(
        f"Haftanın günleri sırayla: {', '.join(WEEKDAYS)}.",
        f"{WEEKDAYS[start]} gününden {offset} gün ileri sayılır.",
        f"{offset} gün sonra {WEEKDAYS[result]} gelir.",
        sonuc=WEEKDAYS[result],
    )
    return q, a_


def gen_json(r):
    name = r.choice(NAMES)
    age = r.randint(18, 65)
    city = r.choice(["İstanbul", "Ankara", "İzmir", "Bursa", "Antalya"])
    q = (
        "Aşağıdaki metinden ad, yaş ve şehir bilgisini JSON olarak çıkar:\n"
        f"\"Benim adım {name}, {age} yaşındayım ve {city}'da yaşıyorum.\""
    )
    a_ = "{\n" f'  "ad": "{name}",\n' f'  "yaş": {age},\n' f'  "şehir": "{city}"\n' "}"
    return q, a_


def gen_howto(r):
    q = "Çay demlemenin adımlarını sırala."
    a_ = (
        "1. Suyu kaynat.\n"
        "2. Demliğe birkaç kaşık çay koy.\n"
        "3. Kaynar suyu demliğe ekle.\n"
        "4. Demliği birkaç dakika beklet.\n"
        "5. Çayı bardaklara süzerek servis et."
    )
    return q, a_


def gen_code(r):
    lst = [r.randint(1, 9) for _ in range(3)]
    idx = r.randint(0, 2)
    q = f"Python kodunun çıktısı nedir?\nprint([{', '.join(str(x) for x in lst)}][{idx}])"
    a_ = _steps(
        f"Liste: [{', '.join(str(x) for x in lst)}].",
        f"İndeksler 0'dan başlar; {idx}. indeks {idx + 1}. elemandır.",
        f"Bu eleman {lst[idx]}.",
        sonuc=str(lst[idx]),
    )
    return q, a_


def gen_refusal(r):
    options = [
        ("Borsa yarın yükselir mi?", "Bunu güvenilir biçimde bilemem. Borsa fiyatları birçok etkene bağlıdır ve kesin tahmin mümkün değildir. Emin değilim."),
        ("2035'te dünya nüfusu tam olarak kaç olacak?", "Kesin bir sayı veremem; nüfus projeksiyonları varsayımlara göre değişir. Güvenilir tek bir değer yok."),
        ("Bu ilaç bana iyi gelir mi?", "Bir doktor olmadığım için bunu söyleyemem. Böyle bir karar için mutlaka bir hekime danışmalısın."),
        ("Geçen hafta ne yediğimi hatırlıyor musun?", "Hayır, önceki sohbetlerimizi hatırlamam. Bunu bilemem."),
        ("Yarın yağmur yağacak mı?", "Kesin bir tahmin veremem; en doğru bilgi için güncel hava durumuna bakmalısın."),
    ]
    return r.choice(options)


def gen_direct(r):
    pairs = [
        ("Türkiye'nin başkenti neresidir?", "Ankara"),
        ("Bir yılda kaç ay vardır?", "12 ay"),
        ("Bir haftada kaç gün vardır?", "7 gün"),
        ("Suyun kimyasal formülü nedir?", "H2O"),
        ("Dünyanın uydusu nedir?", "Ay"),
    ]
    return r.choice(pairs)


def gen_technique(r):
    q = "Bir problemi çözerken nasıl düşünürsün?"
    a_ = (
        "Adım 1: Soruyu dikkatle okur, neyin istendiğini belirlerim.\n"
        "Adım 2: Bilinenleri ve bilinmeyenleri ayırırım.\n"
        "Adım 3: Uygun yöntemi seçip küçük adımlara bölerim.\n"
        "Adım 4: Adımları sırayla uygularım.\n"
        "Adım 5: Sonucu kontrol ederim."
    )
    return q, a_


_DEN_WORD = {2: "iki", 3: "üç", 4: "dört", 5: "beş", 6: "altı", 8: "sekiz", 10: "on"}


def gen_fraction(r):
    den = r.choice(list(_DEN_WORD))
    base = den * r.randint(10, 90)
    num = r.randint(1, den - 1)
    part = base // den
    result = part * num
    q = f"{base} sayısı {_DEN_WORD[den]} eşit parçaya bölünüyor, sonra bir parça {num} ile çarpılıyor. Sonuç kaçtır?"
    a_ = _steps(
        f"Bir parça = {base} ÷ {den} = {part}.",
        f"Sonuç = {part} × {num} = {result}.",
        sonuc=f"{result}",
    )
    return q, a_


def gen_area(r):
    w, h = r.randint(3, 15), r.randint(3, 15)
    if r.random() < 0.5:
        q = f"Kenar uzunlukları {w} cm ve {h} cm olan dikdörtgenin alanı kaç cm²'dir?"
        a_ = _steps(
            f"Alan = uzunluk × genişlik.",
            f"Alan = {w} × {h} = {w * h}.",
            sonuc=f"{w * h} cm²",
        )
    else:
        q = f"Kenar uzunlukları {w} cm ve {h} cm olan dikdörtgenin çevresi kaç cm'dir?"
        a_ = _steps(
            f"Çevre = 2 × (uzunluk + genişlik).",
            f"Çevre = 2 × ({w} + {h}) = 2 × {w + h} = {2 * (w + h)}.",
            sonuc=f"{2 * (w + h)} cm",
        )
    return q, a_


def gen_clock(r):
    hour = r.randint(6, 20)
    minute = r.choice([0, 15, 30, 45])
    add = r.randint(1, 5)
    new_hour = (hour + add) % 24
    q = f"Saat {hour:02d}:{minute:02d} iken {add} saat sonra saat kaç olur?"
    a_ = _steps(
        f"Başlangıç saati {hour:02d}:{minute:02d}.",
        f"{add} saat eklenince saat {hour} + {add} = {hour + add}; 24'ü geçerse sıfırlanır.",
        f"Yeni saat {new_hour:02d}:{minute:02d}.",
        sonuc=f"{new_hour:02d}:{minute:02d}",
    )
    return q, a_


def gen_budget(r):
    income = r.choice([3000, 4000, 5000, 6000, 8000])
    rent = r.randint(500, 1600)
    food = r.randint(400, 1500)
    if rent + food >= income:
        rent = income // 4
        food = income // 5
    remaining = income - rent - food
    q = f"Aylık geliri {income} TL olan biri {rent} TL kira ve {food} TL gıda harcıyor. Geriye kaç TL kalır?"
    a_ = _steps(
        f"Toplam gider = {rent} + {food} = {rent + food} TL.",
        f"Kalan = {income} - {rent + food} = {remaining} TL.",
        sonuc=f"{remaining} TL",
    )
    return q, a_


def gen_ordering(r):
    nums = [r.randint(1, 99) for _ in range(5)]
    ordered = ", ".join(str(x) for x in sorted(nums))
    q = f"{', '.join(str(x) for x in nums)} sayılarını küçükten büyüğe sırala."
    a_ = _steps(
        "Sayılar birbiriyle karşılaştırılır.",
        "En küçükten en büyüğe doğru dizilir.",
        sonuc=ordered,
    )
    return q, a_


CATEGORIES = [
    gen_add, gen_sub, gen_mul, gen_div, gen_percent, gen_discount, gen_average,
    gen_unit, gen_age, gen_shopping, gen_speed, gen_interest, gen_compare,
    gen_sequence, gen_logic, gen_day, gen_json, gen_howto, gen_code,
    gen_refusal, gen_direct, gen_technique, gen_fraction, gen_area, gen_clock,
    gen_budget, gen_ordering,
]


def generate(per_category: int, seed: int):
    r = random.Random(seed)
    rows = []
    seen = set()
    for fn in CATEGORIES:
        made = 0
        attempts = 0
        while made < per_category and attempts < per_category * 20:
            attempts += 1
            q, a = fn(r)
            key = (q.strip().lower(), a.strip().lower())
            if key in seen:
                continue
            seen.add(key)
            rows.append({"turns": [[q, a]]})
            made += 1
    r.shuffle(rows)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="Muhakeme verisi uret")
    root = Path(__file__).resolve().parents[1]
    ap.add_argument("--out", default=str(root / "data" / "reasoning.jsonl"))
    ap.add_argument("--per-category", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(argv)

    rows = generate(args.per_category, args.seed)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"{len(rows)} ornek yazildi: {out}")


if __name__ == "__main__":
    main()
