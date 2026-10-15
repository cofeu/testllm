import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "data.jsonl"

TOPICS = [
    ("uyku", "Uyku düzeni, vücut ve beyin için çok önemlidir. Düzenli uyku hafızayı güçlendirir, stres seviyesini azaltır ve gündelik enerjiyi artırır."),
    ("spor", "Spor yapmak kalbi güçlendirir, kasları korur ve ruh halini iyileştirir. Kısa ama düzenli hareket bile enerji seviyesini artırır."),
    ("su", "Su içmek vücut ısısını düzenler, hücreleri besler ve sindirimi destekler. Günde yeterli su almak baş ağrısını ve yorgunluğu azaltabilir."),
    ("beslenme", "Dengeli beslenme enerji seviyesini korur, bağışıklığı güçlendirir ve uzun vadede sağlığı destekler. Meyve, sebze, protein ve su birlikte daha iyi bir düzen oluşturur."),
    ("çalışma", "Kısa ve net hedefler koymak verimi artırır. Çalışırken mola vermek dikkati korur ve tükenmeyi azaltır."),
    ("hava", "Hava durumu, günlük planları etkileyen önemli bir faktördür. Yağmurlu günlerde iç mekan etkinlikleri, güneşli günlerde dışarıda hareket daha verimli olur."),
    ("çevre", "Çevreye dikkat etmek doğaya saygı göstermek demektir. Atıkları ayrıştırmak, suyu tasarruflu kullanmak ve enerji tüketimini azaltmak çevre dostu bir yaşam için önemlidir."),
    ("eğitim", "Düzenli tekrar, öğrenmeyi pekiştirir. Kısa notlar ve örnek sorular çözmek bilgiyi kalıcı hale getirir."),
    ("aile", "Aile ile düzenli konuşmak güven hissi oluşturur. Gülümseyen ve samimi sohbetler ilişkiyi güçlendirir."),
    ("teknoloji", "Teknoloji hayatı kolaylaştırır ama dikkatli kullanmak gerekir. Ekran süresini kontrollü tutmak dinlenme ve uyku düzenini korur."),
]

QUESTION_TEMPLATES = [
    "{topic} hakkında kısa bilgi verir misin?",
    "{topic} neden önemlidir?",
    "{topic} ile ilgili pratik öneriler verir misin?",
    "{topic} için ne yapmalıyım?",
    "{topic} ile ilgili temel bilgileri özetler misin?",
]

ANSWER_TEMPLATES = [
    "Tabii. {summary}",
    "Kısaca özetlemek gerekirse: {summary}",
    "Önemli olan nokta şudur: {summary}",
    "Bunu birkaç cümleyle açıklayalım: {summary}",
    "Öncelikle şunu bilmek gerekir: {summary}",
]


def load_existing(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            turns = obj.get("turns", [])
            if turns:
                records.append(turns)
    return records


def build_dialogue(topic, summary):
    q = QUESTION_TEMPLATES[len(topic) % len(QUESTION_TEMPLATES)].format(topic=topic)
    a = ANSWER_TEMPLATES[len(summary) % len(ANSWER_TEMPLATES)].format(summary=summary)
    return {
        "turns": [
            [q, a],
            ["Bunu daha basit anlatır mısın?", "Evet. " + summary],
        ]
    }


def augment_file(path: Path, n=40):
    existing = load_existing(path)
    out = list(existing)
    seen = set()
    for topic, summary in TOPICS:
        for _ in range(max(2, n // len(TOPICS))):
            dlg = build_dialogue(topic, summary)
            key = json.dumps(dlg, ensure_ascii=False, sort_keys=True)
            if key not in seen:
                seen.add(key)
                out.append(dlg)
    if len(out) == len(existing):
        return 0

    with open(path, "a", encoding="utf-8") as f:
        for dlg in out[len(existing):]:
            f.write(json.dumps(dlg, ensure_ascii=False) + "\n")
    return len(out) - len(existing)


if __name__ == "__main__":
    added = augment_file(DATA_PATH, n=40)
    print(f"added {added} dialogue records to {DATA_PATH}")
