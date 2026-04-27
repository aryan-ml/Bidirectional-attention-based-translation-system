import pandas as pd
import re
import random
from indic_transliteration.sanscript import transliterate, DEVANAGARI, ITRANS
from tqdm import tqdm

# -----------------------------
# STEP 1: Hindi sentence seeds
# -----------------------------
hindi_sentences = [
    "मुझे बाहर जाना है",
    "तुम क्या कर रहे हो",
    "आज मौसम अच्छा है",
    "मुझे भूख लगी है",
    "क्या तुम मेरे साथ चलोगे",
    "यह बहुत मुश्किल है",
    "मैं थक गया हूँ",
    "तुम कहाँ हो",
    "जल्दी आओ",
    "मुझे पानी चाहिए"
]

# Expand this list manually or using AI later

# -----------------------------
# STEP 2: Transliteration
# -----------------------------
def hindi_to_hinglish(text):
    mapping = {
        "मुझे": "mujhe",
        "पानी": "pani",
        "चाहिए": "chahiye",
        "मैं": "main",
        "थक": "thak",
        "गया": "gaya",
        "हूँ": "hu",
        "आज": "aaj",
        "मौसम": "mausam",
        "अच्छा": "accha",
        "है": "hai",
        "तुम": "tum",
        "कहाँ": "kahan",
        "जल्दी": "jaldi",
        "आओ": "aao",
        "बाहर": "bahar",
        "जाना": "jana"
    }

    words = text.split()
    output = []

    for w in words:
        if w in mapping:
            output.append(mapping[w])
        else:
            # fallback transliteration
            temp = transliterate(w, DEVANAGARI, ITRANS)
            temp = temp.lower().replace("aa", "a").replace(".", "")
            output.append(temp)

    return " ".join(output)

# -----------------------------
# STEP 3: Add Hinglish noise
# -----------------------------
def add_noise(text):
    variations = [
        text,
        text.replace("hai", "h"),
        text.replace("chahiye", "chahiye yaar"),
        text.replace("mujhe", "muje"),
        text + " bro",
        text + " yaar"
    ]
    return random.choice(variations)

def is_pure_hinglish(text):
    # Only allow english letters + spaces
    return re.match(r'^[a-zA-Z\s]+$', text) is not None




# -----------------------------
# STEP 4: Build Hindi-Hinglish
# -----------------------------
hh_data = []

for _ in tqdm(range(7000)):
    sent = random.choice(hindi_sentences)
    hinglish = hindi_to_hinglish(sent)
    hinglish = add_noise(hinglish)

    hh_data.append({
        "input": sent,
        "target": hinglish,
        "type": "hi_to_hinglish"
    })

# -----------------------------
# STEP 5: English → Hinglish
# -----------------------------
eng_to_hinglish = [
    ("I am hungry", "mujhe bhook lagi hai"),
    ("Where are you", "tum kahan ho"),
    ("Come fast", "jaldi aao"),
    ("I need water", "mujhe pani chahiye"),
    ("This is difficult", "yeh mushkil hai")
]

eh_data = []

for _ in tqdm(range(3000)):
    eng, hing = random.choice(eng_to_hinglish)

    eh_data.append({
        "input": eng,
        "target": hing,
        "type": "en_to_hinglish"
    })


# -----------------------------
# FINAL DATASET
# -----------------------------
df = pd.DataFrame(hh_data + eh_data)

# Keep only valid Hinglish rows
df = df[df["target"].apply(is_pure_hinglish)]

df.to_csv("data/raw/dataset.csv", index=False)

print("Final dataset size:", len(df))