from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Modellname auf Hugging Face
modell_name = "mistralai/Mistral-7B-Instruct-v0.2"

# Hinweis für den Nutzer
print("🔄 Lade Modell – das kann beim ersten Mal ein paar Minuten dauern...")

# Modell und Tokenizer laden
tokenizer = AutoTokenizer.from_pretrained(modell_name)
model = AutoModelForCausalLM.from_pretrained(modell_name)

# Textgenerierung vorbereiten
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Beispiel-Aufforderung
prompt = "Formuliere eine höfliche Bieterfrage zur Klärung unklarer Eignungskriterien."

# Text generieren
print("💬 Eingabe:", prompt)
antwort = generator(prompt, max_new_tokens=150, do_sample=True, temperature=0.7)[0]["generated_text"]

# Ergebnis anzeigen
print("\n🤖 Antwort der KI:\n")
print(antwort)

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

modell_name = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(modell_name)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype="float16",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

model = AutoModelForCausalLM.from_pretrained(
    modell_name,
    quantization_config=bnb_config,
    device_map="auto"
)
