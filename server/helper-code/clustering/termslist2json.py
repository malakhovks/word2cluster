import json
import codecs

with codecs.open('./termslist.txt', encoding='UTF-8') as f:
    text = f.read()
words = text.split()

print(words)
jsonStr = json.dumps(words)

with open('./termslist.json', 'w', encoding='utf-8') as fout:
    json.dump(words, fout, ensure_ascii=False, indent=4)