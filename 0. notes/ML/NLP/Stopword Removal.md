#nlp 

```
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')
print(stopwords.words('english'))

filtered_sentence = [w for w in word_tokens if not w.lower() in stop_words]
```
