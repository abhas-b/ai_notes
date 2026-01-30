#nlp 

*  Term Frequency Inverse Document Frequency
* Term Frequency
	* tf(t,d) = count of t in d / number of words in d
* Document Frequency:
	* df(t) =  number of documents in which t occurs
* Inverse Document Frequency
	* N(t) = Number of documents containing the term t = df(t)
	* idf(t) = log(N/ df(t))
	* tf-idf(t, d) = tf(t, d) * idf(t)

```
# create object
tfidf = TfidfVectorizer()

# get tf-df values
result = tfidf.fit_transform(string)

# get idf values
print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names(), tfidf.idf_):
	print(ele1, ':', ele2)

```


Attributes:
1. tfidf.idf_
2. tfidf.vocabulary_
3. tfidf.get_feature_names()
4. 