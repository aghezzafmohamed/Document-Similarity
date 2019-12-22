from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import string
import os


lemmatizer = WordNetLemmatizer()
corpus=[]
files=[]
score={}
new_sentence = []

#Read documents and store them in a list: corpus
for file in os.listdir("data"):
    f=open("data/"+file,encoding="latin-1")
    files.append(file.split(".")[0])
    text=f.read()    
    #Clean the corpus: eliminate stopwords, punctuations, symbols, ...
    stop_words = set(stopwords.words('english')+ list(string.punctuation)+list("\n") )
    words = word_tokenize(text)
    words = [word.lower() for word in words if word.isalpha()]
    for word in words:
        if word not in stop_words:    
            
            #Reduce the tokens to their roots
            s =lemmatizer.lemmatize(word)
            new_sentence.append(s.lower())      
    data=" ".join(new_sentence)
    corpus.append(data)

#Read the request
requete=input("Give your request: ")
corpus.append(requete)

#Clear the BOW Use TF-IDF
vect=TfidfVectorizer()
dtm = vect.fit_transform(corpus)

#Use the Cosine Similarity Method
j=0
for i in files:
    x=cosine_similarity(dtm[-1], dtm[j])
    score[i]=x[0][0]
    j+=1
liste=sorted(score.items(), key=lambda t: t[1], reverse=True)

#Display similarity results in ascending order
print("The request \""+requete+"\" is similar to the document :\n")
for i in liste:
    print("-"+i[0]+": "+str(i[1]))



