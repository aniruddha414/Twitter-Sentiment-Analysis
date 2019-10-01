

from flask import Flask, request
import re
import json
import pandas as pd
import nltk

from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib

nltk.download('punkt')
nltk.download('stopwords')

sw = stopwords.words('english')
ps = PorterStemmer()

df = pd.read_csv(filepath_or_buffer="processed_data.csv",encoding='iso-8859-1')

bow_vectorizer = CountVectorizer(max_df=0.90, min_df=2, max_features=1000, stop_words='english')
bow = bow_vectorizer.fit_transform(df['CleanText'])

model = joblib.load('modelLR.pkl')

def preprocess(tweet):
    
    tweet = tweet.lower()
    ct = re.sub('@[^\s]+','',tweet)
    ct = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', ct)
    ct = re.sub(r'[^\w\s]','',ct)
    ct = re.sub('[0-9]','',ct)
    
    ct = word_tokenize(ct)
    
    ct = [w for w in ct if w not in sw]
    
    regex = re.compile('[@_!$%^&*()<>?/\|}{~:]')
    
    for i in range(len(ct)):
        ct[i] = ct[i].replace('.','')
        
    words = [ps.stem(w) for w in ct if regex.search(w) == None and not w == '' ]
        
    for i in range(len(words)):
        words[i] = words[i].replace('.','')
        
    ct = ' '.join([w for w in words if not '.' in w])
    
    # print("fine in preprocessing")
    return ct

def makeCountVector(tweet):
    
    ct = preprocess(tweet)
    print(ct)
    ct_vec = bow_vectorizer.transform([ct])
    # print('fine in vectorizing')
    return ct_vec


print("Server is setup successfully -- ", len(df))

app = Flask(__name__)

@app.route('/')
def index():
  return 'Server Works!'
  
@app.route('/greet')
def say_hello():
  return 'Hello from Server'

@app.route('/sentiment', methods=['GET','POST'])
def getSentiment():
  if request.method == 'POST':
    content = request.json
    twit = content['tweet']
    print("tweet sent : -- ",twit)
    X = makeCountVector(twit)
    y = model.predict(X)
    print("sentiment : " , y[0])
    sentiment = str(y[0])
    d = {"senitment":sentiment}
    return json.dumps(d)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=80)