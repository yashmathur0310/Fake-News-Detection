from flask import Flask,render_template,request
import pandas as pd 
df=pd.read_csv('fake_news_train.csv',encoding='latin-1')
df=df.dropna()
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
x=df['news_text'].values
y=df['label'].values
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=42,test_size=0.127)
model=Pipeline([('tfidf',TfidfVectorizer(min_df=2,max_df=0.95)),('log',LogisticRegression())])
model.fit(x_train,y_train)



app=Flask(__name__)



@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method=='POST':
        question=request.form['question']

        question=model.predict([question])
        return render_template('result.html',prediction_text=question)




if __name__=='__main__':
    app.run(debug=True)
