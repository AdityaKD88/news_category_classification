from flask import Flask, render_template, request
from joblib import load
app = Flask(__name__)

def load_model():
    filepath = 'news.h5'
    return load(filepath)

def pred(inp):
    userinp=[inp]
    x = load_model().get('vectorizer').transform(userinp)
    p = load_model().get('classifier').predict(x)
    if p == [0]:
        return "Business News"
    elif p == [1]:
        return "Tech News"
    elif p == [2]:
        return "Politics News"
    elif p == [3]:
        return "Sports News"
    elif p == [4]:
        return "Entertainment News"

@app.route('/', methods=['GET','POST'])
def index():
    if request.method=="POST":
        form = request.form
        inp = form.get('inp')
        #print(inp)
        prediction = pred(inp)
        print(prediction)
        return render_template('index.html',inp=inp,prediction=prediction)
    return render_template('index.html')

if __name__ == '__main__':
  app.run(host='127.0.0.1', port=8000, debug=True)