from flask import Flask, render_template, request ,json
from wtforms import Form, TextAreaField, validators
from sklearn.externals import joblib
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from flask import jsonify
from flask import Response

def clf_process(input_description):
	input_description = list([input_description])
	vocabulary = joblib.load('D:\\Python\\FLASK\\Vocabulary.pkl')
	vect = CountVectorizer(ngram_range=(1,3), binary =True, stop_words = 'english', vocabulary = vocabulary)
	input_description = vect.transform(input_description)
	tfidf_data = TfidfTransformer(use_idf=False).transform(input_description)
	clf = joblib.load('D:\\Python\\FLASK\\Classifier.pkl')
	resultant_prediction = clf.predict(tfidf_data)
	return resultant_prediction
	
app = Flask(__name__)

class HelloForm(Form):
	sayhello = TextAreaField('',[validators.DataRequired()])

@app.route('/')
def index():
	form = HelloForm(request.form)
	return render_template('first_app.html', form=form)

@app.route('/prediction', methods=['POST'])
def prediction():
	form = HelloForm(request.form)
	if request.method == 'POST' and form.validate():
		input_description = request.form['sayhello']
		result = clf_process(input_description)
        d=dict({'type':result[0]})
        return render_template('prediction.html', name=d)
	return render_template('first_app.html', form=form)
    
if __name__ == '__main__':
    app.run(debug=True)
	