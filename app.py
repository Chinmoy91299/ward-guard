# Import the Flask class from the flask module
from flask import Flask, render_template
from tensorflow.keras.models import load_model

# Create an instance of the Flask class
app = Flask(__name__)
model = load_model('mask_detector.model')


# Register a route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ImageStream',methods=['POST'])
def ImageStream():
    """the live page"""
    return render_template('RealtimeImage.html',)

# Run the Flask application
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)