from flask import Flask
from source.flask_app.inference_routes import inference_bp
from flask_cors import CORS
from waitress import serve

app = Flask(__name__)
CORS(app)
app.register_blueprint(inference_bp)

if __name__ == '__main__':
    serve(app, host='0.0.0.0', port=5000)
