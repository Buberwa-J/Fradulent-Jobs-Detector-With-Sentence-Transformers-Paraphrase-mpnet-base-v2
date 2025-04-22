import os
from flask import Flask
from source.flask_app.inference_routes import inference_bp
from flask_cors import CORS
from waitress import serve

app = Flask(__name__)
CORS(app)
app.register_blueprint(inference_bp)

if __name__ == '__main__':
    # Get PORT from environment, default to 5000 for local dev
    port = int(os.environ.get("PORT", 5000))
    serve(app, host='0.0.0.0', port=port)
