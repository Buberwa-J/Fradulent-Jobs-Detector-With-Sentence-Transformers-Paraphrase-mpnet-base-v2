from flask import Blueprint, request, jsonify
from source.inference_pipeline import run_inference

inference_bp = Blueprint('inference', __name__)


@inference_bp.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = request.get_json()
        result_df = run_inference(input_data)

        response = {
            "fraudulent_prediction": int(result_df['fraudulent_prediction'].iloc[0]),
            "fraud_probability": float(result_df['fraud_probability'].iloc[0])
        }
        return jsonify(response), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500
