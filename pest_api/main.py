from flask import Flask, request, jsonify
import numpy as np
from soil_pest_identifier import identify_pest, generate_report

app = Flask(__name__)

@app.route("/detect_pest", methods=["POST"])
def detect_pest():
    try:
        body = request.get_json()
        signal = np.array(body["signal"])
        fs = body.get("sampling_rate", 44100)
        depth = body.get("sensor_depth_cm", 10.0)

        results = identify_pest(signal, sampling_rate=fs)
        report = generate_report(results, sensor_depth_cm=depth)

        return jsonify({
            "report": report,
            "matches": [
                {
                    "pest": r.pest.common_name,
                    "score": r.score,
                    "confidence": r.confidence,
                    "damage": r.pest.damage_type,
                    "treatment": r.pest.treatment
                }
                for r in results
            ]
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run()