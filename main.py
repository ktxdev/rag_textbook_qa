from pipeline import Pipeline
from flask import Flask, request, jsonify

app = Flask(__name__)
pipeline = Pipeline()

@app.route('/ask', methods=['POST'])
def ask_question():
    data = request.get_json(force=True)
    query = data.get('question')

    if not query:
        return jsonify({"error": "Missing 'question' in request"}), 400

    result = pipeline.query(query)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)