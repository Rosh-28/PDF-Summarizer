from flask import Flask, jsonify, render_template
from generate_mcqs import generate_mcq

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")  # Loads templates/index.html

@app.route("/generate_mcq")
def generate_mcq():
    mcq = generate_mcq()
    if mcq:
        return jsonify(mcq)
    return jsonify({"error": "Failed to generate MCQ"}), 500

if __name__ == "__main__":
    app.run(debug=True)
