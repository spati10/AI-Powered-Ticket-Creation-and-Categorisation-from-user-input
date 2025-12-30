from flask import Flask, request, jsonify, render_template
from scripts.generate_ticket import generate_ticket

app = Flask(__name__)

# ---------- Routes ----------

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/create_ticket", methods=["POST"])
def create_ticket():
    data = request.json
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Ticket text is required"}), 400

    ticket = generate_ticket(text)
    return jsonify(ticket)

# ---------- Run Server ----------
if __name__ == "__main__":
    app.run(debug=True)
