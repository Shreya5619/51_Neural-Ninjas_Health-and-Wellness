from flask import Flask, request, jsonify, send_from_directory

app = Flask(__name__)

# Serve the HTML file
@app.route('/')
def serve_form():
    return send_from_directory("html", "feedback_form.html")
if __name__ == "__main__":
    app.run(debug=True)
@app.route('/test')
def test():
    return "Server is running!"
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
'''
from flask import Flask, request, jsonify, render_template
import csv
import datetime

app = Flask(__name__)

# Serve the feedback form (HTML file)
@app.route('/')
def serve_form():
    return render_template('feedback_form.html')  # Renders the HTML from templates folder

# Handle the form submission (POST request)
@app.route('/submit', methods=['POST'])
def submit_feedback():
    form_data = request.get_json()  # Get JSON data from the request
    print("Received form data:", form_data)  # Log the data to verify it's coming through

    # Process the form data, save it to CSV (with timestamp)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open('bangalore_hospitals_feedback.csv', mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([form_data.get('hospital'), form_data.get('feedback'), form_data.get('rating'), timestamp])

    return jsonify({"status": "success", "message": "Feedback submitted successfully!"})

if __name__ == '__main__':
    app.run(debug=True)
'''