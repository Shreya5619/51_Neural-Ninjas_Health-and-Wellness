import joblib
import torch
from flask import Flask, request, jsonify, render_template
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import pandas as pd

app = Flask(__name__)

# Load your RandomForest model (make sure the .pkl file path is correct)
model = joblib.load('symptom based disease prediction.pkl')

# Load GPT-2 fine-tuned model for generating first-aid/self-care advice
gpt2_model_path = "fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)

# Load disease severity data
severity_df = pd.read_csv(r'Datasets/disease severity/severity.csv')
data = pd.read_csv(r'Datasets\symptom based disease prediction\train.csv')
X = data.drop(columns=['prognosis']).values
X = X[:, :-1]  
y = data['prognosis']

disease_mapping = {disease: idx for idx, disease in enumerate(y.unique())}
y = y.map(disease_mapping)
disease = list(disease_mapping.keys())

# Symptoms list (lowercased for easier matching)
symptoms = ['Itching', 'Skin Rash', 'Nodayl Skin Eruptions',
           'Continuous Sneezing', 'Shivering', 'Chills', 'Joint Pain',
        	'Stomach Pain', 'Acidity', 'Ulcers on tongue', 'Muscle Wasting',
            'Vomiting',	'Burning Micturition', 'Spotting Urination', 'Fatigue',	'Weight_gain',
        	'Anxiety', 'Cold Hands and Feets', 'Mood Swings', 'Weight Loss',
            'Restlessness', 'Lethargy', 'Patches in Throat',
            'Irregular Sugar Level', 'Cough', 'High Fever', 'Sunken Eyes', 'Breathlessness', 'Sweating', 
            'Dehydration', 'Indigestion', 'Headache', 'Yellowish Skin', 'Dark Urine', 'Nausea', 'Loss Of Appetite', 
            'Pain Behind The Eyes', 'Back Pain', 'Constipation', 'Abdominal Pain', 'Diarrhoea', 'Mild Fever', 
            'Yellow Urine', 'Yellowing Of Eyes', 'Acute Liver Failure', 'Fluid Overload', 'Swelling Of Stomach', 
            'Swelled Lymph Nodes', 'Malaise', 'Blurred And Distorted Vision', 'Phlegm', 'Throat Irritation', 
            'Redness Of Eyes', 'Sinus Pressure', 'Runny Nose', 'Congestion', 'Chest Pain', 'Weakness In Limbs', 
            'Fast Heart Rate', 'Pain During Bowel Movements', 'Pain In Anal Region', 'Bloody Stool', 'Irritation In Anus', 
            'Neck Pain', 'Dizziness', 'Cramps', 'Bruising', 'Obesity', 'Swollen Legs', 'Swollen Blood Vessels', 
            'Puffy Face And Eyes', 'Enlarged Thyroid', 'Brittle Nails', 'Swollen Extremities', 'Excessive Hunger', 
            'Extra Marital Contacts', 'Drying And Tingling Lips', 'Slurred Speech', 'Knee Pain', 'Hip Joint Pain', 
            'Muscle Weakness', 'Stiff Neck', 'Swelling Joints', 'Movement Stiffness', 'Spinning Movements', 
            'Loss Of Balance', 'Unsteadiness', 'Weakness Of One Body Side', 'Loss Of Smell', 'Bladder Discomfort', 
            'Foul Smell Of Urine', 'Continuous Feel Of Urine', 'Passage Of Gases', 'Internal Itching', 'Toxic Look Typhos', 
            'Depression', 'Irritability', 'Muscle Pain', 'Altered Sensorium', 'Red Spots Over Body', 'Belly Pain', 
            'Abnormal Menstruation', 'Dischromic Patches', 'Watering From Eyes', 'Increased Appetite', 'Polyuria', 
            'Family History', 'Mucoid Sputum', 'Rusty Sputum', 'Lack Of Concentration', 'Visual Disturbances', 
            'Receiving Blood Transfusion', 'Receiving Unsterile Injections', 'Coma', 'Stomach Bleeding', 'Distention Of Abdomen', 
            'History Of Alcohol Consumption', 'Fluid Overload', 'Blood In Sputum', 'Prominent Veins On Calf', 'Palpitations', 
            'Painful Walking', 'Pus Filled Pimples', 'Blackheads', 'Scurring', 'Skin Peeling', 'Silver Like Dusting', 
            'Small Dents In Nails', 'Inflammatory Nails', 'Blister', 'Red Sore Around Nose', 'Yellow Crust Ooze'
]

symptoms = [s.lower() for s in symptoms]

@app.route('/')
def index():
    return render_template('index.html')  # Serve the HTML page

from flask import request, jsonify
import torch
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 1. Get the symptoms from the frontend
        user_input = request.json.get('symptoms', '')
        
        if not user_input:
            raise ValueError("No symptoms provided in the request")
        
        # 2. Clean and process the input symptoms
        symptoms_faced = [symptom.strip().lower() for symptom in user_input.split(',')]
        print(f"User Input Symptoms: {symptoms_faced}")

        # 3. Create a vector representation of the symptoms
        vector = [0] * len(symptoms)
        for symptom in symptoms_faced:
            if symptom in symptoms:
                index = symptoms.index(symptom)
                vector[index] = 1
        
        test_vector = torch.Tensor(vector)
        print(f"Test Vector: {test_vector}")

        # 4. Predict the disease using the trained model
        predictions = model.predict(test_vector.unsqueeze(dim=0))
        predicted_disease_index = predictions[0]
        print(f"Predicted Disease Index: {predicted_disease_index}")
        
        # 5. Map predicted index to disease name
        predicted_disease = disease[predicted_disease_index]
        print(f"Predicted Disease: {predicted_disease}")

        # 6. Get severity information for the predicted disease
        severity_df = pd.read_csv(f'Datasets/disease severity/severity.csv')
        severity_info = severity_df[severity_df['Disease'] == predicted_disease]
        
        if severity_info.empty:
            severity_info = {"message": "Severity information not available."}
        else:
            # Convert the DataFrame row to a dictionary (you can pick specific columns as needed)
            severity_info = severity_info.iloc[0].to_dict()

        print(f"Severity Info: {severity_info}")

        # 7. Load fine-tuned GPT-2 model for first-aid/self-care advice
        gpt2_model_path = "fine_tuned_gpt2"
        tokenizer = GPT2Tokenizer.from_pretrained(gpt2_model_path)
        gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model_path)

        # 8. Generate first-aid/self-care advice based on predicted disease
        input_ids = tokenizer.encode(predicted_disease, return_tensors="pt")
        output = gpt2_model.generate(input_ids, max_length=100, num_return_sequences=1)
        generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
        print(f"Generated First-Aid/Self-Care Text: {generated_text}")

        # 9. Return the response to the frontend
        response = {
            "predicted_disease": predicted_disease,
            "severity_info": severity_info['Severity'],  # Ensure it's in a plain dictionary format
            "generated_text": generated_text
        }
        return jsonify(response)

    except Exception as e:
        # Handle any unexpected errors and provide the error message
        print(f"Error during prediction: {e}")
        return jsonify({"error": str(e)}), 400  # Return the error with a 400 status code

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000, debug=True)
