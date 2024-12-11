import pandas as pd
import torch
from random_forest import RandomForest
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import joblib

data = pd.read_csv(r'Datasets\symptom based disease prediction\train.csv')
X = data.drop(columns=['prognosis']).values
X = X[:, :-1]  
y = data['prognosis']

disease_mapping = {disease: idx for idx, disease in enumerate(y.unique())}
y = y.map(disease_mapping)
disease = list(disease_mapping.keys())

X, y = torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.int64)

model = RandomForest()
model.fit(X, y)

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

print('Enter the symptoms you are facing...')
symptoms_faced = list(map(lambda x: x.strip().lower(), input().split(',')))

vector = [0] * len(symptoms)

for symptom in symptoms_faced:
    if symptom in symptoms:
        index = symptoms.index(symptom)
        vector[index] = 1

test_vector = torch.Tensor(vector)

predictions = model.predict(test_vector.unsqueeze(dim=0))

predicted_disease = disease[predictions[0]]

df = pd.read_csv(f'Datasets/disease severity/severity.csv')

print(df[df['Disease'] == predicted_disease])

# Load fine-tuned model
model_path = "fine_tuned_gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)
model = GPT2LMHeadModel.from_pretrained(model_path)

input_ids = tokenizer.encode(predicted_disease, return_tensors="pt")
output = model.generate(input_ids, max_length=100, num_return_sequences=1)

print("Generated Text:")
print(tokenizer.decode(output[0], skip_special_tokens=True))
joblib.dump(model, "symptom based disease prediction.pkl")