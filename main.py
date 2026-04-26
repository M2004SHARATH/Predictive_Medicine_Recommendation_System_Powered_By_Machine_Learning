import os
from flask import Flask, request, render_template, redirect, url_for, flash
import numpy as np
import pandas as pd
import pickle
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo
import datetime # <-- Added for Jinja context processor

# ==============================================================================
# 1. APP & DATABASE CONFIGURATION
# ==============================================================================

# Initialize the Flask App
app = Flask(__name__)

# Secret key for session management and CSRF protection
app.config['SECRET_KEY'] = 'your_super_secret_key_change_this_later'

# Configure the SQLite database
app.config['SQLALCHEMY_DATABASE_URI'] = ''sqlite:///' + os.path.join(BASE_DIR, 'users.db')'
db = SQLAlchemy(app)

# Configure Flask-Login
login_manager = LoginManager(app)
login_manager.login_view = 'login'  # Redirect users to 'login' page if not logged in
login_manager.login_message_category = 'info' # Bootstrap category for flash messages

# ==============================================================================
# 2. CONTEXT PROCESSOR (Fixes 'now is undefined' error)
# ==============================================================================

@app.context_processor
def inject_now():
    """Makes the current datetime object available as 'now' in all templates."""
    # datetime.datetime.now() returns the datetime OBJECT, which we access as 'now.year' in Jinja
    return {'now': datetime.datetime.now()}

# ==============================================================================
# 3. DATABASE MODELS
# ==============================================================================

class User(UserMixin, db.Model):
    """User model for storing credentials."""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    symptoms = db.relationship('SymptomHistory', backref='author', lazy=True)

class SymptomHistory(db.Model):
    """SymptomHistory model for storing user's prediction history."""
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    """Flask-Login function to retrieve a user from the database."""
    return User.query.get(int(user_id))

# ==============================================================================
# 4. WEB FORMS (using Flask-WTF)
# ==============================================================================

class RegistrationForm(FlaskForm):
    """Registration form."""
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[DataRequired(), EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    """Login form."""
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# ==============================================================================
# 5. LOAD DATASETS, MODEL & DEFINE SYMPTOM DATA
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_DIR = os.path.join(BASE_DIR, "datasets")
MODEL_DIR = os.path.join(BASE_DIR, "models")
# Load all necessary CSV files
try:
    # NOTE: You must have 'datasets/' and 'models/' directories with these files
     sym_des = pd.read_csv(os.path.join(DATASET_DIR, "symtoms_df.csv"))
    precautions = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
    workout = pd.read_csv(os.path.join(DATASET_DIR, "workout_df.csv"))
    description = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
    medications = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
    diets = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))

    svc = pickle.load(open(os.path.join(MODEL_DIR, "svc.pkl"), "rb"))
except FileNotFoundError as e:
    print(f"Error loading data or model: {e}")
    # Use dummy data/model if files are missing for development, but stop execution for production
    # For simplicity, we'll proceed assuming files exist as per original code structure

# Dictionaries for symptoms and diseases from your notebook (Truncated for brevity)
symptoms_dict = {'itching': 0, 'skin_rash': 1, 'nodal_skin_eruptions': 2, 'continuous_sneezing': 3, 'shivering': 4, 'chills': 5, 'joint_pain': 6, 'stomach_pain': 7, 'acidity': 8, 'ulcers_on_tongue': 9, 'muscle_wasting': 10, 'vomiting': 11, 'burning_micturition': 12, 'spotting_ urination': 13, 'fatigue': 14, 'weight_gain': 15, 'anxiety': 16, 'cold_hands_and_feets': 17, 'mood_swings': 18, 'weight_loss': 19, 'restlessness': 20, 'lethargy': 21, 'patches_in_throat': 22, 'irregular_sugar_level': 23, 'cough': 24, 'high_fever': 25, 'sunken_eyes': 26, 'breathlessness': 27, 'sweating': 28, 'dehydration': 29, 'indigestion': 30, 'headache': 31, 'yellowish_skin': 32, 'dark_urine': 33, 'nausea': 34, 'loss_of_appetite': 35, 'pain_behind_the_eyes': 36, 'back_pain': 37, 'constipation': 38, 'abdominal_pain': 39, 'diarrhoea': 40, 'mild_fever': 41, 'yellow_urine': 42, 'yellowing_of_eyes': 43, 'acute_liver_failure': 44, 'fluid_overload': 45, 'swelling_of_stomach': 46, 'swelled_lymph_nodes': 47, 'malaise': 48, 'blurred_and_distorted_vision': 49, 'phlegm': 50, 'throat_irritation': 51, 'redness_of_eyes': 52, 'sinus_pressure': 53, 'runny_nose': 54, 'congestion': 55, 'chest_pain': 56, 'weakness_in_limbs': 57, 'fast_heart_rate': 58, 'pain_during_bowel_movements': 59, 'pain_in_anal_region': 60, 'bloody_stool': 61, 'irritation_in_anus': 62, 'neck_pain': 63, 'dizziness': 64, 'cramps': 65, 'bruising': 66, 'obesity': 67, 'swollen_legs': 68, 'swollen_blood_vessels': 69, 'puffy_face_and_eyes': 70, 'enlarged_thyroid': 71, 'brittle_nails': 72, 'swollen_extremeties': 73, 'excessive_hunger': 74, 'extra_marital_contacts': 75, 'drying_and_tingling_lips': 76, 'slurred_speech': 77, 'knee_pain': 78, 'hip_joint_pain': 79, 'muscle_weakness': 80, 'stiff_neck': 81, 'swelling_joints': 82, 'movement_stiffness': 83, 'spinning_movements': 84, 'loss_of_balance': 85, 'unsteadiness': 86, 'weakness_of_one_body_side': 87, 'loss_of_smell': 88, 'bladder_discomfort': 89, 'foul_smell_of urine': 90, 'continuous_feel_of_urine': 91, 'passage_of_gases': 92, 'internal_itching': 93, 'toxic_look_(typhos)': 94, 'depression': 95, 'irritability': 96, 'muscle_pain': 97, 'altered_sensorium': 98, 'red_spots_over_body': 99, 'belly_pain': 100, 'abnormal_menstruation': 101, 'dischromic _patches': 102, 'watering_from_eyes': 103, 'increased_appetite': 104, 'polyuria': 105, 'family_history': 106, 'mucoid_sputum': 107, 'rusty_sputum': 108, 'lack_of_concentration': 109, 'visual_disturbances': 110, 'receiving_blood_transfusion': 111, 'receiving_unsterile_injections': 112, 'coma': 113, 'stomach_bleeding': 114, 'distention_of_abdomen': 115, 'history_of_alcohol_consumption': 116, 'fluid_overload.1': 117, 'blood_in_sputum': 118, 'prominent_veins_on_calf': 119, 'palpitations': 120, 'painful_walking': 121, 'pus_filled_pimples': 122, 'blackheads': 123, 'scurring': 124, 'skin_peeling': 125, 'silver_like_dusting': 126, 'small_dents_in_nails': 127, 'inflammatory_nails': 128, 'blister': 129, 'red_sore_around_nose': 130, 'yellow_crust_ooze': 131}
diseases_list = {15: 'Fungal infection', 4: 'Allergy', 16: 'GERD', 9: 'Chronic cholestasis', 14: 'Drug Reaction', 33: 'Peptic ulcer diseae', 1: 'AIDS', 12: 'Diabetes ', 17: 'Gastroenteritis', 6: 'Bronchial Asthma', 23: 'Hypertension ', 30: 'Migraine', 7: 'Cervical spondylosis', 32: 'Paralysis (brain hemorrhage)', 28: 'Jaundice', 29: 'Malaria', 8: 'Chicken pox', 11: 'Dengue', 37: 'Typhoid', 40: 'hepatitis A', 19: 'Hepatitis B', 20: 'Hepatitis C', 21: 'Hepatitis D', 22: 'Hepatitis E', 3: 'Alcoholic hepatitis', 36: 'Tuberculosis', 10: 'Common Cold', 34: 'Pneumonia', 13: 'Dimorphic hemmorhoids(piles)', 18: 'Heart attack', 39: 'Varicose veins', 26: 'Hypothyroidism', 24: 'Hyperthyroidism', 25: 'Hypoglycemia', 31: 'Osteoarthristis', 5: 'Arthritis', 0: '(vertigo) Paroymsal Positional Vertigo', 2: 'Acne', 38: 'Urinary tract infection', 35: 'Psoriasis', 27: 'Impetigo'}

# Categorized symptoms for the accordion UI
categorized_symptoms = {
    "General & Systemic": ['fatigue', 'chills', 'shivering', 'sweating', 'lethargy', 'nausea', 'vomiting', 'high_fever', 'mild_fever', 'loss_of_appetite', 'weight_loss', 'weight_gain', 'restlessness', 'malaise', 'dizziness', 'unsteadiness', 'loss_of_balance', 'dehydration', 'anxiety', 'mood_swings', 'depression', 'irritability', 'altered_sensorium', 'family_history'],
    "Skin & Nails": ['itching', 'skin_rash', 'nodal_skin_eruptions', 'pus_filled_pimples', 'blackheads', 'scurring', 'skin_peeling', 'silver_like_dusting', 'blister', 'red_sore_around_nose', 'yellow_crust_ooze', 'red_spots_over_body', 'dischromic _patches', 'brittle_nails', 'small_dents_in_nails', 'inflammatory_nails'],
    "Head, Eyes, Ears, Nose & Throat": ['headache', 'patches_in_throat', 'sunken_eyes', 'ulcers_on_tongue', 'yellowing_of_eyes', 'blurred_and_distorted_vision', 'redness_of_eyes', 'pain_behind_the_eyes', 'loss_of_smell', 'sinus_pressure', 'runny_nose', 'congestion', 'phlegm', 'throat_irritation', 'cough', 'mucoid_sputum', 'rusty_sputum', 'blood_in_sputum', 'watering_from_eyes', 'drying_and_tingling_lips', 'slurred_speech', 'puffy_face_and_eyes'],
    "Pain & Aches": ['joint_pain', 'muscle_pain', 'stomach_pain', 'abdominal_pain', 'back_pain', 'neck_pain', 'chest_pain', 'belly_pain', 'hip_joint_pain', 'knee_pain', 'painful_walking', 'pain_during_bowel_movements', 'pain_in_anal_region'],
    "Digestive & Abdominal": ['acidity', 'indigestion', 'constipation', 'diarrhoea', 'stomach_bleeding', 'swelling_of_stomach', 'distention_of_abdomen', 'passage_of_gases', 'bloody_stool', 'irritation_in_anus', 'internal_itching'],
    "Muscles & Joints": ['muscle_wasting', 'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'cramps', 'bruising', 'weakness_in_limbs', 'weakness_of_one_body_side'],
    "Cardiovascular & Circulatory": ['fast_heart_rate', 'palpitations', 'cold_hands_and_feets', 'swollen_legs', 'swollen_blood_vessels', 'prominent_veins_on_calf', 'receiving_blood_transfusion'],
    "Urinary & Reproductive": ['burning_micturition', 'spotting_ urination', 'yellow_urine', 'dark_urine', 'bladder_discomfort', 'foul_smell_of urine', 'continuous_feel_of_urine', 'polyuria', 'abnormal_menstruation'],
    "Other Specific Symptoms": ['continuous_sneezing', 'irregular_sugar_level', 'breathlessness', 'acute_liver_failure', 'fluid_overload', 'swelled_lymph_nodes', 'spinning_movements', 'toxic_look_(typhos)', 'lack_of_concentration', 'visual_disturbances', 'receiving_unsterile_injections', 'coma', 'history_of_alcohol_consumption', 'fluid_overload.1', 'enlarged_thyroid', 'swollen_extremeties', 'excessive_hunger', 'increased_appetite', 'extra_marital_contacts']
}

# Set of diseases that trigger the emergency alert
EMERGENCY_DISEASES = {
    'Heart attack', 'Paralysis (brain hemorrhage)', 'Pneumonia', 'Chronic cholestasis',
    'Drug Reaction', 'AIDS', 'Jaundice', 'Malaria', 'Dengue', 'Typhoid', 'hepatitis A',
    'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E', 'Alcoholic hepatitis', 'Tuberculosis'
}

# ==============================================================================
# 6. HELPER FUNCTIONS FOR PREDICTION
# ==============================================================================

def helper(dis):
    """Fetches recommendations for a given disease."""
    # Ensure all data frames and model were loaded correctly before accessing them
    try:
        desc = description[description['Disease'] == dis]['Description'].values[0]
        pre = precautions[precautions['Disease'] == dis][['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']].values[0].tolist()
        med = medications[medications['Disease'] == dis]['Medication'].values[0]
        die = diets[diets['Disease'] == dis]['Diet'].values[0]
        wrkout = workout[workout['disease'] == dis]['workout'].values[0]
        return desc, pre, med, die, wrkout
    except:
        return "N/A", ["Consult a doctor."], "N/A", "N/A", "N/A"


def get_predicted_value(patient_symptoms):
    if svc is None:
        return "Model not loaded"

    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    try:
        pred = svc.predict([input_vector])[0]
        return diseases_list.get(pred, "Unknown condition")
    except:
        return "Prediction error"

# ==============================================================================
# 7. APP ROUTES
# ==============================================================================

@app.route("/")
def index():
    """Renders the home page with categorized symptoms."""
    return render_template("index.html", categorized_symptoms=categorized_symptoms)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    """Handles symptom submission, prediction, and saving to history."""
    if request.method == 'POST':
        user_symptoms = request.form.getlist('symptoms')
        
        if not user_symptoms:
            flash("Please select at least one symptom.", "danger")
            return redirect(url_for('index'))

        symptoms_for_db = ", ".join(user_symptoms)
        predicted_disease = get_predicted_value(user_symptoms)
        
        # Save to history even if the condition is unknown
        history_entry = SymptomHistory(symptoms=symptoms_for_db, prediction=predicted_disease, author=current_user)
        db.session.add(history_entry)
        db.session.commit()

        # Handle case where the prediction is unknown
        if predicted_disease == "Unknown condition":
             flash("Could not determine the condition based on the symptoms provided. Please consult a doctor.", "warning")
             # Rerender index without detailed results, letting the template handle the "Unknown" state
             return render_template('index.html', 
                               categorized_symptoms=categorized_symptoms,
                               predicted_disease=predicted_disease,
                               is_emergency=False)

        # Fetch detailed recommendations
        desc, pre, med, die, wrkout = helper(predicted_disease)
        is_emergency = predicted_disease in EMERGENCY_DISEASES
        
        return render_template('index.html', 
                               categorized_symptoms=categorized_symptoms,
                               predicted_disease=predicted_disease, 
                               dis_des=desc,
                               my_precautions=pre, 
                               medications=[med],
                               my_diet=[die],
                               workout=[wrkout],
                               is_emergency=is_emergency)
    return redirect(url_for('index'))

@app.route('/history')
@login_required
def history():
    """Displays the current user's prediction history."""
    user_history = SymptomHistory.query.filter_by(user_id=current_user.id).order_by(SymptomHistory.id.desc()).all()
    return render_template('history.html', history=user_history)

# --- AUTHENTICATION ROUTES ---

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Handles user registration."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed_password = generate_password_hash(form.password.data, method='pbkdf2:sha256')
        new_user = User(username=form.username.data, password=hashed_password)
        db.session.add(new_user)
        db.session.commit()
        flash('Your account has been created! You can now log in.', 'success')
        return redirect(url_for('login'))
    return render_template('register.html', title='Register', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Handles user login."""
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user, remember=True)
            next_page = request.args.get('next')
            return redirect(next_page) if next_page else redirect(url_for('index'))
        else:
            flash('Login unsuccessful. Please check username and password.', 'danger')
    return render_template('login.html', title='Login', form=form)

@app.route('/logout')
def logout():
    """Handles user logout."""
    logout_user()
    return redirect(url_for('index'))

# --- STATIC PAGE ROUTES ---

# Note: The original code defines these simple routes, which rely on the HTML templates below.
@app.route('/about')
def about():
    return render_template("about.html")

@app.route('/contact')
def contact():
    return render_template("contact.html")

@app.route('/developer')
def developer():
    return render_template("developer.html")

@app.route('/blog')
def blog():
    return render_template("blog.html")

# ==============================================================================
# 8. RUN THE APPLICATION
# ==============================================================================

if __name__ == '__main__':
    with app.app_context():
        # Creates database and tables if they don't exist
        db.create_all()
    app.run(debug=True)
