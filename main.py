import os
import numpy as np
import pandas as pd
import pickle
import datetime

from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from flask_wtf import FlaskForm
from wtforms import StringField, PasswordField, SubmitField
from wtforms.validators import DataRequired, Length, EqualTo

# ==============================================================================
# BASE DIRECTORY (IMPORTANT)
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# APP CONFIG
# ==============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'change_this_secret_key'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ==============================================================================
# CONTEXT PROCESSOR
# ==============================================================================
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# ==============================================================================
# DATABASE MODELS
# ==============================================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password = db.Column(db.String(150), nullable=False)
    symptoms = db.relationship('SymptomHistory', backref='author', lazy=True)

class SymptomHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.String(500), nullable=False)
    prediction = db.Column(db.String(100), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==============================================================================
# FORMS
# ==============================================================================
class RegistrationForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired(), Length(min=4, max=20)])
    password = PasswordField('Password', validators=[DataRequired(), Length(min=6)])
    confirm_password = PasswordField('Confirm Password', validators=[EqualTo('password')])
    submit = SubmitField('Sign Up')

class LoginForm(FlaskForm):
    username = StringField('Username', validators=[DataRequired()])
    password = PasswordField('Password', validators=[DataRequired()])
    submit = SubmitField('Login')

# ==============================================================================
# LOAD DATA
# ==============================================================================
try:
    DATASET_DIR = os.path.join(BASE_DIR, "datasets")
    MODEL_DIR = os.path.join(BASE_DIR, "models")

    description = pd.read_csv(os.path.join(DATASET_DIR, "description.csv"))
    precautions = pd.read_csv(os.path.join(DATASET_DIR, "precautions_df.csv"))
    medications = pd.read_csv(os.path.join(DATASET_DIR, "medications.csv"))
    diets = pd.read_csv(os.path.join(DATASET_DIR, "diets.csv"))
    workout = pd.read_csv(os.path.join(DATASET_DIR, "workout_df.csv"))

    svc = pickle.load(open(os.path.join(MODEL_DIR, "svc.pkl"), "rb"))

except Exception as e:
    print("❌ Error loading files:", e)

# ==============================================================================
# SYMPTOMS + DISEASES (USE YOUR FULL DICT HERE)
# ==============================================================================
symptoms_dict = {
    'itching': 0,
    'skin_rash': 1,
    'continuous_sneezing': 2
}

diseases_list = {
    0: 'Fungal infection',
    1: 'Allergy',
    2: 'Common Cold'
}

categorized_symptoms = {
    "General": ['itching', 'skin_rash', 'continuous_sneezing']
}

EMERGENCY_DISEASES = {'Heart attack', 'Pneumonia'}

# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================
def helper(dis):
    try:
        desc = description[description['Disease'] == dis]['Description'].values[0]
        pre = precautions[precautions['Disease'] == dis].values[0].tolist()
        med = medications[medications['Disease'] == dis]['Medication'].values[0]
        die = diets[diets['Disease'] == dis]['Diet'].values[0]
        wrkout = workout[workout['disease'] == dis]['workout'].values[0]
        return desc, pre, med, die, wrkout
    except:
        return "N/A", ["Consult doctor"], "N/A", "N/A", "N/A"

def get_predicted_value(patient_symptoms):
    input_vector = np.zeros(len(symptoms_dict))

    for item in patient_symptoms:
        if item in symptoms_dict:
            input_vector[symptoms_dict[item]] = 1

    try:
        prediction_index = svc.predict([input_vector])[0]
        return diseases_list.get(prediction_index, "Unknown condition")
    except:
        return "Unknown condition"

# ==============================================================================
# ROUTES
# ==============================================================================
@app.route("/")
def index():
    return render_template("index.html", categorized_symptoms=categorized_symptoms)

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    user_symptoms = request.form.getlist('symptoms')

    if not user_symptoms:
        flash("Select at least one symptom", "danger")
        return redirect(url_for('index'))

    prediction = get_predicted_value(user_symptoms)

    history = SymptomHistory(
        symptoms=", ".join(user_symptoms),
        prediction=prediction,
        author=current_user
    )
    db.session.add(history)
    db.session.commit()

    desc, pre, med, die, wrkout = helper(prediction)

    return render_template(
        'index.html',
        categorized_symptoms=categorized_symptoms,
        predicted_disease=prediction,
        dis_des=desc,
        my_precautions=pre,
        medications=[med],
        my_diet=[die],
        workout=[wrkout],
        is_emergency=prediction in EMERGENCY_DISEASES
    )

@app.route('/history')
@login_required
def history():
    data = SymptomHistory.query.filter_by(user_id=current_user.id).all()
    return render_template('history.html', history=data)

# ---------------- AUTH ---------------- #

@app.route('/register', methods=['GET', 'POST'])
def register():
    form = RegistrationForm()
    if form.validate_on_submit():
        hashed = generate_password_hash(form.password.data)
        user = User(username=form.username.data, password=hashed)
        db.session.add(user)
        db.session.commit()
        flash("Account created!", "success")
        return redirect(url_for('login'))
    return render_template('register.html', form=form)

@app.route('/login', methods=['GET', 'POST'])
def login():
    form = LoginForm()
    if form.validate_on_submit():
        user = User.query.filter_by(username=form.username.data).first()
        if user and check_password_hash(user.password, form.password.data):
            login_user(user)
            return redirect(url_for('index'))
        flash("Invalid credentials", "danger")
    return render_template('login.html', form=form)

@app.route('/logout')
def logout():
    logout_user()
    return redirect(url_for('index'))

# ---------------- STATIC PAGES (FIXED YOUR ERROR) ---------------- #

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
# RUN APP
# ==============================================================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
