import os
import numpy as np
import pandas as pd
import pickle
import datetime

from flask import Flask, request, render_template, redirect, url_for, flash
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# ==============================================================================
# BASE DIR
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ==============================================================================
# APP CONFIG
# ==============================================================================
app = Flask(__name__)
app.config['SECRET_KEY'] = 'secret123'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///' + os.path.join(BASE_DIR, 'users.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

login_manager = LoginManager(app)
login_manager.login_view = 'login'

# ==============================================================================
# CONTEXT
# ==============================================================================
@app.context_processor
def inject_now():
    return {'now': datetime.datetime.now()}

# ==============================================================================
# DATABASE
# ==============================================================================
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True)
    password = db.Column(db.String(150))

class SymptomHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    symptoms = db.Column(db.String(500))
    prediction = db.Column(db.String(100))
    user_id = db.Column(db.Integer)

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# ==============================================================================
# LOAD DATA (SAFE)
# ==============================================================================
svc = None
description = precautions = medications = diets = workout = None

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
    print("⚠️ Data load failed:", e)

# ==============================================================================
# BASIC DATA (SAFE DEFAULT)
# ==============================================================================
symptoms_dict = {'itching': 0, 'skin_rash': 1}
diseases_list = {0: 'Fungal infection', 1: 'Allergy'}

categorized_symptoms = {
    "General": ['itching', 'skin_rash']
}

# ==============================================================================
# HELPERS
# ==============================================================================
def get_predicted_value(symptoms):
    if svc is None:
        return "Model not loaded"

    input_vector = np.zeros(len(symptoms_dict))

    for s in symptoms:
        if s in symptoms_dict:
            input_vector[symptoms_dict[s]] = 1

    try:
        pred = svc.predict([input_vector])[0]
        return diseases_list.get(pred, "Unknown")
    except Exception as e:
        return f"Prediction error: {str(e)}"

def helper(dis):
    try:
        desc = description[description['Disease'] == dis]['Description'].values[0]
        return desc
    except:
        return "No description available"

# ==============================================================================
# ROUTES
# ==============================================================================
@app.route("/")
def index():
    try:
        return render_template("index.html", categorized_symptoms=categorized_symptoms)
    except Exception as e:
        return f"Template error: {str(e)}"

@app.route("/predict", methods=["POST"])
@login_required
def predict():
    try:
        symptoms = request.form.getlist('symptoms')

        if not symptoms:
            flash("Select symptoms")
            return redirect(url_for('index'))

        prediction = get_predicted_value(symptoms)

        entry = SymptomHistory(
            symptoms=", ".join(symptoms),
            prediction=prediction,
            user_id=current_user.id
        )
        db.session.add(entry)
        db.session.commit()

        desc = helper(prediction)

        return render_template("index.html",
                               categorized_symptoms=categorized_symptoms,
                               predicted_disease=prediction,
                               dis_des=desc)

    except Exception as e:
        return f"Predict error: {str(e)}"

@app.route("/history")
@login_required
def history():
    data = SymptomHistory.query.filter_by(user_id=current_user.id).all()
    return render_template("history.html", history=data)

# ---------------- AUTH ---------------- #

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form.get("username")
        password = generate_password_hash(request.form.get("password"))

        user = User(username=username, password=password)
        db.session.add(user)
        db.session.commit()

        return redirect(url_for("login"))

    return render_template("register.html")

@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        user = User.query.filter_by(username=request.form.get("username")).first()

        if user and check_password_hash(user.password, request.form.get("password")):
            login_user(user)
            return redirect(url_for("index"))

        return "Invalid login"

    return render_template("login.html")

@app.route("/logout")
def logout():
    logout_user()
    return redirect(url_for("index"))

# ---------------- STATIC PAGES ---------------- #

@app.route("/about")
def about():
    return "About Page"

@app.route("/contact")
def contact():
    return "Contact Page"

@app.route("/developer")
def developer():
    return "Developer Page"

@app.route("/blog")
def blog():
    return "Blog Page"

# ==============================================================================
# RUN
# ==============================================================================
if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
