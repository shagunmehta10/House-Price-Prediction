import os
import pickle
import sqlite3
import hashlib

import numpy as np
import pandas as pd

from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS


app = Flask(__name__)

app.secret_key = os.environ.get(
    "SECRET_KEY",
    "houseai-development-secret"
)

app.config.update(
    SESSION_COOKIE_HTTPONLY=True,
    SESSION_COOKIE_SECURE=True,
    SESSION_COOKIE_SAMESITE="None",
)

CORS(
    app,
    supports_credentials=True,
    origins=[
        "http://localhost:5173",
        "https://house-price-prediction-drab-x1.vercel.app",
        "https://house-price-prediction-git-main-shaguns-projects-7b2beb68.vercel.app",
    ],
)

# ============================================================
# DATA + MODEL
# ============================================================

data = pd.read_csv("cleaned_data.csv")

with open("RidgeModel.pkl", "rb") as file:
    pipe = pickle.load(file)


# ============================================================
# DATABASE
# ============================================================

DB_NAME = "houseai.db"


def get_db():
    db = sqlite3.connect(DB_NAME)
    db.row_factory = sqlite3.Row
    return db


def init_db():
    db = get_db()

    db.execute("""
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password TEXT NOT NULL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            location TEXT,
            total_sqft REAL,
            bath INTEGER,
            bhk INTEGER,
            prediction REAL
        )
    """)

    db.execute("""
        CREATE TABLE IF NOT EXISTS favorites (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            location TEXT,
            total_sqft REAL,
            bath INTEGER,
            bhk INTEGER,
            prediction REAL
        )
    """)

    db.commit()
    db.close()


init_db()


def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def current_user():
    user_id = session.get("user_id")

    if not user_id:
        return None

    db = get_db()

    user = db.execute(
        "SELECT id, name, email FROM users WHERE id = ?",
        (user_id,),
    ).fetchone()

    db.close()

    return dict(user) if user else None


# ============================================================
# FRONTEND PAGE
# ============================================================

@app.route("/")
def index():
    locations = sorted(data["location"].dropna().unique())
    return render_template("index.html", locations=locations)


# ============================================================
# AUTH
# ============================================================

@app.route("/api/auth/register", methods=["POST"])
def register():

    body = request.get_json(silent=True) or {}

    name = body.get("name", "").strip()
    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not name or not email or not password:
        return jsonify({
            "success": False,
            "message": "Name, email and password are required."
        }), 400

    db = get_db()

    existing = db.execute(
        "SELECT id FROM users WHERE email = ?",
        (email,),
    ).fetchone()

    if existing:
        db.close()

        return jsonify({
            "success": False,
            "message": "Email already registered."
        }), 409

    cursor = db.execute(
        """
        INSERT INTO users (name, email, password)
        VALUES (?, ?, ?)
        """,
        (name, email, hash_password(password)),
    )

    db.commit()

    user_id = cursor.lastrowid

    session["user_id"] = user_id

    db.close()

    return jsonify({
        "success": True,
        "user": {
            "id": user_id,
            "name": name,
            "email": email,
        },
    })


@app.route("/api/auth/login", methods=["POST"])
def login():

    body = request.get_json(silent=True) or {}

    email = body.get("email", "").strip().lower()
    password = body.get("password", "")

    if not email or not password:
        return jsonify({
            "success": False,
            "message": "Email and password are required."
        }), 400

    db = get_db()

    user = db.execute(
        """
        SELECT id, name, email, password
        FROM users
        WHERE email = ?
        """,
        (email,),
    ).fetchone()

    db.close()

    if not user or user["password"] != hash_password(password):

        return jsonify({
            "success": False,
            "message": "Invalid email or password."
        }), 401

    session["user_id"] = user["id"]

    return jsonify({
        "success": True,
        "user": {
            "id": user["id"],
            "name": user["name"],
            "email": user["email"],
        },
    })


@app.route("/api/auth/me", methods=["GET"])
def me():

    user = current_user()

    if not user:
        return jsonify({
            "authenticated": False,
            "user": None,
        }), 401

    return jsonify({
        "authenticated": True,
        "user": user,
    })


@app.route("/api/auth/logout", methods=["POST"])
def logout():

    session.clear()

    return jsonify({
        "success": True,
        "message": "Logged out successfully."
    })


# ============================================================
# LOCATIONS
# ============================================================

@app.route("/api/locations", methods=["GET"])
def locations():

    result = sorted(
        data["location"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    return jsonify({
        "locations": result
    })


# ============================================================
# PREDICTION
# ============================================================

@app.route("/api/predict", methods=["POST"])
def api_predict():

    body = request.get_json(silent=True) or {}

    location = body.get("location")
    total_sqft = body.get("total_sqft")
    bath = body.get("bath")
    bhk = body.get("bhk")

    if not all([
        location,
        total_sqft is not None,
        bath is not None,
        bhk is not None,
    ]):
        return jsonify({
            "success": False,
            "message": "location, total_sqft, bath and bhk are required."
        }), 400

    try:

        input_df = pd.DataFrame(
            [[
                str(location),
                float(total_sqft),
                float(bath),
                float(bhk),
            ]],
            columns=[
                "location",
                "total_sqft",
                "bath",
                "bhk",
            ],
        )

        prediction = float(pipe.predict(input_df)[0] * 1e5)

        user = current_user()

        if user:

            db = get_db()

            db.execute(
                """
                INSERT INTO history
                (user_id, location, total_sqft, bath, bhk, prediction)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    user["id"],
                    location,
                    total_sqft,
                    bath,
                    bhk,
                    prediction,
                ),
            )

            db.commit()
            db.close()

        return jsonify({
            "success": True,
            "prediction": round(prediction, 2),
            "input": {
                "location": location,
                "total_sqft": float(total_sqft),
                "bath": int(float(bath)),
                "bhk": int(float(bhk)),
            },
        })

    except Exception as error:

        print("Prediction error:", error)

        return jsonify({
            "success": False,
            "message": str(error),
        }), 500


# ============================================================
# HISTORY
# ============================================================

@app.route("/api/history", methods=["GET"])
def history():

    user = current_user()

    if not user:
        return jsonify({
            "history": []
        })

    db = get_db()

    rows = db.execute(
        """
        SELECT id, location, total_sqft, bath, bhk, prediction
        FROM history
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user["id"],),
    ).fetchall()

    db.close()

    return jsonify({
        "history": [dict(row) for row in rows]
    })


@app.route("/api/history/<int:item_id>", methods=["DELETE"])
def delete_history(item_id):

    user = current_user()

    if not user:
        return jsonify({
            "success": False,
            "message": "Not authenticated."
        }), 401

    db = get_db()

    db.execute(
        """
        DELETE FROM history
        WHERE id = ? AND user_id = ?
        """,
        (item_id, user["id"]),
    )

    db.commit()
    db.close()

    return jsonify({
        "success": True
    })


# ============================================================
# FAVORITES
# ============================================================

@app.route("/api/favorites", methods=["GET"])
def get_favorites():

    user = current_user()

    if not user:
        return jsonify({
            "favorites": []
        })

    db = get_db()

    rows = db.execute(
        """
        SELECT id, location, total_sqft, bath, bhk, prediction
        FROM favorites
        WHERE user_id = ?
        ORDER BY id DESC
        """,
        (user["id"],),
    ).fetchall()

    db.close()

    return jsonify({
        "favorites": [dict(row) for row in rows]
    })


@app.route("/api/favorites", methods=["POST"])
def add_favorite():

    user = current_user()

    if not user:
        return jsonify({
            "success": False,
            "message": "Not authenticated."
        }), 401

    body = request.get_json(silent=True) or {}

    db = get_db()

    db.execute(
        """
        INSERT INTO favorites
        (user_id, location, total_sqft, bath, bhk, prediction)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        (
            user["id"],
            body.get("location"),
            body.get("total_sqft"),
            body.get("bath"),
            body.get("bhk"),
            body.get("prediction"),
        ),
    )

    db.commit()
    db.close()

    return jsonify({
        "success": True
    })


@app.route("/api/favorites/<int:item_id>", methods=["DELETE"])
def remove_favorite(item_id):

    user = current_user()

    if not user:
        return jsonify({
            "success": False,
            "message": "Not authenticated."
        }), 401

    db = get_db()

    db.execute(
        """
        DELETE FROM favorites
        WHERE id = ? AND user_id = ?
        """,
        (item_id, user["id"]),
    )

    db.commit()
    db.close()

    return jsonify({
        "success": True
    })


# ============================================================
# DASHBOARD
# ============================================================

@app.route("/api/dashboard", methods=["GET"])
def dashboard():

    user = current_user()

    if not user:
        return jsonify({
            "authenticated": False,
            "total_predictions": 0,
            "favorites": 0,
        })

    db = get_db()

    prediction_count = db.execute(
        "SELECT COUNT(*) FROM history WHERE user_id = ?",
        (user["id"],),
    ).fetchone()[0]

    favorite_count = db.execute(
        "SELECT COUNT(*) FROM favorites WHERE user_id = ?",
        (user["id"],),
    ).fetchone()[0]

    db.close()

    return jsonify({
        "authenticated": True,
        "total_predictions": prediction_count,
        "favorites": favorite_count,
    })


# ============================================================
# PROFILE
# ============================================================

@app.route("/api/profile", methods=["GET"])
def profile():

    user = current_user()

    if not user:
        return jsonify({
            "message": "Not authenticated."
        }), 401

    return jsonify({
        "user": user
    })


@app.route("/api/profile", methods=["PUT"])
def update_profile():

    user = current_user()

    if not user:
        return jsonify({
            "message": "Not authenticated."
        }), 401

    body = request.get_json(silent=True) or {}

    name = body.get("name")

    if name:

        db = get_db()

        db.execute(
            "UPDATE users SET name = ? WHERE id = ?",
            (name.strip(), user["id"]),
        )

        db.commit()
        db.close()

    return jsonify({
        "success": True,
        "user": current_user(),
    })


# ============================================================
# LOCAL RUN
# ============================================================

if __name__ == "__main__":
    app.run(
        debug=True,
        port=5001,
    )

