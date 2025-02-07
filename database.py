import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from dotenv import load_dotenv
from sqlalchemy.pool import QueuePool

# Load environment variables first
load_dotenv()

# Retrieve environment variables
DB_USER = os.getenv('DATABASUSERENAME')
DB_PASSWORD = os.getenv('DATABASENAMEPASSWORD')
DB_HOST = os.getenv('DBIP')
DB_NAME = os.getenv('DATABASENAME')
ENVRONMENT = os.getenv('FLASK_ENV')

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Configure SQLAlchemy with the loaded environment variables
# if ( ENVRONMENT == 'docker'):
#     print(ENVRONMENT, "ENVRONMENT")
# app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://motion_detection:motionDetection$1@db:3306/motion_detection'
# else:
#     print(ENVRONMENT, "dev")
app.config['SQLALCHEMY_DATABASE_URI'] = f"mysql+pymysql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:3306/{DB_NAME}"


app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize SQLAlchemy
db = SQLAlchemy(app, engine_options={"pool_size": 10, "poolclass":QueuePool, "pool_pre_ping":True})

