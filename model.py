from database import db
import enum
from datetime import datetime

# 🔹 Enum for Categories
class CategoryEnum(enum.Enum):
    CONTROL_ROOM = "CONTROL_ROOM"
    EXAM_HALL = "EXAM_HALL"
    GROUND_AREA = "GROUND_AREA"

class Settings(db.Model):
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    rtspUrl = db.Column(db.String(767), unique=True)
    api_key = db.Column(db.String(255))
    api_url = db.Column(db.String(255))
    
    # 🔹 Motion & Detection Settings
    normal_motion = db.Column(db.Boolean, default=False)
    person_motion = db.Column(db.Boolean, default=False)
    faces = db.Column(db.Boolean, default=False)
    high_person_count = db.Column(db.Boolean, default=False)
    pose = db.Column(db.Boolean, default=False)
    fire_detections = db.Column(db.Boolean, default=False)
    electronic_devices = db.Column(db.Boolean, default=False)
    stopped_persons = db.Column(db.Boolean, default=False)

    # 🔹 New Fields
    category = db.Column(db.Enum(CategoryEnum), nullable=False, default=CategoryEnum.CONTROL_ROOM)
    algorithm_start_time = db.Column(db.Time, nullable=True)  # Time when the algorithm starts
    algorithm_end_time = db.Column(db.Time, nullable=True)  # Time when the algorithm stops
    
    deleted_all = db.Column(db.Boolean, default=False)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())
    algorithm_status = db.Column(db.Boolean, default=True)

    @staticmethod
    def create(new_setting):
        """
        Static method to add a new Settings record to the database.
        Expects a Settings instance as input.
        """
        db.session.add(new_setting)
        db.session.commit()
        return new_setting
