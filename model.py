from database import db

class Settings(db.Model):
    __tablename__ = 'settings'
    id = db.Column(db.Integer, primary_key=True)
    rtspUrl = db.Column(db.String(767), unique=True)
    api_key = db.Column(db.String(255))
    api_url = db.Column(db.String(255))
    normal_motion = db.Column(db.Boolean, default=False)
    person_motion = db.Column(db.Boolean, default=False)
    faces = db.Column(db.Boolean, default=False)
    high_person_count = db.Column(db.Boolean, default=False)
    pose = db.Column(db.Boolean, default=False)
    fire_detections = db.Column(db.Boolean, default=False)
    electronic_devices = db.Column(db.Boolean, default=False)
    stopped_persons = db.Column(db.Boolean, default=False)
    deleted_all = db.Column(db.Boolean, default=False)
    date = db.Column(db.DateTime, default=db.func.current_timestamp())

    @staticmethod
    def create(new_setting):
        """
        Static method to add a new Settings record to the database.
        Expects a Settings instance as input.
        """
        db.session.add(new_setting)
        db.session.commit()
        return new_setting
