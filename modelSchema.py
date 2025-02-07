from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, auto_field
from marshmallow import fields, validate
from model import Settings
from database import db

class SettingsSchema(SQLAlchemyAutoSchema):
    class Meta:
        model = Settings
        sqla_session = db.session
        load_instance = True
        include_fk = True

    id = auto_field(dump_only=True)
    rtspUrl = auto_field(required=True, validate=validate.Length(max=2000))
    api_key = auto_field(validate=validate.Length(max=255))
    api_url = auto_field(validate=validate.Length(max=255))
    date = fields.DateTime(dump_only=True)
