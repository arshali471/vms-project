from marshmallow_sqlalchemy import SQLAlchemyAutoSchema, auto_field
from marshmallow import fields, validate, validates, ValidationError, Schema
from model import Settings, CategoryEnum
from database import db
from datetime import datetime

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

    # 🔹 New Fields in Schema
    category = fields.Enum(CategoryEnum, required=True)
    algorithm_start_time = fields.Time(required=False)  # Start time for processing
    algorithm_end_time = fields.Time(required=False)  # End time for processing
    algorithm_status = fields.Boolean(required=False)
    algorithm_status = fields.Boolean(required=False)


class UpdateAlgorithmTimingSchema(Schema):
    category = fields.Str(required=True)
    algorithm_start_time = fields.Str(required=True)
    algorithm_end_time = fields.Str(required=True)

    @validates("category")
    def validate_category(self, value):
        """ Ensure category is one of the allowed Enum values """
        if value not in [e.value for e in CategoryEnum]:
            raise ValidationError(f"Invalid category '{value}'. Allowed values: {', '.join(e.value for e in CategoryEnum)}")

    @validates("algorithm_start_time")
    @validates("algorithm_end_time")
    def validate_time(self, value):
        """ Ensure time is in HH:MM format """
        try:
            datetime.strptime(value, "%H:%M")
        except ValueError:
            raise ValidationError("Invalid time format. Use HH:MM (24-hour format).")