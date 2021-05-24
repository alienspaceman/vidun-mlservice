from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


class Result(db.Model):
    __tablename__ = 'results'

    id = db.Column(db.Integer, primary_key=True)
    request = db.Column(db.String())
    result = db.Column(db.String())
    created_at = db.Column(db.DateTime,
                           index=False,
                           unique=False,
                           nullable=False)

    def __init__(self, request, result, created_at):
        self.request = request
        self.result = result
        self.created_at = created_at

    def __repr__(self):
        return '<id {}>'.format(self.id)
