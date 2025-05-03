from flask import Flask, render_template, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os

app = Flask(__name__)
app.secret_key = 'replace-this-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
db = SQLAlchemy(app)

# Models
class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    invoices = db.relationship('Invoice', backref='client', lazy=True)

class Invoice(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date_due = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)

# Routes
@app.route('/')
def index():
    invoices = Invoice.query.all()
    return render_template('index.html', invoices=invoices)

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or not file.filename.endswith('.csv'):
            flash('Please upload a valid CSV file.')
            return redirect(request.url)

        df = pd.read_csv(file)
        for _, row in df.iterrows():
            client_name = row['client_name']
            client = Client.query.filter_by(name=client_name).first()
            if not client:
                client = Client(name=client_name)
                db.session.add(client)
                db.session.commit()

            invoice = Invoice(
                invoice_id=row['invoice_id'],
                amount=row['amount'],
                date_due=row['date_due'],
                status=row['status'],
                client_id=client.id
            )
            db.session.add(invoice)

        db.session.commit()
        flash('Invoices uploaded successfully.')
        return redirect('/')

    return render_template('upload.html')

if __name__ == '__main__':
    if not os.path.exists('data.db'):
        with app.app_context():
            db.create_all()
    app.run(debug=True)