from flask import Flask, render_template, request, redirect, flash
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
import os
os.makedirs("data", exist_ok=True)
from sqlalchemy import inspect

app = Flask(__name__)
app.secret_key = 'replace-this-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data/data.db'
db = SQLAlchemy(app)
with app.app_context():
    inspector = inspect(db.engine)
    if not inspector.has_table("invoice"):
        print("[INFO] Creating tables...")
        db.create_all()

# Models
class Client(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128), nullable=True)
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
@app.route('/edit/<int:invoice_id>', methods=['GET', 'POST'])
@app.route('/delete/<int:invoice_id>', methods=['POST'])
def delete_invoice(invoice_id):
    invoice = Invoice.query.get_or_404(invoice_id)
    db.session.delete(invoice)
    db.session.commit()
    flash('Invoice deleted successfully.')
    return redirect('/')

def edit_invoice(invoice_id):
    invoice = Invoice.query.get_or_404(invoice_id)
    if request.method == 'POST':
        invoice.amount = request.form['amount']
        invoice.date_due = request.form['date_due']
        invoice.status = request.form['status']
        invoice.client.email = request.form['client_email']
        db.session.commit()
        flash('Invoice updated successfully.')
        return redirect('/')
    return render_template('edit_invoice.html', invoice=invoice)

@app.route('/export')
@app.route('/export/<int:invoice_id>')
def export_invoice(invoice_id):
    inv = Invoice.query.get_or_404(invoice_id)
    data = [{
        'client_name': inv.client.name,
        'client_email': inv.client.email,
        'invoice_id': inv.invoice_id,
        'amount': inv.amount,
        'date_due': inv.date_due,
        'status': inv.status
    }]
    df = pd.DataFrame(data)
    filename = f"invoice_{inv.invoice_id}.csv"
    return df.to_csv(index=False), 200, {'Content-Type': 'text/csv', 'Content-Disposition': f'attachment; filename={filename}'}

def export():
    invoices = Invoice.query.all()
    data = []
    for inv in invoices:
        data.append({
            'client_name': inv.client.name,
            'invoice_id': inv.invoice_id,
            'amount': inv.amount,
            'date_due': inv.date_due,
            'status': inv.status
        })
    df = pd.DataFrame(data)
    return df.to_csv(index=False), 200, {'Content-Type': 'text/csv', 'Content-Disposition': 'attachment; filename=invoices.csv'}

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
                client = Client(name=client_name, email=row.get('client_email'))
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
    with app.app_context():
        inspector = inspect(db.engine)
        if not inspector.has_table("invoice"):
            print("[INFO] Creating tables...")
            db.create_all()
    app.run(debug=True, host='0.0.0.0', port=5005)