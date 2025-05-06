import os
os.makedirs("/app/data", exist_ok=True)

from flask import Flask, render_template, request, redirect, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
import pandas as pd
from sqlalchemy import inspect
import openai
import configparser


app = Flask(__name__)
app.secret_key = 'replace-this-secret'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:////app/data/data.db'
db = SQLAlchemy(app)
# Load OpenAI API key from cfg file
config = configparser.ConfigParser()
config.read('cfg/openai.cfg')
openai.api_key = config.get('DEFAULT', 'OPENAI_API_KEY', fallback='')

# Run table creation if needed
with app.app_context():
    inspector = inspect(db.engine)
    if not inspector.has_table("invoice"):
        print("[INFO] Creating tables...")
        db.create_all()

# --- Models ---
class Client(db.Model):
    __tablename__ = 'client'
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), unique=True, nullable=False)
    email = db.Column(db.String(128), nullable=True)
    invoices = db.relationship('Invoice', backref='client', lazy=True)

class Invoice(db.Model):
    __tablename__ = 'invoice'
    id = db.Column(db.Integer, primary_key=True)
    invoice_id = db.Column(db.String(64), nullable=False)
    amount = db.Column(db.Float, nullable=False)
    date_due = db.Column(db.String(64), nullable=False)
    status = db.Column(db.String(32), nullable=False)
    client_id = db.Column(db.Integer, db.ForeignKey('client.id'), nullable=False)


# --- Routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    """
    GET: Show all invoices
    POST: Upload a CSV file with invoices (hidden in the UI, but still functional)
    """
    if request.method == 'POST':
        file = request.files.get('csv_file')
        if not file or not file.filename.endswith('.csv'):
            flash('Please upload a valid CSV file.')
            return redirect(request.url)

        df = pd.read_csv(file)
        for _, row in df.iterrows():
            client_name = row['client_name']
            client_email = row.get('client_email')

            # Check if client already exists
            client = Client.query.filter_by(name=client_name).first()
            if not client:
                client = Client(name=client_name, email=client_email)
                db.session.add(client)
                db.session.commit()

            # Create the invoice record
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

    invoices = Invoice.query.all()
    return render_template('index.html', invoices=invoices)


@app.route('/delete/<int:invoice_id>', methods=['POST'])
def delete_invoice(invoice_id):
    """
    POST: Delete a specific invoice by ID.
    """
    invoice = Invoice.query.get_or_404(invoice_id)
    db.session.delete(invoice)
    db.session.commit()
    flash('Invoice deleted successfully.')
    return redirect('/')


@app.route('/edit/<int:invoice_id>', methods=['POST'])
def edit_invoice(invoice_id):
    """
    POST: Update a specific invoice. Returns JSON so the front-end can update without reloading.
    Expecting form fields: amount, date_due, status, client_email
    """
    invoice = Invoice.query.get_or_404(invoice_id)
    # Update fields
    invoice.amount = request.form['amount']
    invoice.date_due = request.form['date_due']
    invoice.status = request.form['status']

    # Update client email
    invoice.client.email = request.form['client_email']

    db.session.commit()

    # Return JSON to match front-end's fetch() expectations
    return jsonify({
        "client_name": invoice.client.name,
        "invoice_id": invoice.invoice_id,
        "amount": invoice.amount,
        "date_due": invoice.date_due,
        "status": invoice.status
    }), 200


@app.route('/export')
@app.route('/export/<int:invoice_id>')
def export_invoice(invoice_id=None):
    """
    GET: Export all invoices or a single invoice as CSV.
    If <invoice_id> is provided, export only that one invoice.
    """
    if invoice_id:
        inv = Invoice.query.get_or_404(invoice_id)
        data = [{
            'client_name': inv.client.name,
            'client_email': inv.client.email,
            'invoice_id': inv.invoice_id,
            'amount': inv.amount,
            'date_due': inv.date_due,
            'status': inv.status
        }]
        filename = f"invoice_{invoice_id}.csv"
    else:
        invoices = Invoice.query.all()
        data = [{
            'client_name': inv.client.name,
            'client_email': inv.client.email,
            'invoice_id': inv.invoice_id,
            'amount': inv.amount,
            'date_due': inv.date_due,
            'status': inv.status
        } for inv in invoices]
        filename = "invoices.csv"

    df = pd.DataFrame(data)
    csv_data = df.to_csv(index=False)

    return csv_data, 200, {
        'Content-Type': 'text/csv',
        'Content-Disposition': f'attachment; filename="{filename}"'
    }



@app.route('/chat', methods=['POST'])
def chat():
    """
    POST: Receive user message and respond with OpenAI Chat response
    """
    data = request.get_json()
    user_message = data.get('message', '').strip()
    if not user_message:
        return jsonify({'error': 'Empty message'}), 400

    try:
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant for telecom compliance."},
                {"role": "user", "content": user_message}
            ]
        )
        answer = response['choices'][0]['message']['content']
        return jsonify({"response": answer})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
if __name__ == '__main__':
    # Ensure tables exist before running
    with app.app_context():
        inspector = inspect(db.engine)
        if not inspector.has_table("invoice"):
            print("[INFO] Creating tables...")
            db.create_all()

    app.run(debug=True, host='0.0.0.0', port=5005)
