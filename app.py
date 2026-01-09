import os
import json
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import hashlib

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load model configuration
with open('models/model_config.json', 'r') as f:
    model_config = json.load(f)

with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Load the trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.vit_b_16(pretrained=False)
num_classes = len(class_names)
model.heads = torch.nn.Linear(model.heads.head.in_features, num_classes)
model.load_state_dict(torch.load('models/cattle_disease_vit_model.pth', map_location=device))
model = model.to(device)
model.eval()

# Image preprocessing
image_size = model_config.get('image_size', 224)
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Database initialization
def init_db():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Create reports table
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  filepath TEXT NOT NULL,
                  prediction TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  notes TEXT,
                  cattle_id TEXT,
                  location TEXT)''')
    
    # Create users table for admin
    c.execute('''CREATE TABLE IF NOT EXISTS users
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  username TEXT UNIQUE NOT NULL,
                  password_hash TEXT NOT NULL,
                  role TEXT DEFAULT 'user')''')
    
    # Create default admin if not exists
    admin_hash = hashlib.sha256('admin123'.encode()).hexdigest()
    try:
        c.execute("INSERT INTO users (username, password_hash, role) VALUES (?, ?, ?)",
                  ('admin', admin_hash, 'admin'))
    except sqlite3.IntegrityError:
        pass
    
    conn.commit()
    conn.close()

init_db()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def predict_image(image_path):
    """Predict disease from image"""
    try:
        image = Image.open(image_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
        predicted_class = class_names[predicted.item()]
        confidence_score = confidence.item() * 100
        
        # Get all class probabilities
        all_probs = {class_names[i]: float(probabilities[0][i] * 100) 
                     for i in range(len(class_names))}
        
        return {
            'prediction': predicted_class,
            'confidence': round(confidence_score, 2),
            'all_probabilities': all_probs
        }
    except Exception as e:
        return {'error': str(e)}

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Predict
            result = predict_image(filepath)
            
            if 'error' in result:
                os.remove(filepath)
                return jsonify({'error': result['error']}), 500
            
            # Save to database
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("""INSERT INTO reports (filename, filepath, prediction, confidence, 
                         cattle_id, location, notes) 
                         VALUES (?, ?, ?, ?, ?, ?, ?)""",
                      (filename, filepath, result['prediction'], result['confidence'],
                       request.form.get('cattle_id', ''),
                       request.form.get('location', ''),
                       request.form.get('notes', '')))
            report_id = c.lastrowid
            conn.commit()
            conn.close()
            
            return jsonify({
                'success': True,
                'report_id': report_id,
                'filename': filename,
                'filepath': filepath,
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'all_probabilities': result['all_probabilities']
            })
        
        return jsonify({'error': 'Invalid file type'}), 400
    
    return render_template('upload.html')

@app.route('/reports')
def reports():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes 
                 FROM reports ORDER BY timestamp DESC LIMIT 100""")
    reports_data = c.fetchall()
    conn.close()
    
    reports_list = []
    for row in reports_data:
        reports_list.append({
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'prediction': row[3],
            'confidence': row[4],
            'timestamp': row[5],
            'cattle_id': row[6],
            'location': row[7],
            'notes': row[8]
        })
    
    return render_template('reports.html', reports=reports_list)

@app.route('/report/<int:report_id>')
def report_detail(report_id):
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes 
                 FROM reports WHERE id = ?""", (report_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        report = {
            'id': row[0],
            'filename': row[1],
            'filepath': row[2],
            'prediction': row[3],
            'confidence': row[4],
            'timestamp': row[5],
            'cattle_id': row[6],
            'location': row[7],
            'notes': row[8]
        }
        return jsonify(report)
    return jsonify({'error': 'Report not found'}), 404

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        
        conn = sqlite3.connect('database.db')
        c = conn.cursor()
        c.execute("SELECT id, role FROM users WHERE username = ? AND password_hash = ?",
                  (username, password_hash))
        user = c.fetchone()
        conn.close()
        
        if user:
            session['user_id'] = user[0]
            session['role'] = user[1]
            flash('Login successful!', 'success')
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('admin.html', login_page=True)

@app.route('/admin')
def admin():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect(url_for('admin_login'))
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    # Get statistics
    c.execute("SELECT COUNT(*) FROM reports")
    total_reports = c.fetchone()[0]
    
    c.execute("SELECT prediction, COUNT(*) FROM reports GROUP BY prediction")
    disease_stats = c.fetchall()
    
    c.execute("""SELECT AVG(confidence) FROM reports 
                 WHERE prediction != 'Healthy'""")
    avg_confidence = c.fetchone()[0] or 0
    
    conn.close()
    
    return render_template('admin.html', 
                          total_reports=total_reports,
                          disease_stats=disease_stats,
                          avg_confidence=round(avg_confidence, 2))

@app.route('/admin/logout')
def admin_logout():
    session.clear()
    flash('Logged out successfully', 'success')
    return redirect(url_for('home'))

@app.route('/api/stats')
def api_stats():
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    
    c.execute("SELECT COUNT(*) FROM reports")
    total = c.fetchone()[0]
    
    c.execute("SELECT prediction, COUNT(*) FROM reports GROUP BY prediction")
    by_disease = dict(c.fetchall())
    
    c.execute("""SELECT DATE(timestamp) as date, COUNT(*) 
                 FROM reports 
                 GROUP BY DATE(timestamp) 
                 ORDER BY date DESC LIMIT 30""")
    daily_reports = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'total_reports': total,
        'by_disease': by_disease,
        'daily_reports': daily_reports
    })

@app.route('/delete_report/<int:report_id>', methods=['POST'])
def delete_report(report_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return jsonify({'error': 'Unauthorized'}), 403
    
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("SELECT filepath FROM reports WHERE id = ?", (report_id,))
    row = c.fetchone()
    
    if row:
        filepath = row[0]
        if os.path.exists(filepath):
            os.remove(filepath)
        
        c.execute("DELETE FROM reports WHERE id = ?", (report_id,))
        conn.commit()
        conn.close()
        return jsonify({'success': True})
    
    conn.close()
    return jsonify({'error': 'Report not found'}), 404

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
