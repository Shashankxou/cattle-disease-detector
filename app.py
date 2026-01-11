import os
import json
import sqlite3
from datetime import datetime
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, session, send_file
from PIL import Image
import torch
import torchvision.transforms as transforms
from torchvision import models
import hashlib
from io import BytesIO
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'your-secret-key-change-in-production')
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg', 'gif'}

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load translations
with open('translations.json', 'r', encoding='utf-8') as f:
    translations = json.load(f)

# Load model configuration
with open('models/model_config.json', 'r') as f:
    model_config = json.load(f)

with open('models/class_names.json', 'r') as f:
    class_names = json.load(f)

# Treatment recommendations database
TREATMENT_RECOMMENDATIONS = {
    "Healthy": {
        "en": "Your cattle is healthy! Continue regular care and monitoring.",
        "hi": "आपका पशु स्वस्थ है! नियमित देखभाल और निगरानी जारी रखें।",
        "ta": "உங்கள் கால்நடை ஆரோக்கியமாக உள்ளது! வழக்கமான பராமரிப்பு மற்றும் கண்காணிப்பைத் தொடரவும்.",
        "kn": "ನಿಮ್ಮ ಜಾನುವಾರು ಆರೋಗ್ಯಕರವಾಗಿದೆ! ನಿಯಮಿತ ಆರೈಕೆ ಮತ್ತು ಮೇಲ್ವಿಚಾರಣೆಯನ್ನು ಮುಂದುವರಿಸಿ."
    },
    "Foot-and-Mouth Disease": {
        "en": "URGENT: Isolate immediately. Contact veterinarian. Provide soft feed and clean water. Disinfect area.",
        "hi": "तत्काल: तुरंत अलग करें। पशु चिकित्सक से संपर्क करें। नरम चारा और साफ पानी दें। क्षेत्र को कीटाणुरहित करें।",
        "ta": "அவசரம்: உடனடியாக தனிமைப்படுத்தவும். கால்நடை மருத்துவரை தொடர்பு கொள்ளவும். மென்மையான உணவு மற்றும் சுத்தமான தண்ணீர் வழங்கவும்.",
        "kn": "ತುರ್ತು: ತಕ್ಷಣ ಪ್ರತ್ಯೇಕಿಸಿ. ಪಶುವೈದ್ಯರನ್ನು ಸಂಪರ್ಕಿಸಿ. ಮೃದುವಾದ ಆಹಾರ ಮತ್ತು ಶುದ್ಧ ನೀರು ಒದಗಿಸಿ."
    },
    "Lumpy Skin Disease": {
        "en": "Isolate cattle. Apply antiseptic to lesions. Provide nutritious feed. Consult vet for vaccination.",
        "hi": "पशु को अलग करें। घावों पर एंटीसेप्टिक लगाएं। पौष्टिक चारा दें। टीकाकरण के लिए पशु चिकित्सक से परामर्श करें।",
        "ta": "கால்நடைகளை தனிமைப்படுத்தவும். காயங்களுக்கு கிருமி நாசினி தடவவும். சத்தான உணவு வழங்கவும்.",
        "kn": "ಜಾನುವಾರುಗಳನ್ನು ಪ್ರತ್ಯೇಕಿಸಿ. ಗಾಯಗಳಿಗೆ ನಂಜುನಿರೋಧಕವನ್ನು ಅನ್ವಯಿಸಿ. ಪೌಷ್ಟಿಕ ಆಹಾರವನ್ನು ಒದಗಿಸಿ."
    },
    "Mastitis": {
        "en": "Clean udder with warm water. Apply prescribed ointment. Milk out infected quarter. Antibiotic treatment needed.",
        "hi": "गर्म पानी से थन साफ करें। निर्धारित मरहम लगाएं। संक्रमित भाग से दूध निकालें। एंटीबायोटिक उपचार की आवश्यकता है।",
        "ta": "மடியை சூடான நீரில் சுத்தம் செய்யவும். பரிந்துரைக்கப்பட்ட களிம்பு தடவவும். நுண்ணுயிர் எதிர்ப்பு சிகிச்சை தேவை.",
        "kn": "ಕೆಚ್ಚಲನ್ನು ಬೆಚ್ಚಗಿನ ನೀರಿನಿಂದ ಸ್ವಚ್ಛಗೊಳಿಸಿ. ಸೂಚಿಸಿದ ಮುಲಾಮು ಹಚ್ಚಿ. ಪ್ರತಿಜೀವಕ ಚಿಕಿತ್ಸೆ ಅಗತ್ಯವಿದೆ."
    }
}

# Load the trained model
try:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = models.vit_b_16(pretrained=False)
    num_classes = len(class_names)
    model.heads = torch.nn.Linear(model.heads.head.in_features, num_classes)
    model.load_state_dict(torch.load('models/cattle_disease_vit_model.pth', map_location=device))
    model = model.to(device)
    model.eval()
    MODEL_LOADED = True
except Exception as e:
    print(f"⚠️  WARNING: Model not loaded - {e}")
    MODEL_LOADED = False

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
    
    # Create reports table with language field
    c.execute('''CREATE TABLE IF NOT EXISTS reports
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT NOT NULL,
                  filepath TEXT NOT NULL,
                  prediction TEXT NOT NULL,
                  confidence REAL NOT NULL,
                  timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                  notes TEXT,
                  cattle_id TEXT,
                  location TEXT,
                  language TEXT DEFAULT 'en')''')
    
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

def get_translation(key, lang='en'):
    """Get translated text"""
    return translations.get(lang, {}).get(key, translations['en'].get(key, key))

def get_treatment_recommendation(disease, lang='en'):
    """Get treatment recommendation for disease"""
    return TREATMENT_RECOMMENDATIONS.get(disease, {}).get(lang, "Consult a veterinarian for proper treatment.")

def predict_image(image_path):
    """Predict disease from image"""
    if not MODEL_LOADED:
        return {'error': 'Model not loaded. Please add trained model file.'}
    
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

def generate_pdf_report(report_data, lang='en'):
    """Generate PDF report"""
    buffer = BytesIO()
    p = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter
    
    # Title
    p.setFont("Helvetica-Bold", 24)
    p.drawString(50, height - 50, get_translation('app_name', lang))
    
    # Report details
    p.setFont("Helvetica", 12)
    y = height - 100
    
    p.drawString(50, y, f"{get_translation('date', lang)}: {report_data['timestamp']}")
    y -= 30
    p.drawString(50, y, f"{get_translation('cattle_id', lang)}: {report_data.get('cattle_id', 'N/A')}")
    y -= 30
    p.drawString(50, y, f"{get_translation('location', lang)}: {report_data.get('location', 'N/A')}")
    y -= 30
    
    # Prediction
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, f"{get_translation('prediction', lang)}:")
    p.setFont("Helvetica", 14)
    p.drawString(200, y, report_data['prediction'])
    y -= 30
    
    # Confidence
    p.setFont("Helvetica-Bold", 16)
    p.drawString(50, y, f"{get_translation('confidence', lang)}:")
    p.setFont("Helvetica", 14)
    p.drawString(200, y, f"{report_data['confidence']}%")
    y -= 50
    
    # Treatment
    p.setFont("Helvetica-Bold", 14)
    p.drawString(50, y, f"{get_translation('treatment', lang)}:")
    y -= 20
    p.setFont("Helvetica", 10)
    treatment = get_treatment_recommendation(report_data['prediction'], lang)
    
    # Wrap text
    words = treatment.split()
    line = ""
    for word in words:
        if len(line + word) < 80:
            line += word + " "
        else:
            p.drawString(50, y, line)
            y -= 15
            line = word + " "
    if line:
        p.drawString(50, y, line)
    
    p.showPage()
    p.save()
    buffer.seek(0)
    return buffer

@app.route('/')
def home():
    lang = request.args.get('lang', 'en')
    return render_template('home.html', lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/set_language/<lang>')
def set_language(lang):
    session['language'] = lang
    return redirect(request.referrer or url_for('home'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    lang = session.get('language', 'en')
    
    if request.method == 'POST':
        if 'file' not in request.files:
            return jsonify({'error': get_translation('error_upload', lang)}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': get_translation('error_upload', lang)}), 400
        
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
            
            # Get treatment recommendation
            treatment = get_treatment_recommendation(result['prediction'], lang)
            
            # Save to database
            conn = sqlite3.connect('database.db')
            c = conn.cursor()
            c.execute("""INSERT INTO reports (filename, filepath, prediction, confidence, 
                         cattle_id, location, notes, language) 
                         VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                      (filename, filepath, result['prediction'], result['confidence'],
                       request.form.get('cattle_id', ''),
                       request.form.get('location', ''),
                       request.form.get('notes', ''),
                       lang))
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
                'all_probabilities': result['all_probabilities'],
                'treatment': treatment
            })
        
        return jsonify({'error': get_translation('error_upload', lang)}), 400
    
    return render_template('upload.html', lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/reports')
def reports():
    lang = session.get('language', 'en')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes, language
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
            'notes': row[8],
            'language': row[9] if len(row) > 9 else 'en'
        })
    
    return render_template('reports.html', reports=reports_list, lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/report/<int:report_id>')
def report_detail(report_id):
    lang = session.get('language', 'en')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes, language
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
            'notes': row[8],
            'language': row[9] if len(row) > 9 else 'en',
            'treatment': get_treatment_recommendation(row[3], lang)
        }
        return jsonify(report)
    return jsonify({'error': 'Report not found'}), 404

@app.route('/download_pdf/<int:report_id>')
def download_pdf(report_id):
    lang = session.get('language', 'en')
    conn = sqlite3.connect('database.db')
    c = conn.cursor()
    c.execute("""SELECT id, filename, filepath, prediction, confidence, 
                 timestamp, cattle_id, location, notes 
                 FROM reports WHERE id = ?""", (report_id,))
    row = c.fetchone()
    conn.close()
    
    if row:
        report_data = {
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
        
        pdf_buffer = generate_pdf_report(report_data, lang)
        return send_file(
            pdf_buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'cattle_report_{report_id}.pdf'
        )
    
    return jsonify({'error': 'Report not found'}), 404

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    lang = session.get('language', 'en')
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
            flash(get_translation('success_upload', lang), 'success')
            return redirect(url_for('admin'))
        else:
            flash('Invalid credentials', 'error')
    
    return render_template('admin.html', login_page=True, lang=lang, t=lambda k: get_translation(k, lang))

@app.route('/admin')
def admin():
    lang = session.get('language', 'en')
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
    
    # Get daily reports for chart
    c.execute("""SELECT DATE(timestamp) as date, COUNT(*) 
                 FROM reports 
                 GROUP BY DATE(timestamp) 
                 ORDER BY date DESC LIMIT 30""")
    daily_reports = c.fetchall()
    
    # Get monthly trends
    c.execute("""SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) 
                 FROM reports 
                 GROUP BY month 
                 ORDER BY month DESC LIMIT 12""")
    monthly_reports = c.fetchall()
    
    conn.close()
    
    return render_template('admin.html', 
                          total_reports=total_reports,
                          disease_stats=disease_stats,
                          avg_confidence=round(avg_confidence, 2),
                          daily_reports=daily_reports,
                          monthly_reports=monthly_reports,
                          lang=lang,
                          t=lambda k: get_translation(k, lang))

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
    
    c.execute("""SELECT strftime('%Y-%m', timestamp) as month, COUNT(*) 
                 FROM reports 
                 GROUP BY month 
                 ORDER BY month DESC LIMIT 12""")
    monthly_reports = c.fetchall()
    
    conn.close()
    
    return jsonify({
        'total_reports': total,
        'by_disease': by_disease,
        'daily_reports': daily_reports,
        'monthly_reports': monthly_reports
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
