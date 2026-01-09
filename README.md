# Cattle Disease Detection System

![Cattle Health AI](https://img.shields.io/badge/AI-Powered-blue)
![Flask](https://img.shields.io/badge/Flask-3.0.0-green)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1.0-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

An advanced AI-powered web application for detecting cattle diseases using Vision Transformer (ViT) deep learning models. Upload an image of your cattle and get instant diagnosis with confidence scores.

## 🌟 Features

- **AI-Powered Detection**: Uses Vision Transformer (ViT) model for accurate disease classification
- **Instant Results**: Get diagnosis in under 2 seconds
- **Interactive UI**: Modern, responsive design with drag-and-drop upload
- **Comprehensive Reports**: Track all diagnoses with detailed metadata
- **Admin Dashboard**: Visualize statistics and trends with interactive charts
- **Mobile Friendly**: Works seamlessly on all devices
- **Secure**: Admin authentication and session management

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Shashankxou/cattle-disease-detector.git
cd cattle-disease-detector
```

2. **Create virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Add your trained model**
   - Place your trained `cattle_disease_vit_model.pth` file in the `models/` directory
   - Update `models/class_names.json` with your disease classes
   - Update `models/model_config.json` with your model configuration

5. **Run the application**
```bash
python app.py
```

6. **Access the application**
   - Open your browser and navigate to `http://localhost:5000`
   - Admin login: username `admin`, password `admin123`

## 📁 Project Structure

```
cattle_disease_app/
├── app.py                          # Main Flask application
├── requirements.txt                # Python dependencies
├── database.db                     # SQLite database (auto-created)
├── models/
│   ├── cattle_disease_vit_model.pth   # Trained model weights
│   ├── class_names.json               # Disease class names
│   └── model_config.json              # Model configuration
├── static/
│   ├── css/
│   │   └── style.css               # Application styles
│   ├── js/
│   │   └── script.js               # JavaScript utilities
│   └── uploads/                    # Uploaded images (runtime)
└── templates/
    ├── base.html                   # Base template
    ├── home.html                   # Landing page
    ├── upload.html                 # Upload & diagnosis page
    ├── reports.html                # Reports listing
    └── admin.html                  # Admin dashboard
```

## 🎯 Usage

### For Farmers/Users

1. **Home Page**: Learn about the system and its features
2. **Diagnose**: Upload cattle images for instant disease detection
   - Drag & drop or browse to select image
   - Add optional metadata (Cattle ID, Location, Notes)
   - Get instant results with confidence scores
3. **Reports**: View all past diagnoses with filtering and search

### For Administrators

1. **Login**: Access admin dashboard with credentials
2. **Dashboard**: View statistics, charts, and trends
   - Total reports and disease distribution
   - Timeline of diagnoses
   - Average confidence scores
3. **Manage**: Monitor system usage and performance

## 🔧 Configuration

### Model Configuration

Edit `models/model_config.json`:
```json
{
    "image_size": 224,
    "model_type": "vit_b_16",
    "num_classes": 4,
    "pretrained": false
}
```

### Class Names

Edit `models/class_names.json`:
```json
["Healthy", "Foot-and-Mouth Disease", "Lumpy Skin Disease", "Mastitis"]
```

### Environment Variables

Set these for production:
```bash
export SECRET_KEY="your-secret-key-here"
export FLASK_ENV="production"
```

## 🚀 Deployment

### Deploy to Railway

1. Install Railway CLI:
```bash
npm install -g @railway/cli
```

2. Login and deploy:
```bash
railway login
railway init
railway up
```

### Deploy to Heroku

1. Create `Procfile`:
```
web: gunicorn app:app
```

2. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Deploy to Docker

1. Create `Dockerfile`:
```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["gunicorn", "-b", "0.0.0.0:5000", "app:app"]
```

2. Build and run:
```bash
docker build -t cattle-disease-detector .
docker run -p 5000:5000 cattle-disease-detector
```

## 📊 API Endpoints

- `GET /` - Home page
- `GET /upload` - Upload page
- `POST /upload` - Submit image for diagnosis
- `GET /reports` - View all reports
- `GET /report/<id>` - Get specific report details
- `GET /admin` - Admin dashboard
- `POST /admin/login` - Admin authentication
- `GET /api/stats` - Get statistics (JSON)

## 🛠️ Technology Stack

- **Backend**: Flask 3.0.0
- **ML Framework**: PyTorch 2.1.0, TorchVision 0.16.0
- **Model**: Vision Transformer (ViT-B/16)
- **Database**: SQLite
- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Charts**: Chart.js
- **Icons**: Font Awesome 6.4.0
- **Fonts**: Google Fonts (Inter)

## 🔒 Security

- Password hashing with SHA-256
- Session-based authentication
- CSRF protection
- File upload validation
- SQL injection prevention
- XSS protection

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Support

For support, email support@cattlehealth.ai or open an issue on GitHub.

## 🙏 Acknowledgments

- Vision Transformer (ViT) paper by Dosovitskiy et al.
- PyTorch team for the excellent framework
- Flask community for the web framework
- All contributors and testers

## 📈 Roadmap

- [ ] Multi-language support
- [ ] Mobile app (iOS/Android)
- [ ] Real-time video analysis
- [ ] Integration with veterinary systems
- [ ] Batch processing
- [ ] Export reports to PDF
- [ ] Email notifications
- [ ] Advanced analytics

---

Made with ❤️ for cattle health and welfare
