# 🚀 AI-Based IDS - Startup Guide

A complete step-by-step guide to run the AI-Based Intrusion Detection System.

---

## 📋 Prerequisites

Before starting, ensure you have the following installed:

### Required Software

| Software | Version | Download |
|----------|---------|----------|
| **Node.js** | 18+ or 20+ | [nodejs.org](https://nodejs.org/) |
| **Python** | 3.11+ | [python.org](https://www.python.org/) |
| **Git** | Latest | [git-scm.com](https://git-scm.com/) |
| **Bun** (optional) | Latest | [bun.sh](https://bun.sh/) |

### Accounts Needed

| Service | Purpose | Sign Up |
|---------|---------|---------|
| **Neon** | PostgreSQL database | [neon.tech](https://neon.tech/) |
| **Google AI Studio** | Gemini API key | [aistudio.google.com](https://aistudio.google.com/) |

---

## 🔧 Step 1: Clone the Repository

```bash
# Clone the repository
git clone https://github.com/KIRITO-AR/Major-Project.git

# Navigate to project directory
cd Major-Project
```

---

## 🗄️ Step 2: Set Up the Database (Neon PostgreSQL)

### 2.1 Create Neon Account
1. Go to [neon.tech](https://neon.tech/)
2. Sign up with GitHub or email
3. Create a new project

### 2.2 Get Database URL
1. In Neon dashboard, click on your project
2. Click "Connection Details"
3. Copy the "Connection string" (starts with `postgresql://...`)

### 2.3 Create Environment File
```bash
# In the project root directory
cp .env.example .env
```

Or create `.env` manually:
```env
# Database
DATABASE_URL="postgresql://username:password@host/database?sslmode=require"

# Google Gemini AI
GEMINI_API_KEY="your-gemini-api-key"

# Python Backend URL (for Next.js to connect)
PYTHON_API_URL="http://localhost:8000"
```

---

## 🐍 Step 3: Set Up Python ML Backend

### 3.1 Navigate to Python Directory
```bash
cd python-ml
```

### 3.2 Create Virtual Environment

**Windows (PowerShell):**
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**
```cmd
python -m venv venv
venv\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3.3 Install Python Dependencies
```bash
pip install -r requirements.txt
```

> ⚠️ **Note**: TensorFlow installation may take a few minutes.

### 3.4 Start Python Backend
```bash
uvicorn app.main:app --reload --port 8000
```

### 3.5 Verify Python Backend
Open in browser: **http://localhost:8000**

You should see:
```json
{
  "name": "AI-Based IDS ML Backend",
  "version": "1.0.0",
  "status": "running"
}
```

**API Documentation:** http://localhost:8000/docs

> 💡 **Keep this terminal running!** Open a new terminal for the next steps.

---

## ⚡ Step 4: Set Up Next.js Frontend

### 4.1 Open New Terminal & Navigate to Project Root
```bash
cd Major-Project
```

### 4.2 Install Node.js Dependencies

**Using npm:**
```bash
npm install
```

**Using Bun (faster):**
```bash
bun install
```

### 4.3 Generate Prisma Client
```bash
npx prisma generate
```

### 4.4 Push Database Schema
```bash
npx prisma db push
```

### 4.5 Start Next.js Development Server

**Using npm:**
```bash
npm run dev
```

**Using Bun:**
```bash
bun dev
```

### 4.6 Verify Frontend
Open in browser: **http://localhost:3000**

You should see the AI-Based IDS Dashboard!

---

## 🧩 Step 5: Install Chrome Extension (Optional)

### 5.1 Open Chrome Extensions
1. Open Chrome browser
2. Go to: `chrome://extensions/`
3. Enable **"Developer mode"** (toggle in top-right)

### 5.2 Load Extension
1. Click **"Load unpacked"**
2. Navigate to: `Major-Project/chrome-extension`
3. Select the folder

### 5.3 Verify Extension
- You should see the **"AI-Based IDS Monitor"** extension
- Click the extension icon to open the popup
- It will connect to `localhost:3000`

---

## ✅ Step 6: Verify Everything is Working

### Running Services Checklist

| Service | URL | Status |
|---------|-----|--------|
| Python ML Backend | http://localhost:8000 | 🟢 Running |
| Python API Docs | http://localhost:8000/docs | 🟢 Available |
| Next.js Frontend | http://localhost:3000 | 🟢 Running |
| Chrome Extension | Browser toolbar | 🟢 Installed |

### Test the System

1. **Open Dashboard**: http://localhost:3000
2. **Click "Analyze Traffic"** button
3. **Check Detection Feed** - should show detection results
4. **Check ML Models tab** - should show model metrics
5. **Try Auto-Response tab** - configure blocking settings
6. **Try Training tab** - view/export training data

---

## 🔄 Quick Start Commands (After Initial Setup)

Once everything is set up, use these commands to start the project:

### Terminal 1: Python Backend
```bash
cd Major-Project/python-ml
.\venv\Scripts\activate  # Windows
# source venv/bin/activate  # macOS/Linux
uvicorn app.main:app --reload --port 8000
```

### Terminal 2: Next.js Frontend
```bash
cd Major-Project
bun dev
# or: npm run dev
```

---

## 🐳 Alternative: Docker Setup

If you prefer Docker, you can run the Python backend in a container:

### Build Docker Image
```bash
cd python-ml
docker build -t ids-ml-backend .
```

### Run Container
```bash
docker run -p 8000:8000 ids-ml-backend
```

---

## 🛠️ Troubleshooting

### Common Issues

#### ❌ "Python not found"
- Ensure Python 3.11+ is installed
- Add Python to your system PATH
- Try `python3` instead of `python`

#### ❌ "pip install fails"
- Update pip: `python -m pip install --upgrade pip`
- Try: `pip install --user -r requirements.txt`

#### ❌ "TensorFlow installation error"
- Ensure you have Python 3.11 (not 3.12+)
- Try: `pip install tensorflow-cpu` for lighter version

#### ❌ "Cannot connect to database"
- Check `DATABASE_URL` in `.env`
- Ensure Neon database is active
- Check if SSL is required (`?sslmode=require`)

#### ❌ "Prisma errors"
- Run: `npx prisma generate`
- Run: `npx prisma db push`

#### ❌ "Port already in use"
- Kill the process using the port:
  - Windows: `netstat -ano | findstr :8000` then `taskkill /PID <PID> /F`
  - macOS/Linux: `lsof -i :8000` then `kill -9 <PID>`

#### ❌ "Chrome extension not connecting"
- Ensure Next.js is running on port 3000
- Check extension permissions
- Reload the extension

---

## 📁 Project URLs Summary

| Component | URL |
|-----------|-----|
| **Dashboard** | http://localhost:3000 |
| **Python API** | http://localhost:8000 |
| **API Docs (Swagger)** | http://localhost:8000/docs |
| **API Docs (ReDoc)** | http://localhost:8000/redoc |
| **Health Check** | http://localhost:8000/health |

---

## 🎯 Next Steps

After startup, you can:

1. **Explore the Dashboard** - View real-time detections
2. **Test RLHF** - Provide feedback on detections to improve weights
3. **Configure Auto-Response** - Set up automatic IP blocking
4. **Export Training Data** - Download detection data for analysis
5. **Use AI Assistant** - Chat with Gemini for threat analysis

---

## 📞 Need Help?

- Check the `implementation.md` for detailed architecture
- Review `python-ml/README.md` for Python backend details
- Open an issue on GitHub for bugs

---

*Happy Detecting! 🛡️*
