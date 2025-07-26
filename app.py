import os
import uuid
import re
import csv
import requests
import PyPDF2
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.secret_key = 'nexuscore_supersecretkey'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['DATASET_FOLDER'] = 'datasets'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB
app.config['ALLOWED_EXTENSIONS'] = {'pdf'}

# Groq API
import os
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

GROQ_API_KEY = os.getenv('GROQ_API_KEY')
GROQ_API_URL = 'https://api.groq.com/openai/v1/chat/completions'
GROQ_HEADERS = {
    'Authorization': f'Bearer {GROQ_API_KEY}',
    'Content-Type': 'application/json'
}

# Ensure directories exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['DATASET_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def extract_text_from_pdf(file_path):
    """Extract text from PDF using PyPDF2"""
    text = ""
    try:
        with open(file_path, 'rb') as f:
            reader = PyPDF2.PdfReader(f)
            for page in reader.pages:
                text += page.extract_text() + "\n"
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")
        return f"Error extracting text: {str(e)}"
    
    return text

def generate_with_groq(prompt, rows=10):
    """Generate dataset using Groq API with improved prompting"""
    system_prompt = (
        "You are a dataset generation expert. Create a dataset in CSV format based on the user's request. "
        f"Generate exactly {rows} rows of data. "
        "Include only the CSV data with a header row. Do not include any explanations or additional text. "
        "Ensure the dataset has proper column headers and valid data."
    )
    
    data = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.3,
        "max_tokens": 4000,
        "stop": ["```"]
    }
   
    try:
        response = requests.post(GROQ_API_URL, headers=GROQ_HEADERS, json=data, timeout=60)
        response.raise_for_status()
        result = response.json()
        return result['choices'][0]['message']['content']
    except Exception as e:
        print(f"Groq API error: {str(e)}")
        return None

def process_groq_response(groq_response):
    """Extract CSV data from Groq response with robust parsing"""
    # Find the start of the CSV data
    lines = groq_response.split('\n')
    csv_lines = []
    in_csv = False
    
    for line in lines:
        # Skip markdown code blocks
        if line.strip().startswith('```'):
            in_csv = not in_csv
            continue
            
        if in_csv or (',' in line and any(char.isalpha() for char in line)):
            csv_lines.append(line)
    
    # Join valid CSV lines
    csv_data = '\n'.join(csv_lines).strip()
    
    # Ensure we have valid CSV
    if not csv_data or len(csv_data.split('\n')) < 2:
        return None
    
    return csv_data

def parse_csv_for_preview(csv_data, max_rows=5):
    """Parse CSV data for preview with error handling"""
    try:
        # Parse CSV
        lines = csv_data.split('\n')
        headers = []
        rows = []
        
        # Use CSV reader for proper parsing
        reader = csv.reader(lines)
        for i, row in enumerate(reader):
            if i == 0:
                headers = row
            elif i <= max_rows:
                rows.append(row)
            else:
                break
                
        return headers, rows
    except Exception as e:
        print(f"CSV parsing error: {str(e)}")
        return None, None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_dataset():
    # Get form data
    prompt = request.form.get('prompt', '')
    dataset_name = request.form.get('dataset_name', 'Untitled Dataset')
    rows = int(request.form.get('rows', 10))
    file = request.files.get('file')
    
    if not prompt:
        return jsonify({'error': 'Prompt is required'}), 400
    
    if rows < 1 or rows > 1000:
        return jsonify({'error': 'Number of rows must be between 1 and 1000'}), 400
    
    # Handle file upload if present
    pdf_text = ""
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        pdf_text = extract_text_from_pdf(file_path)
        print(f"Extracted {len(pdf_text)} characters from PDF")
    
    # Combine prompt with PDF text
    full_prompt = f"Dataset Name: {dataset_name}\n\n" if dataset_name else ""
    full_prompt += f"User Prompt: {prompt}\n\n"
    if pdf_text:
        full_prompt += f"PDF Content:\n{pdf_text[:10000]}\n\n"  # Limit to 10k chars
    
    print(f"Sending prompt to Groq API: {full_prompt[:500]}...")  # Log first 500 chars
    
    # Generate dataset with Groq API
    groq_response = generate_with_groq(full_prompt, rows)
    
    if not groq_response:
        return jsonify({'error': 'Failed to generate dataset with Groq API'}), 500
    
    print(f"Received Groq response: {groq_response[:500]}...")  # Log first 500 chars
    
    # Process the Groq response to extract CSV data
    csv_data = process_groq_response(groq_response)
    
    if not csv_data:
        return jsonify({'error': 'Failed to extract CSV data from Groq response'}), 500
    
    print(f"Generated CSV data:\n{csv_data[:500]}...")  # Log first 500 chars
    
    # Generate dataset ID
    dataset_id = str(uuid.uuid4())
    dataset_path = os.path.join(app.config['DATASET_FOLDER'], f'{dataset_id}.csv')
    
    # Save CSV to file
    with open(dataset_path, 'w', encoding='utf-8') as f:
        f.write(csv_data)
    
    # Parse CSV for preview
    headers, rows = parse_csv_for_preview(csv_data)
    if not headers or not rows:
        return jsonify({'error': 'Generated dataset is invalid'}), 500
    
    return jsonify({
        'dataset_id': dataset_id,
        'name': dataset_name,
        'headers': headers,
        'rows': rows
    })

@app.route('/dataset/download/<dataset_id>', methods=['GET'])
def download_dataset(dataset_id):
    dataset_path = os.path.join(app.config['DATASET_FOLDER'], f'{dataset_id}.csv')
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404
        
    # Get custom filename if provided
    filename = request.args.get('name', f'dataset_{dataset_id}.csv')
    
    return send_file(
        dataset_path,
        as_attachment=True,
        download_name=filename
    )

@app.route('/dataset/preview/<dataset_id>', methods=['GET'])
def preview_dataset(dataset_id):
    dataset_path = os.path.join(app.config['DATASET_FOLDER'], f'{dataset_id}.csv')
    if not os.path.exists(dataset_path):
        return jsonify({'error': 'Dataset not found'}), 404
    
    # Read CSV for preview
    headers = []
    rows = []
    try:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            headers = next(reader, [])
            for i, row in enumerate(reader):
                if i < 5:  # Only get first 5 rows for preview
                    rows.append(row)
    except Exception as e:
        return jsonify({'error': f'Error reading dataset: {str(e)}'}), 500
    
    return jsonify({
        'dataset_id': dataset_id,
        'headers': headers,
        'rows': rows
    })

if __name__ == '__main__':
    app.run(debug=True, port=5000)
