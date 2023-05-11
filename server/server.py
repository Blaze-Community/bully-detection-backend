import requests
from flask import Flask, request, redirect, jsonify,render_template, flash
import video
import audio
from flask_cors import CORS
import urllib.request

app = Flask(__name__)
app.secret_key = 'super secret key'

cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET','POST'])
def home_page():
    
    if request.method == 'GET':
        response = requests.get('https://college-app-backend-production.up.railway.app/api/bully')
        data = response.json()
        bully_list = []
        if data['success'] and len(data['list']) > 0:
            bully_list = data['list']
        return render_template('home.html',bully_list = bully_list)
    
    if request.method == 'POST':
        data = request.get_json()
        _id = data.get('_id')
        video_uri = data.get('URI')
        try:
            video_path = '../static/uploads/video/temp.mp4'
            print("Downloading starts...\n")
            urllib.request.urlretrieve(video_uri, video_path)
            print("Download completed..!!")

            result = video.classify_video(video_path)
            result = result[0]
            data = {     
                "bully_id":_id,
                "percentage": int(float(result['confidence'])*100),
                "result": "VIOLENCE" if result['action_predicted'] == 'violence' else 'NON-VIOLENCE'
            }
            response = requests.post('https://college-app-backend-production.up.railway.app/api/bully',data=data)
            data = response.json()
            if data['success']:
                return jsonify(data),200
            else: 
                return jsonify(data),400
        
        except Exception as e:
            print(e)
            return redirect(request.url) 

@app.route('/predict-audio', methods=['POST', 'GET'])
def classify_audio():
    if request.method=='GET':
        return render_template('predict_audio.html')
    if request.method=='POST':
        if 'audiofile' not in request.files:
            flash('No file part')
            return redirect(request.url)
        audiofile1 = request.files['audiofile']
        if audiofile1.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        else:
            pred = 'no file found'
            audio_path = audio.upload_audio(audiofile1)
            result = audio.prediction(audio_path)
            data = { "result" : result }
            response = jsonify(data)
            print(response)
            return response, 200

@app.route('/predict-video', methods=['POST', 'GET'])
def classify_video():
    if request.method=='GET':
        return render_template('predict_video.html')

    if request.method=='POST':
        if 'video_file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['video_file']
        if file.filename == '':
            flash('No image selected for uploading')
            return redirect(request.url)
        else:
            video_path = video.upload_video(file)
            result = video.classify_video(video_path)
            data = { "result" : result }
            response = jsonify(data)
            print(response)
            return response, 200

@app.after_request
def adding_header_content(head):
    head.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    head.headers["Pragma"] = "no-cache"
    head.headers["Expires"] = "0"
    head.headers['Cache-Control'] = 'public, max-age=0'
    return head

if __name__ == "__main__":
    print("Starting Python Flask Server For Video Classification")
    video.load_saved_video_artifacts()
    audio.load_saved_audio_artifacts()
    app.run(debug=True, host='0.0.0.0', port=33507)