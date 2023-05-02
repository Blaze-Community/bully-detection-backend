from flask import Flask, request, redirect, jsonify,render_template, flash
import video
import audio
import os
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = 'super secret key'

cors = CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/', methods=['GET'])
def home_page():
    return render_template('home.html')

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

if __name__ == "__main__":
    print("Starting Python Flask Server For Video Classification")
    video.load_saved_video_artifacts()
    audio.load_saved_audio_artifacts()
    app.run(debug=True, host='0.0.0.0', port=33507)