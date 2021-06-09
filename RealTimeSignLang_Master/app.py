# -*- coding: utf-8 -*-

from scripts import tabledef
from scripts import forms
from scripts import helpers
from flask import Flask, redirect, url_for, render_template, request, session, Response, redirect
import json
import sys
import os

app = Flask(__name__)
app.secret_key = os.urandom(12)  # Generic key for dev purposes only
# app.config['SERVER_NAME'] = "example.com"

# Heroku
#from flask_heroku import Heroku
#heroku = Heroku(app)

# ======== Routing =========================================================== #
# -------- Login ------------------------------------------------------------- #









# from flask import Flask, render_template, Response
# import cv2

# import os

# CUSTOM_MODEL_NAME = 'my_ssd_mobnet' 
# PRETRAINED_MODEL_NAME = 'ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8'
# PRETRAINED_MODEL_URL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz'
# TF_RECORD_SCRIPT_NAME = 'generate_tfrecord.py'
# LABEL_MAP_NAME = 'label_map.pbtxt'

# paths = {
#     'WORKSPACE_PATH': os.path.join('Tensorflow', 'workspace'),
#     'SCRIPTS_PATH': os.path.join('Tensorflow','scripts'),
#     'APIMODEL_PATH': os.path.join('Tensorflow','models'),
#     'ANNOTATION_PATH': os.path.join('Tensorflow', 'workspace','annotations'),
#     'IMAGE_PATH': os.path.join('Tensorflow', 'workspace','images'),
#     'MODEL_PATH': os.path.join('Tensorflow', 'workspace','models'),
#     'PRETRAINED_MODEL_PATH': os.path.join('Tensorflow', 'workspace','pre-trained-models'),
#     'CHECKPOINT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME), 
#     'OUTPUT_PATH': os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'export'), 
#     'TFJS_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfjsexport'), 
#     'TFLITE_PATH':os.path.join('Tensorflow', 'workspace','models',CUSTOM_MODEL_NAME, 'tfliteexport'), 
#     'PROTOC_PATH':os.path.join('Tensorflow','protoc')
#  }

# files = {
#     'PIPELINE_CONFIG':os.path.join('Tensorflow', 'workspace','models', CUSTOM_MODEL_NAME, 'pipeline.config'),
#     'TF_RECORD_SCRIPT': os.path.join(paths['SCRIPTS_PATH'], TF_RECORD_SCRIPT_NAME), 
#     'LABELMAP': os.path.join(paths['ANNOTATION_PATH'], LABEL_MAP_NAME)
# }

# labels = [{'name':'A', 'id':1}, 
#           {'name':'B', 'id':2}, 
#           {'name':'C', 'id':3}, 
#           {'name':'D', 'id':4}, 
#           {'name':'E', 'id':5}, 
#           {'name':'F', 'id':6},
#           {'name':'G', 'id':7},
#           {'name':'H', 'id':8},
#           {'name':'I', 'id':9},
#           {'name':'J', 'id':10},
#           {'name':'ILOVEYOU', 'id':11}, 
#           {'name':'L', 'id':12}, 
#           {'name':'O', 'id':13},
#           {'name':'Q', 'id':14},
#           {'name':'Y', 'id':15} ]

# with open(files['LABELMAP'], 'w') as f:
#     for label in labels:
#         f.write('item { \n')
#         f.write('\tname:\'{}\'\n'.format(label['name']))
#         f.write('\tid:{}\n'.format(label['id']))
#         f.write('}\n')

# import tensorflow as tf
# from object_detection.utils import config_util
# from object_detection.protos import pipeline_pb2
# from google.protobuf import text_format

# import os
# import tensorflow as tf
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder
# from object_detection.utils import config_util

# # Load pipeline config and build a detection model
# configs = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])
# detection_model = model_builder.build(model_config=configs['model'], is_training=False)

# # Restore checkpoint
# ckpt = tf.compat.v2.train.Checkpoint(model=detection_model)
# ckpt.restore(os.path.join(paths['CHECKPOINT_PATH'], 'ckpt-3')).expect_partial()

# @tf.function
# def detect_fn(image):
#     image, shapes = detection_model.preprocess(image)
#     prediction_dict = detection_model.predict(image, shapes)
#     detections = detection_model.postprocess(prediction_dict, shapes)
#     return detections
    
# import cv2 
# import numpy as np
# from matplotlib import pyplot as plt
# # %matplotlib inline

# category_index = label_map_util.create_category_index_from_labelmap(files['LABELMAP'])

# cap = cv2.VideoCapture(0)
# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# import time

# # app = Flask(__name__)

# camera = cv2.VideoCapture(0)  # use 0 for web camera
# #  for cctv camera use rtsp://username:password@ip_address:554/user=username_password='password'_channel=channel_number_stream=0.sdp' instead of camera

# sentence = []
def gen_frames():
    
    while cap.isOpened(): 
        ret, frame = cap.read()
        image_np = np.array(frame)
        
        input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
        detections = detect_fn(input_tensor)
        
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                    for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        label_id_offset = 1
        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
                    image_np_with_detections,
                    detections['detection_boxes'],
                    detections['detection_classes']+label_id_offset,
                    detections['detection_scores'],
                    category_index,
                    use_normalized_coordinates=True,
                    max_boxes_to_draw=5,
                    min_score_thresh=.8,
                    agnostic_mode=False)

        # yield cv2.imshow('object detection',  cv2.resize(image_np_with_detections, (800, 600)))
        frame = cv2.resize(image_np_with_detections, (800, 600))
        word = category_index[detections['detection_classes'][np.argmax(detections['detection_scores'])]+1]['name']
        sentence.append(word)
        
        # time.sleep(0.5)
        # c = ''
        # if len(sentence)>5:
        #     c+=max(sentence,key=sentence.count)
        #     print(c)
        #     sentence.clear()

        # print(sentence)
        # sentence.pop()
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

      # generate frame by frame from camera
    # while True:
    #     # Capture frame-by-frame
    #     success, frame = camera.read()  # read the camera frame
    #     if not success:
    #         break
    #     else:
    #         ret, buffer = cv2.imencode('.jpg', frame)
    #         frame = buffer.tobytes()
    #         yield (b'--frame\r\n'
    #                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result



# print(sentence)


def classify():
    img = cv2.imread(IMAGE_PATH)
    image_np = np.array(img)

    input_tensor = tf.convert_to_tensor(np.expand_dims(image_np, 0), dtype=tf.float32)
    detections = detect_fn(input_tensor)

    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy()
                for key, value in detections.items()}
    detections['num_detections'] = num_detections

    # detection_classes should be ints.
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    label_id_offset = 1
    image_np_with_detections = image_np.copy()

    viz_utils.visualize_boxes_and_labels_on_image_array(
                image_np_with_detections,
                detections['detection_boxes'],
                detections['detection_classes']+label_id_offset,
                detections['detection_scores'],
                category_index,
                use_normalized_coordinates=True,
                max_boxes_to_draw=5,
                min_score_thresh=.8,
                agnostic_mode=False)

    plt.imshow(cv2.cvtColor(image_np_with_detections, cv2.COLOR_BGR2RGB))
    plt.show()



@app.route('/', methods=['GET', 'POST'])
def login():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = request.form['password']
            if form.validate():
                if helpers.credentials_valid(username, password):
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Login successful'})
                return json.dumps({'status': 'Invalid user/pass'})
            return json.dumps({'status': 'Both fields required'})
        return render_template('login.html', form=form)
    user = helpers.get_user()
    return render_template('home.html', user=user)



@app.route('/video_feed')
def video_feed():
    """Video streaming route. Put this in the src attribute of an img tag."""
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route("/logout")
def logout():
    session['logged_in'] = False
    return redirect(url_for('login'))

@app.route("/real")
def real():
    return render_template('index.html',sentence=['Rabi'])

@app.route("/image")
def image():
    return redirect("http://localhost:3000/", code=302)



# -------- Signup ---------------------------------------------------------- #
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if not session.get('logged_in'):
        form = forms.LoginForm(request.form)
        if request.method == 'POST':
            username = request.form['username'].lower()
            password = helpers.hash_password(request.form['password'])
            email = request.form['email']
            if form.validate():
                if not helpers.username_taken(username):
                    helpers.add_user(username, password, email)
                    session['logged_in'] = True
                    session['username'] = username
                    return json.dumps({'status': 'Signup successful'})
                return json.dumps({'status': 'Username taken'})
            return json.dumps({'status': 'User/Pass required'})
        return render_template('login.html', form=form)
    return redirect(url_for('login'))


# -------- Settings ---------------------------------------------------------- #
@app.route('/settings', methods=['GET', 'POST'])
def settings():
    if session.get('logged_in'):
        if request.method == 'POST':
            password = request.form['password']
            if password != "":
                password = helpers.hash_password(password)
            email = request.form['email']
            helpers.change_user(password=password, email=email)
            return json.dumps({'status': 'Saved'})
        user = helpers.get_user()
        return render_template('settings.html', user=user)
    return redirect(url_for('login'))


# ======== Main ============================================================== #
if __name__ == "__main__":
    app.run(debug=True, use_reloader=True, host="0.0.0.0")
