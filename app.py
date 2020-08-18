#################################################
# Dependencies and setup
#################################################
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import requests
import warnings
import logging
import os
import time
from datetime import datetime
from dateutil.relativedelta import relativedelta
from flask import Flask, render_template, jsonify, request
from flask import redirect, send_file, url_for, flash, send_from_directory
from werkzeug.utils import secure_filename
from sthreefiles import download_file, upload_file, list_files
from pprint import pprint
warnings.filterwarnings('ignore')
logger = logging.Logger('catch_all')
import secrets
import matplotlib
matplotlib.use('Agg')

# setting the dependencies...
print("Before setting dependencies...")
import cv2
print(" setting dependencies...-1")
# from google.colab.patches import cv2_imshow
# import tensorflow as tf
import keras
print(" setting dependencies...0")
## from tensorflow import keras
from keras.preprocessing import image
print(" setting dependencies...1")
from keras.preprocessing.image import img_to_array
print(" setting dependencies...2")
from skimage import io
print(" setting dependencies...3")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
rmodel_path = "models/face_emotion.h5"
print("Before loading model...")
# model3 = tf.keras.models.load_model(rmodel_path)
model3 = keras.models.load_model(rmodel_path)

# get os environments settings
aws_bucket = os.environ.get('AWS_BUCKET')
print("Bucket: ", aws_bucket)

# Load the model
# model_path = "models/facial_expressions_cnn_R158.h5"
# model_path2 = "models/facial_expressions_cnn_R20.json"
# model3 = load_model(model_path)
# model3 = model_from_json(open(model_path2).read())
expression = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

# Flask Setup
app = Flask(__name__)
time.sleep(5)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0
secret = secrets.token_urlsafe(32)
app.secret_key = secret
# for local export OS_ROOT = actual path where app.py is located and end with /
# myroot = os.environ.get('OS_ROOT')
myroot = ''
# myroot = os.path.dirname(__file__)
print("Printing root directory : ", myroot)
# UPLOAD_FOLDER = myroot+"/static/uploads"
# OUT_FOLDER = myroot+"/static/uploads/out/"
UPLOAD_FOLDER = myroot+"static/uploads"
OUT_FOLDER = myroot+"static/uploads/out/"
simage = 'stats.png'
# webcam_icon = myroot+"static/webcam.jpg"
# changing back to relative path
webcam_icon = "static/webcam.jpg"
wip_icon = myroot+"static/wipicon.png"
BUCKET = aws_bucket
app.config['UPLOAD_EXTENSIONS'] = ['.jpg', '.png', '.gif', '.mp4']
content = ''

#################################################
# Flask Routes
#################################################

@app.route("/", methods={'GET'})
def home():
    print("Server received request for 'Home' page...")
    print("Printing root directory : ", myroot)
    print("Printing webcam directory : ", webcam_icon)
    # files = os.listdir(app.config['UPLOAD_FOLDER'])
    # return render_template("test.html", files=files)
    return render_template("index.html", webcamicon=webcam_icon, rootdir = myroot)

# uploads a file to S3 bucket
@app.route("/", methods=['GET', 'POST'])
def upload():
    print("Server received request for upload page...")
    print("Call inside upload file ", request.method)
    print("method value :",  request.form['submit_button'])
    ofilename = ''
    presult = ''
    if request.method == "POST":
        if request.form['submit_button'] == 'Upload':
            if os.path.isfile(OUT_FOLDER+'predicted_image.jpg'):
                os.remove(OUT_FOLDER+'predicted_image.jpg')
                os.remove(OUT_FOLDER+'stats.png')
            if os.path.isfile(OUT_FOLDER+'predicted_video.mp4'):
                os.remove(OUT_FOLDER+'predicted_video.mp4')
                # os.remove(OUT_FOLDER+'stats.png')
            plt.clf()
            simage = 'stats.png'
            f = request.files['file']
            print("Original File:", f.filename)
            filename = secure_filename(f.filename)
            print("Destination File Name: ", filename)
            if filename != '':
                file_ext = os.path.splitext(filename)[1]
                if file_ext not in app.config['UPLOAD_EXTENSIONS']:
                    print("Invalid Extension.. skipping the file...")
                    # error = "Invalid File Extension..."
                    # flash(u'Invalid File Extension...', 'error')
                    flash('Invalid File Extension...')
                    # abort(400)
                else:
                    f.save(os.path.join(UPLOAD_FOLDER, filename))
                    # uploading to S3 bucket
                    upload_file(f"{UPLOAD_FOLDER}/{filename}", filename, BUCKET)
                    print("File uploaded to S3 Bucket............")
                    # showorig(f"uploads/{filename}")
                    ofilename = filename
                    print("    ")
                    print("Calling prediction function.....")
                    pfilename, content = prediction(f"{UPLOAD_FOLDER}/{ofilename}", filename, file_ext)
                    # presult = "predicted_image.jpg"
                    if file_ext == '.mp4':
                        simage = ''
                    # verify all are having expected value before hittnig html
                    print("root......  :", myroot)
                    pfilename = myroot+"static/uploads/out/"+pfilename
                    ofilename = myroot+"static/uploads/"+ofilename
                    spchart = myroot+"static/uploads/out/"+simage
                    print("Finished prediction function.....", pfilename)
                    print("Content   : ", content)
                    print("Extension :", file_ext)
                    print("stats :", simage)
                    print("Webcam   : ", webcam_icon)
                    print("userimage   : ", ofilename)
                    print("predicted image   : ", pfilename)
        
            return render_template("index.html", rootdir = myroot, webcamicon=webcam_icon, userimage = ofilename, predicted_image = pfilename, stats_image = simage, spchart=spchart, content=content, wipicon=wip_icon)
        else:
            call_webcam()
            render_template("index.html", rootdir = myroot, webcamicon=webcam_icon, wipicon=wip_icon)
    else:
        render_template("index.html", rootdir = myroot, webcamicon=webcam_icon, wipicon=wip_icon)


# return ('', 204)
# @app.route("/showorig/<filename>")
# def showorig(filename):
#     print("Server received request to place original image to html")
#     full_filename = os.path.join("static/uploads/", filename)
#     print(filename)
#     userimage = filename
#     print(userimage, full_filename)
#     # return render_template("test.html", userimage = filename)
#     # return send_from_directory(UPLOAD_FOLDER, userimage)
#     # return redirect(url_for('showorig', userimage))
#     return ('', 204)

# activate webcam
@app.route("/call_webcam", methods=['GET'])
def call_webcam():
    print("wip")

    return ('OK', 204)





# download a file from s3
@app.route("/download/<filename>", methods=['GET'])
def download(filename):
    print("Server received request for download page...")
    if request.method == 'GET':
        print("Get Request and File Name :", filename)
        print(filename)
        output = download_file(filename, BUCKET)
        send_file(output, as_attachment=True)

    return redirect(url_for('home'))

# # download a file from s3 subfolder
# @app.route("/download/<subfolder>/<filename>", methods=['GET'])
# def download_sub(subfolder, filename):
#     print("Server received request for download_sub page...")
#     if request.method == 'GET':
#         if filename == '' & subfolder != '':
#             filename = subfolder
#             subfolder = ''
#         print("Get Request and File Name :", filename)
#         output = download_file(subfolder, filename, BUCKET)
#         send_file(output, as_attachment=True)
#
#     return redirect(url_for('home'))

# list s3 contents
@app.route("/list")
def list():
    print("Server received request for s3 contents list page...")
    contents = list_files(BUCKET)
    return render_template('s3contents.html', contents=contents)

# To call image prediction page
@app.route("/prediction/<imagefile>/<filename>/<fileext>")
def prediction(imagefile, fname, fileext):
    print("Server received request for 'Prediction' page...")
    # # setting the dependencies...
    # print("Before setting dependencies...")
    # import cv2
    # print(" setting dependencies...-1")
    # # from google.colab.patches import cv2_imshow
    # # import tensorflow as tf
    # import keras
    # print(" setting dependencies...0")
    # ## from tensorflow import keras
    # from keras.preprocessing import image
    # print(" setting dependencies...1")
    # from keras.preprocessing.image import img_to_array
    # print(" setting dependencies...2")
    # from skimage import io
    # print(" setting dependencies...3")
    # face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # rmodel_path = "models/face_emotion_old.h5"
    # print("Before loading model...")
    # # model3 = tf.keras.models.load_model(rmodel_path)
    # model3 = keras.models.load_model(rmodel_path)
    pfilename = ''
    if fileext in ['.jpg', '.png', '.gif']:
        print("Inside image file processing...")
        content = 'image'
        img = cv2.imread(imagefile,1)     
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces= face_cascade.detectMultiScale(gray, 1.1, 4)
        print("Before for loop model prediction...")
        for (x,y,w,h) in faces:
            print("Faces xywh:", x,y,w,h)
            imgg=cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            crop_img = img[y:y+h, x:x+w]
            dsize = (48, 48)
            crop_img = cv2.resize(crop_img, dsize)
            crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
            # cv2_imshow(crop_img)
            test = image.img_to_array(crop_img)
            test = np.expand_dims(test, axis = 0)
            test /= 255
            custom = model3.predict(test)
            print("Analysing all possible faces:", custom)
            sortedarray = np.array(custom[0])
            #  stats plot
            y_pos = np.arange(len(expression))
            print(y_pos)
            plt.bar(y_pos, custom[0], align='center', alpha=0.9)
            plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
            plt.xticks(y_pos, expression)
            plt.ylabel('percentage')
            plt.title('emotion')
            pltfile = OUT_FOLDER+simage
            plt.tight_layout()
            plt.savefig(pltfile)
            m=0.000000000000000000001
            a=custom[0]
            for i in range(0,len(a)):
                if a[i]>m:
                    m=a[i]
                    ind=i
            print('Expression Prediction:',expression[ind])
            newimg = cv2.putText(imgg, expression[ind], (x-10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.imwrite(OUT_FOLDER+"predicted_image.jpg", newimg)
            results = newimg
            pfilename = "predicted_image.jpg"
            # f.save(os.path.join(UPLOAD_FOLDER, newimg))
            # upload_file(f"uploads/{newimg}", newimg, BUCKET)

            print("End of prediction...")
    
    else:
        print("Inside video processing...")
        content = 'video'
        # filenamewithpath = imagefile
        cap = cv2.VideoCapture(str(imagefile))
        size = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        frame_count = int(cap.get(cv2.CAP_PROP_FPS))
        print("Frame Properties:", frame_count, size)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        print("Before video writer........ ")
        # out = cv2.VideoWriter(OUT_FOLDER+fname, cv2.VideoWriter_fourcc(*'MP4V'), 30, (1280,720))
        # out = cv2.VideoWriter(OUT_FOLDER+fname,cv2.VideoWriter_fourcc(*'MP4V'), 25, (484, 272))
        out = cv2.VideoWriter(OUT_FOLDER+"predicted_video.mp4", fourcc, frame_count, size)
        print("After Video out and before cap is Opened........ ")
        xval = []
        # if (cap.isOpened() == False):
        #     print("Error opening video stream or file")
        while(True):
            # Capture frame-by-frame
            print("Before if ret == True")
            ret, frame = cap.read()
            if ret == True:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces= face_cascade.detectMultiScale(gray, 1.1, 4)
                # print(faces)
                for (x,y,w,h) in faces:
                    frame=cv2.rectangle(frame, (x,y), (x+w, y+h), (255,0,0), 2)
                    crop_img = frame[y:y+h, x:x+w]
                    dsize = (48, 48)
                    crop_img = cv2.resize(crop_img, dsize)
                    crop_img = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                    print("Inside faces for loop...... ")
                    # cv2_imshow(crop_img)
                    test = image.img_to_array(crop_img)
                    test = np.expand_dims(test, axis = 0)
                    test /= 255
                    custom = model3.predict(test)
                    print("Video analysis :", custom)
                    m=0.000000000000000000001
                    a=custom[0]
                    print(a)
                    # data = [row.split('\t') for row in a]
                    # data = np.array(data, dtype='float')
                    # xval.append(data)
                    print("Inside faces for loop. before for loop..... ")
                    for i in range(0,len(a)):
                        if a[i]>m:
                            m=a[i]
                            ind=i
                    print('Video Expression Prediction:',expression[ind])
                    cv2.putText(frame, expression[ind], (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    print("Before write loop...... ")
                    out.write(frame)
            else:
                break
                # key = cv2.waitKey(1) & 0xFF
                # # if the `q` key was pressed, break from the loop
                # if key == ord("q"):
                #     break
                            
        print("Before cap release........ ")
        cap.release()
        out.release()
        # Closes all the frames
        cv2.destroyAllWindows()
        print("After destroy all windows ........ ")
        # print("Plot values :", xval)
        # #  stats plot
        # y_pos = np.arange(len(expression))
        # print(y_pos)
        # plt.bar(y_pos, xval, align='center', alpha=0.9)
        # plt.tick_params(axis='x', which='both', pad=10,width=4,length=10)
        # plt.xticks(y_pos, expression)
        # plt.ylabel('percentage')
        # plt.title('emotion')
        # pltfile = OUT_FOLDER+simage
        # plt.tight_layout()
        # plt.savefig(pltfile)
        pfilename = 'predicted_video.mp4'
        # pfilename = fname
        time.sleep(5)
        print("Video Results:", pfilename, content)

    print("Before moving the file to s3........ Filename", pfilename)
    # takes longer time based on file size to load into S3 bucket...
    # upload_file(f"static/uploads/out/{pfilename}", pfilename, BUCKET)
    print("File uploaded to S3 Bucket and display the image............")
    print("content.......  :  ", content)

    return (pfilename, content)

# To call training (ML) facial emotions page
@app.route("/train_facialemotions")
def train_facialemotions():
    print("Server received request for 'Train Facial Emotions' page...")
    filname = myroot+"data/fer2013.csv"
    Y = []
    X = []
    first = True
    print("Before reading csv file ...")
    for line in open(filname):
        if first:
            first = False
        else:
            row = line.split(',')
            Y.append(int(row[0]))
            X.append([int(p) for p in row[1].split()])

    X, Y = np.array(X) / 255.0, np.array(Y)
    num_class = len(set(Y))
    N, D = X.shape
    X = X.reshape(N, 48, 48, 1)
    print("Before setting the environment ...")
    #  settting required to train the model...
    import tensorflow as tf
    # from tensorflow.keras.models import Sequential
    # print("Before setting the environment ...1")
    # from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, BatchNormalization
    # print("Before setting the environment ...2")
    # from tensorflow.keras.layers import AveragePooling2D, Activation, MaxPooling2D
    # print("Before setting the environment ...3")
    # from tensorflow.keras.preprocessing import image
    # print("Before setting the environment ...4")
    # from tensorflow.keras.utils import to_categorical
    # print("Before setting the environment ...5")
    # from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
    # print("Before setting the environment ...6")
    # from tensorflow.keras.optimizers import adam
    # print("Before setting the environment ...7")

    # testing again after setting new environment..
    import keras
    from keras.models import Sequential
    from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
    from keras.layers import Dense, Activation, Dropout, Flatten
    #
    from keras.preprocessing import image
    # from keras.preprocessing.image import ImageDataGenerator
    from keras.layers.convolutional import Conv2D
    from keras.layers.convolutional import MaxPooling2D
    from keras.metrics import categorical_accuracy
    from keras.models import model_from_json
    from keras.callbacks import ModelCheckpoint
    from keras.optimizers import Adam
    from keras.layers.normalization import BatchNormalization
    #
    from sklearn.metrics import accuracy_score
    from sklearn.model_selection import train_test_split
    #
    print("Before train/test/split...")
    # train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.1, random_state=0)
    y_train = (np.arange(num_class) == y_train[:, None]).astype(np.float32)
    y_test = (np.arange(num_class) == y_test[:, None]).astype(np.float32)
    print("Before defining model...")
    # define model
    model = Sequential()
    input_shape = (48,48,1)
    model.add(Conv2D(64, (5, 5), input_shape=input_shape,activation='relu', padding='same'))
    model.add(Conv2D(64, (5, 5), activation='relu', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add 2nd layer
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(Conv2D(128, (5, 5),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # add 3rd layer
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(Conv2D(256, (3, 3),activation='relu',padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # flatten the model
    model.add(Flatten())
    # define hidden layers
    model.add(Dense(128))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    model.add(Dense(7))
    model.add(Activation('softmax'))
    # compile the model
    model.compile(loss='categorical_crossentropy', metrics=['accuracy'],optimizer='adam')
    # define epoch
    epochs = 5
    # define the path to store the model
    checkpoint_path = myroot+"models/model_checkpoint_"+str(epochs)+".h5"
    model_path = myroot+"models/facial_expressions_cnn_R"+str(epochs)+".h5"
    json_path = myroot+"models/facial_expressions_cnn_R"+str(epochs)+".json"
    print("Before training the model...")
    h=model.fit(x=X_train,
        y=y_train, 
        batch_size=64, 
        epochs=epochs, 
        verbose=1, 
        validation_data=(X_test,y_test),
        shuffle=True,
        callbacks=[ModelCheckpoint(filepath=checkpoint_path),]
    )
    print("Before saving the model...")
    # Save the model
    model.save(model_path)
    # Save json format
    model_json = model.to_json()
    with open(json_path, "w") as f:
        f.write(model_json)
    
    print("End of training model...")
    return ('', 204)

#
if __name__ == "__main__":
    # app.run(debug=True)
    app.run(host='0.0.0.0')
