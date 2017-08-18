#decoding camera images
import base64
#real-time server
import socketio
#concurrent networking 
import eventlet
#web server gateway interface
import eventlet.wsgi
#image manipulation
from PIL import Image
#web framework
from flask import Flask
#input output
from io import BytesIO

import argparse
import numpy as np
import tensorflow as tf

#helper class
import utils_tf

#initialize our server
sio = socketio.Server()
#our flask (web) app
app = Flask(__name__)


#set min/max speed for our autonomous car
MAX_SPEED = 25
MIN_SPEED = 10

#and a speed limit
speed_limit = MAX_SPEED

#registering event handler for the server
@sio.on('telemetry')
def telemetry(sid, data):
    if data:
        # The current steering angle of the car
        steering_angle = float(data["steering_angle"])
        # The current throttle of the car, how hard to push peddle
        throttle = float(data["throttle"])
        # The current speed of the car
        speed = float(data["speed"])
        # The current image from the center camera of the car
        image = Image.open(BytesIO(base64.b64decode(data["image"])))
        try:
            image = np.asarray(image)       # from PIL image to numpy array
            image = utils_tf.preprocess(image) # apply the preprocessing
            image = np.array([image])       # the model expects 4D array

            feed_dict = {x:image,keep_prob:1.0}
            steering_angle = float(sess.run(op_to_restore,feed_dict=feed_dict)[0])


            global speed_limit
            if speed > speed_limit:
                speed_limit = MIN_SPEED 
            else:
                speed_limit = MAX_SPEED
            throttle = 1.0 - steering_angle**2 - (speed/speed_limit)**2

            print('Angle: {}, Throttle: {}, Speed: {}, Speed Lim: {}'.format(steering_angle, throttle, speed, speed_limit))
            send_control(steering_angle, throttle)

        except Exception as e:
            print(e)

    else:
        
        sio.emit('manual', data={}, skip_sid=True)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0)


def send_control(steering_angle, throttle):
    sio.emit(
        "steer",
        data={
            'steering_angle': steering_angle.__str__(),
            'throttle': throttle.__str__()
        },
        skip_sid=True)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str, default='sess1.0',
        help='Path to model h5 file. Model should be on the same path.'
    )

    args = parser.parse_args()

    # Grab saved model files and initialize the model
    global sess
    sess = tf.InteractiveSession()
    saver = tf.train.import_meta_graph('./{}.meta'.format(args.model))
    saver.restore(sess,'./{}'.format(args.model))
    # Get tensorflow graph
    graph = tf.get_default_graph()
    # Get saved variables for inputing data and running calculations with prebuilt model
    x = graph.get_tensor_by_name("x:0")
    keep_prob = graph.get_tensor_by_name("keep_prob:0")
    op_to_restore = graph.get_tensor_by_name("op_to_restore:0") # operation to calculate output steering angle

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
