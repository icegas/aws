from sagemaker.tensorflow import TensorFlowModel
import requests
import json
import base64
from PIL import Image
from io import BytesIO
import numpy as np
#import tensorflow as tf


#savedir = '../src/prod_models/fake_face_model_v01'
#predictor = tf.saved_model.load(savedir)

model = TensorFlowModel(model_data='s3://fake_face_model_v01.tar.gz', role='MySageMakerRole')

predictor = model.deploy(initial_instance_count=1, instance_type='ml.c5.xlarge', accelerator_type='ml.eia1.medium')

def lambda_handler(event, context):
    try:
        url = event['img_url']
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = img.resize( (512, 512) , Image.ANTIALIAS)
        img = np.asarray(img).reshape((1, 512, 512, 3)).astype('float32')
        img = img / 255.0
        out = predictor.predict(img)
        return out
    except Exception as e:
        raise Exception('ProcessingError')

#out = lambda_handler({'img_url' : "https://images.unsplash.com/photo-1541963463532-d68292c34b19?ixid=MnwxMjA3fDB8MHxwaG90by1wYWdlfHx8fGVufDB8fHx8&ixlib=rb-1.2.1&auto=format&fit=crop&w=688&q=80"}, 0 )
#print(out)
