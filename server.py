# encoding: utf-8
"""
@author: banifeng 
@contact: banifeng@126.com

@version: 1.0
@file: viewpoint_mining.py
@time: 2019-08-09 19:39

这一行开始写关于本文件的说明与解释
"""
from flask import Flask, jsonify, request
from flask.templating import render_template
from gevent import pywsgi
from sa import SA

app = Flask(__name__, static_url_path='' ,template_folder='templates')
app.config['JSON_AS_ASCII'] = False


# from scripts.evaluate import Evaluator
import logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    level=logging.INFO,
    handlers=[
        logging.FileHandler(filename='sa.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
    
@app.route('/sa/', methods=['POST'])
def tag():
    try:
        message = request.form.get("message")
        print(message)
        score = sa.predict(args, message)
        print(score)

        result = {
            "code": 200,
            "score": str(score)
        }
    except Exception as e:
        print(e)
        result = {
            "code": 500,
            "message": "后端发生异常，请检查"
        }
    return jsonify({"result":result})

@app.route('/')
def hhhh():
    print('23232')
    return render_template('index.html')

class ARGS():
    def __init__(self) -> None:
        self.no_cuda=False
        self.local_rank=-1
        self.max_seq_length=256
        self.model_type="bert"
        # self.model_name_or_path="/home/mhxia/whou/workspace/pretrained_models/roberta-large #roberta-large/",
        self.output_dir="./output/checkpoint-best/"
        self.seed=1

if __name__ == '__main__':
    args= ARGS()
    # print(args)
    sa = SA(args)
    server = pywsgi.WSGIServer(('0.0.0.0', 10001), app)
    server.serve_forever()
    
