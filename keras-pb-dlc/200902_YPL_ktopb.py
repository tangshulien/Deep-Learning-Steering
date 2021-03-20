'''
Convert xxx.keras (or .h5) to frozen xxx.pb. YPL 2020.8.26

我簡單說明一下我轉keras的python程式使用方法

1. 首先必須從leon的環境中安裝requirment(在虛擬環境下安裝)
jinn@Liu:~$ mkdir ~/ktopb
jinn@Liu:~$ python --version
Python 3.8.2
jinn@Liu:~$ pyenv virtualenv 3.7.3 ktopb
jinn@Liu:~$ cd ktopb
(ktopb) jinn@Liu:~/ktopb$ python --version
Python 3.7.3

--- downlaod Leon's requirements.txt
(ktopb) jinn@Liu:~/ktopb$ curl https://raw.githubusercontent.com/littlemountainman/modeld/master/requirements.txt -o requirements.txt

(ktopb) jinn@Liu:~/ktopb$ gedit requirements.txt
--- change line 17 to #carla==0.9.7 
--- keep line 116 as tensorflow-gpu==1.14.0 (OK for Python 3.7.3 but not 3.8.2)
--- delete lines 133-394

--- install requirements
(ktopb) jinn@Liu:~/ktopb$ pip3 install -r requirements.txt

2. 接著在環境中將此檔案放進去
--- go to browser and downlaod retinanet.h5 and put it in ~/ktopb
https://drive.google.com/file/d/1m0JITrswPqxbABDJDwwrjwi5G8TbkUpF/view?usp=sharing

--- go to browser and downlaod ktopb.py and put it in ~/ktopb
https://drive.google.com/file/d/12hbxygrMu1d3YyeFENVpF4PD51G0uY7t/view?usp=sharing

3. 在終端機的地方輸入
--- convert retinanet.h5 (saved in 'json mode') to frozen retinanet.pb

(ktopb) jinn@Liu:~/ktopb$ python ktopb.py retinanet.h5 retinanet.pb
xxx
Error mode:You use 'json mode' to covert this model

(ktopb) jinn@Liu:~/ktopb$ python ktopb.py retinanet.h5 retinanet.pb --json True
xxx
  File "ktopb.py", line 91, in <module>
    h5_to_pb(h5_save_path,args.pb,isjson=args.json)
  File "ktopb.py", line 53, in h5_to_pb
    model=model_from_json(json.load(jfile))
  File "/home/jinn/.pyenv/versions/3.7.3/lib/python3.7/json/__init__.py", line 293, in load
    return loads(fp.read(),
  File "/home/jinn/.pyenv/versions/3.7.3/lib/python3.7/codecs.py", line 322, in decode
    (result, consumed) = self._buffer_decode(data, self.errors, final)
UnicodeDecodeError: 'utf-8' codec can't decode byte 0x89 in position 0: invalid start byte

4. 若為json 儲存模式 則指令要改為 python ktopb.py xxx.keras xxx.pb --json True

(ktopb) jinn@Liu:~/ktopb$ python -m pip install --upgrade pip setuptools



'''
import tensorflow as tf
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2
import argparse
import json 
from tensorflow.keras.models import model_from_json

def h5_to_pb(h5_save_path,pb_save_name,isjson):
    if not isjson:
       try:
          model = tf.keras.models.load_model(h5_save_path, compile=False)
          model.summary()
       except ValueError:
              print("Error mode:You use 'json mode' to covert this model")
              return 1
    else:
       with open(h5_save_path,'r') as jfile:
            model=model_from_json(json.load(jfile))
            weights_file=h5_save_path.replace('json','keras')
            model.load_weights(weights_file)
    
    full_model = tf.function(lambda Input: model(Input))
  
    full_model = full_model.get_concrete_function([tf.TensorSpec(model.inputs[i].shape, model.inputs[i].dtype) for i in range(len(model.inputs))])
    

    # Get frozen ConcreteFunction
    frozen_func = convert_variables_to_constants_v2(full_model)
    #frozen_func.graph.as_graph_def()
    print('a')
    print(frozen_func.graph)
    layers = [op.name for op in frozen_func.graph.get_operations()]
    print("-" * 50)
    print("Frozen model layers: ")
    #for layer in layers:
        #print(layer)

    print("-" * 50)
    print("Frozen model inputs: ")
    print(frozen_func.inputs)
    print("Frozen model outputs: ")
    print(frozen_func.outputs)

    # Save frozen graph from frozen ConcreteFunction to hard drive
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir="./frozen_models",
                      name=pb_save_name,
                      as_text=False)
if __name__=="__main__":
   parser=argparse.ArgumentParser(description='keras to pb')
   parser.add_argument('k',type=str,default="transition",help='keras file name')
   parser.add_argument('pb',type=str,default="transition",help='save pb name')
   parser.add_argument('--json',type=bool,default=False,help='json format')
   args=parser.parse_args()
   h5_save_path=args.k
   h5_to_pb(h5_save_path,args.pb,isjson=args.json)
