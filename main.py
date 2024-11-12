import subprocess
import sys
import os
from Raft.demo import 

deeplab_predict_path = os.path.join("DeeplabV3Plus", "predict.py")
raft_demo_path = os.path.join("Raft", "demo.py")

def run_script(script_path):
    env = os.environ.copy()
    env['PYTHONPATH'] = os.pathsep.join([os.path.abspath("DeeplabV3Plus"), os.path.abspath("Raft")])
    command = [sys.executable, script_path]
    subprocess.run(command, env=env)

if __name__ == '__main__':
    print("Running image segmentation")
    run_script(deeplab_predict_path)

    print("Running RAFT ")
    run_script(raft_demo_path)
