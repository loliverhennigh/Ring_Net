import random

# really dumb script to generate random runs of the yolo stuff

def generate_one_run_string():
  run = "python /home/hennigho/git_things/Ring_Net/ring_net_eval.py"
  return run

def make_run_scripts(num):
  file_name = "/home/hennigho/git_things/Ring_Net/eval_experiments.lst" 
  gpu_num = 0
  with open(file_name, 'w') as stream: 
    for i in xrange(num):
      #stream.write('export CUDA_VISIBLE_DEVICES="' + str(gpu_num) + '" \n')
      gpu_num += 2
      run = generate_one_run_string()
      stream.write(run + ' --checkpoint_dir=/home/hennigho/git_things/Ring_Net/run_' + str(i).zfill(4) + '  --video_name=video_run_' + str(i).zfill(4) + '.mp4 \n')

make_run_scripts(8)

