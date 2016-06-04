import random

# really dumb script to generate random runs of the yolo stuff

def generate_one_run_string():
  run = "python /home/hennigho/git_things/Ring_Net/ring_net_train.py"
  t_weight = float(random.randint(1,3))
  m_weight = float(random.randint(1,3))
  b_weight = float(random.randint(1,3))
  xi_weight = float(random.randint(1,4))
  yi_weight = float(random.randint(0,3))

  run = run + " --t_weight=" + str(t_weight) 
  run = run + " --m_weight=" + str(m_weight) 
  run = run + " --b_weight=" + str(t_weight) 
  run = run + " --xi_weight=" + str(xi_weight) 
#  run = run + " --yi_weight=" + str(yi_weight)  

  return run

def make_run_scripts(num):
  file_name = "/home/hennigho/git_things/Ring_Net/experiments.lst" 
  gpu_num = 0
  with open(file_name, 'w') as stream: 
    for i in xrange(num):
      stream.write('export CUDA_VISIBLE_DEVICES="' + str(gpu_num) + '" \n')
      gpu_num += 2
      run = generate_one_run_string()
      stream.write(run + ' --train_dir=/home/hennigho/git_things/Ring_Net/run_' + str(i).zfill(4) + ' &\n')

make_run_scripts(8)

