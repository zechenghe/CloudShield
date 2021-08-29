import numpy as np
import matplotlib.pyplot as plt
import sys
import os

if __name__ == '__main__':

  f_res_badd = sys.argv[1]
  f_res_good = sys.argv[2]
  img_size = (10,6)
  dir_name = './figures/'
  plot_name = 'spectre_result'

  os.system('mkdir '+dir_name)

  res_badd = np.loadtxt(f_res_badd)
  res_good = np.loadtxt(f_res_good)
  assert res_badd.ndim == 1
  assert res_good.ndim == 1
  assert res_badd.shape[0] == 256
  assert res_good.shape[0] == 256

  print np.amax(res_badd)
  print np.amax(res_good)

  fig1, ax1 = plt.subplots(figsize = img_size)
  #ax1.plot(range(256), res_badd, linestyle = '+', color = 'r', linewidth = 3)
  #ax1.plot(range(256), res_badd, 'x-', color = 'r', linewidth = 4)
  ax1.plot(range(256), res_badd, color = 'r', linewidth = 5)
  ax1.set_xlim((-1, 256))
  ax1.set_ylim((-1, np.amax(res_badd)+1))
  ax1.set_xlabel("Secret Value", fontsize = 20)
  ax1.set_ylabel("Number of Hits", fontsize = 20)
  ax2 = ax1.twinx()
  #ax2.plot(range(256), res_good, linestyle = '*-', color = 'g', linewidth = 2)
  ax2.plot(range(256), res_good, color = 'b', linewidth = 3)
  ax2.set_xlim((-1, 256))
  ax2.set_ylim((-1, np.amax(res_badd)+1))
  plt.title('Insecure Processor (red) vs Speculative Load Blocking (blue) ', {'color': 'k', 'fontsize': 20})
  #plt.tight_layout()
  plt.savefig(dir_name+plot_name+'.pdf', bbox_inches='tight')
  #plt.show()
