import numpy as np
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':

  f_ary_time = sys.argv[1]

  ary_time = np.loadtxt(f_ary_time)
  #ary_time = np.loadtxt('spectre_time_realmachine.csv')
  #ary_time = np.loadtxt('spectre_time.csv')
  print 'arary time size:', ary_time.shape

  n_trial = ary_time.shape[0]
  n_time = ary_time.shape[1]
  assert n_time == 256

  print np.amax(ary_time)
  print np.argmax(ary_time)

  avg_time = np.mean(ary_time, axis = 0)
  assert avg_time.shape[0] == n_time

  fig1, ax1 = plt.subplots()
  ax1.plot(avg_time)
  #fig1.show()
  plt.show()
