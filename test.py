import matplotlib.pyplot as plt
import numpy as np
Eff_pool= np.array([-1.9005091621832058, -1.8343453127069986, -1.2262406490262432])
Rew_pool= np.array([0.17249196559441257, 0.30463044638303416, 1.81227786810822])
plt.plot(Rew_pool+Eff_pool, 'o')
plt.show()