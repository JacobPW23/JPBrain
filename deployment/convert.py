import numpy as np

TUMOR_LABELS=['glioma (glejak)','meningioma (oponiak)','notumor (zdrowy)','pituitary (guz przysadki)']
#np.save('tumor_labels.npy',TUMOR_LABELS)

loaded=np.load('tumor_labels.npy')
print(f'Loaded: {loaded}')