import pickle
import imageio
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

LOGS_PATH = "../../image2sdf/logs/decoder/log.pkl"
LATENT_CODE_PATH = "models_and_codes/latent_code.pkl"
PLOT_PATH = "../../image2sdf/plots/pca/"

logs = pickle.load(open(LOGS_PATH, 'rb'))
dict_hash_2_code = pickle.load(open(LATENT_CODE_PATH, 'rb'))

list_model_hash = list(dict_hash_2_code.keys())
norm = []
for hash_1 in list_model_hash:
    for hash_2 in list_model_hash:
        norm.append((dict_hash_2_code[hash_1] - dict_hash_2_code[hash_2]).norm())

code = []
for hash in dict_hash_2_code.keys():
    code.append(np.array(dict_hash_2_code[hash]))
code = np.array(code)

# code = StandardScaler().fit_transform(code)
# pca = PCA(n_components=6)
# principalComponents = pca.fit_transform(code)

# plt.plot(principalComponents[:,0], principalComponents[:,1], 'x')

plt.plot(code[:,0], code[:,1], 'x')
plt.title("Final latent codes")
plt.savefig(PLOT_PATH + "latent_space.png")
