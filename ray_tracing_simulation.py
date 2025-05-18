
from sionna.rt import Scene, load_obj, Transmitter, Receiver
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


scene = Scene()
scene.add(load_obj("/content/ground.obj"))
scene.add(load_obj("/content/vegetation.obj"))
scene.add(load_obj("/content/buildings.obj"))
scene.add(load_obj("/content/base_stations.obj"))

df_tx = pd.read_excel("/content/hucre_bilgileri.xlsx")
for _, row in df_tx.iterrows():
    try:
        scene.add(Transmitter(position=[row["Longitude"], row["Latitude"], row["Height"]]))
    except:
        continue

df_rx = pd.read_excel("/content/5g_dl.xlsx")
for _, row in df_rx.iterrows():
    try:
        scene.add(Receiver(position=[row["Longitude"], row["Latitude"], row["Height"]]))
    except:
        continue

scene.max_depth = 2
scene.update()
scene.compute_paths()


pl = scene.compute_pathloss()
print("Pathloss (dB):", pl.numpy())

plt.plot(pl.numpy(), marker='o')
plt.xlabel("RX No")
plt.ylabel("Pathloss (dB)")
plt.title("Pathloss vs Alıcı")
plt.grid(True)
plt.show()
