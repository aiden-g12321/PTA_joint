'Script to write .tim files.'


import numpy as np
import os


# observe pulsars monthly for 15 years
obs_times = np.arange(53000, 58479, 30.)
obs_times += np.random.normal(loc=0., scale=5., size=len(obs_times))

# half-microsecond uncertainty per TOA
uncertainties = [0.5] * len(obs_times)

# load and save pulsar names
pulsar_names = []
directory = 'par'
for filename in os.listdir(directory):
    if filename.endswith(".par"):
        file_path = os.path.join(directory, filename)
        with open(file_path, "r") as file:
            for line in file:
                if line.startswith("PSR"):
                    psr_name = line.split()[1]
                    pulsar_names.append(psr_name)
                    break
np.save('psr_names.npy', np.array(pulsar_names))

# write .tim file for every pulsar
freq = 1440.0
backend = 'GBT -pta NANOGrav'
for psr_name in pulsar_names:
    filename = f'tim/{psr_name}.tim'
    with open(filename, 'w') as f:
        f.write("FORMAT 1\n")
        f.write("MODE 1\n")
        for toa, uncertainty in zip(obs_times, uncertainties):
            f.write(f"{psr_name}  {freq:.6f}  {toa:.15f}  {uncertainty:.3f} {backend}\n")

