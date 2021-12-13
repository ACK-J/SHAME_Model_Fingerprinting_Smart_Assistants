# Smart_Home_Voice_Fingerprinting
:warning: Experimental - PLEASE BE CAREFUL. Intended for reasearch purposes only.
# Installation
`pip3 install -r requirements.txt`

` sudo apt install graphviz`

# Datasets
Our full dataset referenced in the paper can be found here: https://drive.google.com/file/d/1K19SDZ3IdvAv_0rK6mG9d8WTpHg85gzV/view?usp=sharing

The DeepVC dataset used in the paper can be found here: https://drive.google.com/drive/folders/1l-fSX9VdZH5kF9z7gm82xgYX5ca0kRI0?usp=sharing

# Run
```
python3 run.py --input-data ./commands
python3 run.py --load-time ./time_dump.pkl --load-size ./size_dump.pkl --load-classes ./y.pkl
python3 run.py --load-weights-time ./time_weights/time-model-050-0.708984-0.727088.h5 --load-weights-size ./size_weights/size-model-040-0.720978-0.672098.h5
python3 run.py --load-time ./time_dump.pkl --load-size ./size_dump.pkl --load-classes ./y.pkl --load-weights-time ./time_weights/time-model-050-0.708984-0.727088.h5 --load-weights-size ./size_weights/size-model-040-0.720978-0.672098.h5
```
- Make sure to check that the IP address of the smart home assistant is correctly assigned at the top of the `run.py` file.
