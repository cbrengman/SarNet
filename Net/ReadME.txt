Dependencies:
	-time
	-torch
	-tqdm
	-tkinter
	-numpy
	-cv2
	-PIL
	-tifffile
	-matplotlib
	-mpl_toolkits
To Run:
	-Open Main
	-Edit line 45 to directory of synthetic data
	-Edit line 140 for number of wanted epochs and batch size
	-Run file
To Transfer Learn:
	-Open transfer_learn.py
	-Edit line 49 to directory of transfer dataset (should be directory --> train/val --> data/noise --> filename filestructure)
	-Edit line 70 to determine how much of the network is updated 
	
