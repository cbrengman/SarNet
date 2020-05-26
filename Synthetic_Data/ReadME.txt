Main File: generate_synthetic_int.py
Dependencies:
	-numpy
	-cv2
	-PIL
	-multiprocessing
	-math
	-h5py
	-scipy
	-skimage
To Run:
	-Choose large or small displacements 
		-Edit line 8 genrand_synth_displacements(_smol)
	-Choose output folder
		-Edit lines 48-58 with folder location 
		-Folder structure should be Main --> train/val/test --> ndata/ndata_wrap --> data/noise/orig --> filename		
		-You only have to edit the Main section to be the folder name you want
	-Choose number of samples
		-Edit the range function in line 75 to do this. 
	-Choose to output data to training or validation datasets
		-Edit line 76 [(proc,'train')] or [(proc,'val')]
	-If you only want to make 1 and plot it comment out lines 75-80 and uncomment the remainder
