import matplotlib.pyplot as plt
import numpy as np

def __onclick__(self, event):
	button = event.button
	x = int(np.floor(event.xdata))
	y = int(np.floor(event.ydata))

	print(x, y)

	self.click_num += 1
	self.click_x.append(x)
	self.click_y.append(y)

def __xy2ind__(self, x, y):

	ind = x + y*self.x

	return ind

def click_cell(self, data):

	self.click_num = 0
	self.click_x = []
	self.click_y = []

	plt.imshow(data, extent=(0, self.x, self.y, 0) )
	fig = plt.gcf()
	ax = plt.gca()

	cid = fig.canvas.mpl_connect('button_press_event', self.__onclick__)
	plt.show()


def cell_ind_to_cat_id(self, cell_ind, data_name=None):

	'''
	input indices of selected SOM pixels (cells).
	Output the indices of galaxies (in source catalog) that landed on those pixels.

	cell_ind:  A list of integers. Each integer is an index of a cell (pixel) of SOM. Each index ranges from 0 to npix-1.
		   The indices are also the indices of cell_ids.
	cell_ids:  A list of lists. The outter list is the same length as npix, with each inner list representing a cell.
		   Each inner list contains ids of galaxies landed in that cell. Inner lists can be empty.
	cat_ids:   Collection of all ids of galaxies that landed on the cells recorded in cell_ind.

	'''

	cell_ids = self.ids[data_name]

	cat_ids = []
	for i in cell_ind:
		cat_ids += cell_ids[i]

	return cat_ids


def get_click_ind(self):

	if len(self.click_x) < 1 or len(self.click_y) <1:
		print("Call click_cell to click on the map first")
		return 0

	self.clicked_ind = []

	for x, y in zip(self.click_x, self.click_y):
		ind = self.__xy2ind__(x, y)
		self.clicked_ind.append(ind)

	return self.clicked_ind




