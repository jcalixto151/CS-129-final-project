from astropy.io import fits
from tqdm import tqdm, trange
import numpy as np

class SelfOrgMap():

	from __lib_cell__ import click_cell, cell_ind_to_cat_id, get_click_ind, __onclick__, __xy2ind__


	def __init__(self, MapDim=2, x=100, y=1, boundary='Periodic', data_keys=[], dist_type='Euclidean', init='random', Path_distLib='./distLib/'):

		self.MapDim = MapDim                                   ## dimension of SOM
		self.x = x                                             ## size of SOM; only x is used if MapDim=1
		self.y = y                                             ## size of SOM
		self.npix = self.x*self.y                              ## pixel (cell) number
		self.boundary = boundary

		self.data_keys = data_keys
		self.dim = len(data_keys)                              ## dimension of features
		self.dist_type = dist_type                             ## type of distance metric; Euclidean or Chi2

		if type(init)==str:
			self.w = 2 * np.random.rand(self.npix, self.dim) - 1   ## the weights
		else:
			self.w = init

		self.iw = self.w.copy()

		self.distLib_name = Path_distLib + f"distLib{x}x{y}_{boundary}.fits"
		self.data_names = None

	def info(self):

		print("Map Dimension:", self.MapDim)
		print(f"Map Size: {self.x}x{self.y}")
		print("Number of SOM cells:", self.npix)
		print("Boundary Condition:", self.boundary)
		print("Dimension of SOM weight vectors:", self.w.shape)
		print("Feature Distance Metric (Euc/Chi2):", self.dist_type)
		print("Initial Weights: self.iw")
		print()

		if self.data_names is None:
			print("SOM not trained")
		else:
			print("Data Sets mapped onto SOM:", self.data_names)
			print("Trained Weights: self.w")

			if hasattr(self, 'uw'):
				print("Un-Scaled Weights: self.uw")

			if hasattr(self, 'somz'):
				print()
				print("Calculated SOMz: self.somz")
				print("SOMz error: self.somz_err")
				print("SOMz histogram: self.ztrue_dist")



	def map_object(self, data, var, mask, num_att, weights):

		## inputs:
		## 	data is an 1d array of features of 1 object
		## 	var is the variance in each feature
		## 	mask is for each feature too
		##
		## outputs:
		##	index of SOM cell that this object lands on (best-match unit, bmu)
		## 	chi-2 distance of the object to its bmu

		
		if self.dist_type == 'Chi2':
			dist = np.sum(mask * (data - weights)**2 / (var + 1), axis=1)     ## chi-2 distance from object to all cells
		elif self.dist_type == 'Euclidean':
			dist = np.sum(mask * (data - weights)**2, axis=1)                 ## Euclidean distance from object to all cells

		dist = dist * num_att / np.sum(mask)                            	  ## effective distance for sources with missing features

		dist_to_bmu = dist.min()
		bmu_ind = np.where(dist == dist_to_bmu)

		return bmu_ind[0][0], np.sqrt(dist_to_bmu)


	def train_SOM(self, train_data, train_var, mask, ind_seq='random', save_iter=[]):

		if self.data_names != None: 
			if 'training' in self.data_names:
				flag = input(f"Removing previous training and mapped data sets? Enter Y to continue")
				if flag == 'Y':
					self.w = self.iw.copy()
				else:
					return False


		N = train_data.shape[0]
		num_att = train_data.shape[1]

		self.data_names = ["training"]
		self.num_att = {"training": num_att}

			
		if len(save_iter):
			self.w_at_iter = {i:None for i in save_iter}
			self.ids_at_iter = {i:None for i in save_iter}
			self.occ_at_iter = {i:None for i in save_iter}
			save_iter = iter(save_iter)

		
		print("Training the SOM:")

		hdu = fits.open(self.distLib_name)
		distLib = hdu[0].data

		sigma_0 = np.max(distLib)
		sigma_f = np.min( distLib[np.nonzero(distLib)] )

		ids =  [ [] for i in range(self.npix)]              ## container for indices of galaxies that land on each cell

		if type(ind_seq) == str:
			train_seq = np.random.randint(0, high=N, size=N)
		else:
			train_seq = ind_seq
		
		iter_to_save = next(save_iter)
		for step, ind in enumerate(tqdm(train_seq)):

			if step == iter_to_save:

				self.w_at_iter[iter_to_save] = self.w.copy()
				self.ids_at_iter[iter_to_save] = ids
				self.occ_at_iter[iter_to_save] = np.array([ len(id) for id in ids ])

				try: 
					iter_to_save = next(save_iter)
				except StopIteration: 
					pass


			bmu_ind, dist_to_bmu = self.map_object(train_data[ind,:], train_var[ind,:], mask[ind,:], self.dim, self.w)

			ids[bmu_ind].append(ind)
			cell_dist_to_bmu = distLib[:, [bmu_ind]]

			alpha = 0.9 * (0.5/0.9) ** (1.0*step/N)
			sigma = sigma_0 * (sigma_f/sigma_0) ** (1.0*step/N)
			H = np.exp( - cell_dist_to_bmu**2 / sigma**2 )

			self.w += alpha * H * mask[ind,:]*(train_data[ind,:] - self.w)


		self.occ = {"training": np.array([ len(id) for id in ids ])}


	def map_objects(self, keys, data, var, mask, data_name='object'):

		if "training" not in self.data_names:
			print("Err: train the weights first")
			return False

		elif data_name in self.data_names:
			flag = input(f"Overwriting previous mapped objects under name {data_name}; Enter Y to continue")
			if flag == 'Y':
				print("Overwritten")
			else:
				return False
		else:
			self.data_names.append(data_name)


		N = data.shape[0]
		num_att = data.shape[1]
		self.num_att[data_name] = num_att


		ids = [ [] for i in range(self.npix)]
		bmu_ind = np.zeros(N, dtype=np.int32) - 1
		dist_to_bmu = np.zeros(N) - 1
	
		print("Mapping Objects onto SOM")

		if num_att == self.dim:
			weights = self.w
		elif num_att < self.dim:
			indices = [self.data_keys.index(key) for key in keys]
			weights = self.w[:, indices]

		for i in trange(N):
			this_ind, this_dist = self.map_object(data[i,:], var[i,:], mask[i,:], num_att, weights)
			ids[this_ind].append(i)
			
			bmu_ind[i] = this_ind
			dist_to_bmu[i] = this_dist


		## length of npix
		if not hasattr(self, 'ids'):
			self.ids = {data_name:ids}
		else:
			self.ids[data_name] = ids
		
		self.occ[data_name] = np.array([ len(id) for id in ids ])

		return 	bmu_ind, dist_to_bmu

	def cell_average(self, feature, data_name='object'):

		## input:
		##	feature: feature of each object in data previously mapped onto SOM with the same data_name.
		## output:
		##	averaged feature in each cell and the uncertainty.

		if data_name not in self.data_names:
			print("Err: Map Objects onto SOM first")
			return False

		cell_feature = [ [] for i in range(self.npix)] 
		ids = self.ids[data_name]

		for i in trange(self.npix):
			id = ids[i]                          ## id is a list of indicies of zture
			cell_feature[i] = feature[id]        ## cell_ave has the features in each cell
			
		feature_ave = np.nan_to_num(np.array([ np.mean(f) for f in cell_feature ]), nan=-1.0)
		feature_err = np.nan_to_num(np.array([ np.std(f, ddof=1)/np.sqrt(len(f)) for f in cell_feature ]), nan=-1.0)

		return feature_ave, feature_err


	def calc_SOMz(self, ztrue=None, data_name='object'):

		## input:
		##	zture: redshift of each objects in data previously mapped onto SOM with the same data_name.
		## output:
		##	No output, but the redshift of each SOM cell is updated.

		if ztrue is None:
			print("Err: True Redshift Needed to Train SOMz")
			return False

		elif data_name not in self.data_names:
			print("Err: Map Objects onto SOM first")
			return False


		cell_z = [ [] for i in range(self.npix)] 
		ids = self.ids[data_name]

		for i in trange(self.npix):
			id = ids[i]                  ## id is a list of indicies of zture
			cell_z[i] = ztrue[id]        ## cell_z has the redshift distribution in each cell
			
		somz = np.nan_to_num(np.array([ np.mean(z) for z in cell_z ]), nan=-1.0)
		somz_err = np.nan_to_num(np.array([ np.std(z, ddof=1)/np.sqrt(len(z)) for z in cell_z ]), nan=-1.0)


		if not hasattr(self, 'somz'):
			self.ztrue_dist = {data_name: cell_z}
			self.somz = {data_name: somz}
			self.somz_err = {data_name: somz_err}
		else:
			self.ztrue_dist[data_name] = cell_z
			self.somz[data_name] = somz
			self.somz_err[data_name] = somz_err


	def extract_SOMz(self, bmu_ind, data_name='object'):

		if data_name not in self.data_names:
			print("Err: Map Objects onto SOM first")

		if not hasattr(self, 'somz'):
			print("Err: Calculate SOMz first")

		somz = self.somz[data_name]
		somz_err = self.somz_err[data_name]
		ztrue_dist = self.ztrue_dist[data_name]

		return somz[bmu_ind], somz_err[bmu_ind], ztrue_dist[bmu_ind]


	def unscale_weights(self, ave, std):  ## rescale the weights to un-scaled colors/mags.

		unscaled_weights = np.zeros(self.w.shape)

		for i in range(self.dim):
			unscaled_weights[:,i] = self.w[:,i] * std[i] + ave[i]

		self.scale_ave = ave
		self.scale_std = std
		self.uw = unscaled_weights


	def get_2d_map(self, data):           ## map 1d cell info onto 2d

		data_2d = np.zeros((self.x, self.y))
		for i, d in enumerate(data):
			indx = i%self.x
			indy = int(np.floor(i/self.x))
			data_2d[indx, indy] = d

		return data_2d






