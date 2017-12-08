# soachishti (p146011@nu.edu.pk)

from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
import os
import matplotlib.font_manager
from scipy import misc
import numpy as np
import random
import sys

class TESTSet:
	fonts = []
	def __init__(self):
		random.seed()
		self.fonts = self.load_fonts()

	def load_fonts(self):
		# Get all font file
		flist = matplotlib.font_manager.get_fontconfig_fonts()
		fonts = [matplotlib.font_manager.FontProperties(fname=fname).get_file() for fname in flist]

		# Using only selected fonts
		"""
		fonts = [
			'C:/Windows/Fonts/AGENCYR.TTF',
			'C:/Windows/Fonts/AGARAMONDPRO-BOLD.OTF',
			'C:/Windows/Fonts/ARIAL.TTF',
			'C:/Windows/Fonts/CALIBRI.TTF',
			'C:/Windows/Fonts/CAMBRIA.TTC',
			'C:/Windows/Fonts/CANDARA.TTF',
			'C:/Windows/Fonts/CENTURY.TTF',
			'C:/Windows/Fonts/CONSOLA.TTF',
			'C:/Windows/Fonts/COMIC.TTF',
			'C:/Windows/Fonts/GEORGIA.TTF'
		]
		"""
		"""
		fonts = [
			'C:/Windows/Fonts/AGENCYR.TTF',
			'C:/Windows/Fonts/AGARAMONDPRO-BOLD.OTF',
			'C:/Windows/Fonts/ARIAL.TTF',
			'C:/Windows/Fonts/CALIBRI.TTF',
			'C:/Windows/Fonts/CAMBRIA.TTC',
			'C:/Windows/Fonts/AGENCYR.TTF',
			'C:/Windows/Fonts/AGARAMONDPRO-BOLD.OTF',
			'C:/Windows/Fonts/ARIAL.TTF',
			'C:/Windows/Fonts/CALIBRI.TTF',
			'C:/Windows/Fonts/CAMBRIA.TTC',
		]
		"""
		
		return fonts

	def generate_data(self, folder='test-set'):
		for number in range(10):
			for i in range(len(self.fonts)):
				img = Image.new('1', (10, 10))
				
				draw = ImageDraw.Draw(img)
				font = ImageFont.truetype(font=self.fonts[i], size=10)
				
				draw.text((2, 0), str(number), 1, font=font)

				directory = folder +"/" + str(number) + "/";
				if not os.path.exists(directory):
					os.makedirs(directory)
				
				img.save(directory + str(i) + ".bmp")

	def load_data_simple(self, folder='test-set', train_percent = 0.8):
		if not os.path.exists(folder):
			self.generate_data(folder)

		data = []
		label = []

		for number in range(10):
			DIR = folder + "/" + str(number) + "/"
			data_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
			test_count = int(data_count * (1 - train_percent))
			
			for i in range(data_count):
				path = folder + "/" + str(number) + "/" + str(i) + ".bmp"
				img = misc.imread(path) # / 255
				data.append(img)
				label.append(number)

		return np.array(data), np.array(label)

	def load_data(self, folder='test-set', train_percent = 0.8):
		if not os.path.exists(folder):
			self.generate_data(folder)

		x_train = []
		y_train = []

		x_test = []
		y_test = []

		for number in range(10):
			DIR = folder + "/" + str(number) + "/"
			data_count = len([name for name in os.listdir(DIR) if os.path.isfile(os.path.join(DIR, name))])
			train_count = int(data_count * train_percent)
			
			# Get unique selection for each images.
			random_selection = random.sample(range(data_count), data_count)

			for i in random_selection:
				path = folder + "/" + str(number) + "/" + str(i) + ".bmp"
				img = misc.imread(path) # / 255
				if train_count > 0:
					x_train.append(img)
					y_train.append(number)
					train_count -= 1
				else:
					x_test.append(img)
					y_test.append(number)

		return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)

if __name__ == "__main__":
	ss = TESTSet()
	#ss.generate_data("test-set-519")
	(x_train, y_train, x_test, y_test) = ss.load_data("test-set-519")
	print(x_train[600])
	print(y_train[600])
	
	print(x_test[100])
	print(y_test[100])
	#print(len(x_train))
	#print(len(x_test))