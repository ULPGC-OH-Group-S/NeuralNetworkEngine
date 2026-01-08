import numpy as np
from scipy import ndimage


class ImageAugmentation:
	"""
	Applies random transformations to images to improve generalization.
	Designed for 2D grayscale images in format (batch, channels, height, width).
	"""
	
	def __init__(self, rotation_range=15, shift_range=0.1, horizontal_flip=True):
		"""
		Initialize augmentation parameters.
		
		Args:
			rotation_range (float): Maximum rotation angle in degrees. Default is 15.
			shift_range (float): Maximum shift as fraction of image size. Default is 0.1.
			horizontal_flip (bool): Whether to randomly flip images horizontally. Default is True.
		"""
		self.rotation_range = rotation_range
		self.shift_range = shift_range
		self.horizontal_flip = horizontal_flip
	
	def augment_batch(self, images):
		"""
		Apply random augmentation to a batch of images.
		
		Args:
			images (numpy.ndarray): Batch of images with shape (batch, channels, height, width)
			
		Returns:
			numpy.ndarray: Augmented images with same shape
		"""
		batch_size, channels, height, width = images.shape
		augmented = np.zeros_like(images)
		
		for i in range(batch_size):
			for c in range(channels):
				img = images[i, c]
				
				# Random rotation
				if self.rotation_range > 0:
					angle = np.random.uniform(-self.rotation_range, self.rotation_range)
					img = ndimage.rotate(img, angle, reshape=False, mode='nearest')
				
				# Random shift
				if self.shift_range > 0:
					shift_h = int(np.random.uniform(-self.shift_range, self.shift_range) * height)
					shift_w = int(np.random.uniform(-self.shift_range, self.shift_range) * width)
					img = ndimage.shift(img, [shift_h, shift_w], mode='nearest')
				
				# Random horizontal flip
				if self.horizontal_flip and np.random.random() > 0.5:
					img = np.fliplr(img)
				
				augmented[i, c] = img
		
		return augmented


def simple_augment(images, rotation=10, shift=0.1, flip=True):
	"""
	Function to augment a batch of images.
	
	Args:
		images (numpy.ndarray): Images with shape (batch, channels, height, width)
		rotation (float): Max rotation in degrees. Default is 10.
		shift (float): Max shift as fraction. Default is 0.1.
		flip (bool): Enable horizontal flip. Default is True.
		
	Returns:
		numpy.ndarray: Augmented images
	"""
	augmenter = ImageAugmentation(rotation_range=rotation, shift_range=shift, horizontal_flip=flip)
	return augmenter.augment_batch(images)
