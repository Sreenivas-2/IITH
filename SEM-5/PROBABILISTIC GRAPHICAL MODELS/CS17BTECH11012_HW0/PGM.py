import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Reading the image
im1 = np.array(Image.open('HW0-left-gray.png').convert('L'))
im2 = np.array(Image.open('HW0-right-gray.png').convert('L'))

# Noise
mean = 0
sigma = 1
noise1 = np.random.normal(mean, sigma, (im1.shape[0], im1.shape[1]))
noise2 = np.random.normal(mean, sigma, (im2.shape[0], im2.shape[1]))

# Adding noise to the images
im1 = im1 + noise1
im2 = im2 + noise2

# Normalizing the images
im1 = im1/255
im2 = im2/255

disparity = np.zeros((im1.shape[0], im1.shape[1]))

compatibility_nodes = np.zeros((256, 256, 10))

gamma = 1

threshold = 3

def get_node_compatibility(row, column, disparity):
	'''
	Computes the compatibility at a given pixel.
	'''
	return np.exp((-1/(2 * (sigma ** 2))) * ((im1[row, column] - im2[row, column - disparity]) ** 2))


def get_message(row, column, direction, root_value, counter):
	'''
	Computes the message.
	'''
	Message = 0
	counter += 1

	if direction == 'UP':
		neighbor_x, neighbor_y = row - 1, column

	elif direction == 'DOWN':
		neighbor_x, neighbor_y = row + 1, column

	elif direction == 'RIGHT':
		neighbor_x, neighbor_y = row, column + 1

	elif direction == 'LEFT':
		neighbor_x, neighbor_y = row, column - 1

	elif direction == 'UP_RIGHT':
		neighbor_x, neighbor_y = row - 1, column + 1

	elif direction == 'UP_LEFT':
		neighbor_x, neighbor_y = row - 1, column - 1

	elif direction == 'DOWN_LEFT':
		neighbor_x, neighbor_y = row + 1, column - 1

	elif direction == 'DOWN_RIGHT':
		neighbor_x, neighbor_y = row + 1, column + 1


	for i in range(0,9):
		# print((root_value-i)**2)
		edge_compatibility = np.exp((-1/(2 * (gamma ** 2))) * (min((root_value - i) ** 2, threshold ** 2)))
		# node_compatibility = get_node_compatibility(row, column, i)
		node_compatibility =  compatibility_nodes[neighbor_x, neighbor_y, i]

		if counter == 1:
			message = 1

		else:
			message = get_message(neighbor_x, neighbor_y, direction, i, counter)

		Message += edge_compatibility * node_compatibility * message


	return Message

def main(image_l, image_r, H = 0):
	'''
	Computes the compatibility values for edges between pixels in the image.
	param @image_l : left image.
	param @image_r : right image.
	param @H : neighborhood for each pixel (size of the window).
	'''
	for i in range(15, image_l.shape[0] - 15):
		for j in range(15, image_l.shape[1] - 15):
			for value in range(0, 9):
				# print("counter : ",i, j)
				compatibility_nodes[i, j, value] = get_node_compatibility(i, j, value)


	for i in range(15, image_l.shape[0] - 15):
		for j in range(15, image_l.shape[1] - 15):
			print("index : ", i, j)
			prev_marginal_probability = 0
			disparity_value = 0
			for value in range(0, 9):
				message = 1
				# node_compatibility = get_node_compatibility(i, j, value)
				node_compatibility =  compatibility_nodes[i, j, value]
				message *= get_message(i, j, 'UP', value, 0)
				message *= get_message(i, j, 'DOWN', value, 0)
				message *= get_message(i, j, 'LEFT', value, 0)
				message *= get_message(i, j, 'RIGHT', value, 0)
				message *= get_message(i, j, 'UP_LEFT', value, 0)
				message *= get_message(i, j, 'UP_RIGHT', value, 0)
				message *= get_message(i, j, 'DOWN_LEFT', value, 0)
				message *= get_message(i, j, 'DOWN_RIGHT', value, 0)

				marginal_probability = node_compatibility * message

				# print("marginal_probability : ", marginal_probability)

				if marginal_probability > prev_marginal_probability:
					prev_marginal_probability = marginal_probability
					disparity_value = value

			disparity[i, j] = disparity_value
			# print("disparity : ", disparity[i,j])

	return disparity



if __name__ == '__main__':
    disparity = main(im1, im2)
    # imgplot = plt.imshow(disparity, interpolation='nearest', origin='upper', cmap='gray')
    # Image.fromarray(disparity).save('depth_' + str(gamma) + '_' + str(sigma) + '_' + str(threshold) + '_.png')
    plt.imsave('depth_' + str(gamma) + '_' + str(sigma) + '_' + str(threshold) + '_.png', disparity, cmap = 'gray')
    # plt.show()
