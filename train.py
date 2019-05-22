


import data
import model
import project



def train(dataset, point_net, image_net, optimizer):
	projector = project.Projector()
	for _ in range(epoches):
		for batch in dataset:
			point_pred = point_net(batch['x'])
			images = projector.project(point_pred)
			pred = image_net(images)
			loss = loss(pred, batch['y'])
			optimizer.optimize(loss)
			loss.backward()


if __name__ == '__main__':

	dataset = data.Dataset()
	point_net = model.PointNet()
	image_net = model.ImageNet()
	optimizer = tf.AdamOptimizer()
	train(dataset, point_net, image_net, optimizer)
