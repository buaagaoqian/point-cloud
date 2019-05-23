
#import tensorflow as tf
import scene
import model


epochs = 10

def train(s, net):

	#sess = tf.Session()
	print('begin training...')
	for epoch in range(epochs):
		for batch in s:
			print('  sample:', batch['sample'].shape)
			predict = net.predict(batch['sample'])
			print('  predict:', predict.shape)
			discriminate = net.discriminate(predict)
			print('  discriminate:', discriminate.shape)
			net.train_generator(discriminate, batch['ground_truth'])
			net.train_discrimitor(predict, batch['ground_truth'])
		print('epoch:', epoch)
	print('training completed')
	#sess.close()

if __name__ == '__main__':
	# 初始化场景
	s = scene.Scene()
	# 初始化网络
	net = model.Net()
	# 训练
	train(s, net)
