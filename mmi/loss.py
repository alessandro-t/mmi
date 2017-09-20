import tensorflow as tf

class TensorFunction(object):
	def __init__(self):
		self.name = self.__class__.__name__.lower()

	def make(self):
		self.placeholder = tf.placeholder(tf.float32, shape=None, name='placeholder/' + self.name)
		self.summary = tf.summary.scalar('summary/' + self.name, self.placeholder)

	def get_function(self):
		return self.tf_function, None

	def get_summary(self):
		return [self.summary]

	def get_reset_op(self):
		return []

class Regularizer(TensorFunction):
	def build(self, *args, **kwargs):
		raise NotImplementedError() 

class L1Regularizer(Regularizer):
	def __init__(self, scale):
		self.scale = scale
	def build(self, *args, **kwargs):
		super(L1Regularizer, self)             
		assert 'layers' in kwargs# and 'scale' in kwargs
		all_weights = []
		for layer in kwargs['layers']:
			if hasattr(layer, 'weights'):
				all_weights += [tf.reshape(layer.weights, [-1])]
		all_weights = tf.concat(all_weights, axis=0)

		self.tf_function = self.scale*tf.reduce_sum(tf.abs(all_weights))

class L2Regularizer(Regularizer):
	def __init__(self, scale):
		self.scale = scale

	def build(self, *args, **kwargs):
		super(L2Regularizer, self)             
		assert 'layers' in kwargs# and 'scale' in kwargs
		all_weights = []
		for layer in kwargs['layers']:
			if hasattr(layer, 'weights'):
				all_weights += [tf.reshape(layer.weights, [-1])]
		all_weights = tf.concat(all_weights, axis=0)

		self.tf_function = self.scale*tf.reduce_sum(all_weights**2)

class BinaryCrossEntropyLoss(TensorFunction):

	def __call__(self, session):
		result = self.accumulator/max(self.length,1.0)
		self.reset_accumulator(session)
		return result

	def update(self, session, mb_dim, feed_dict):
		self.accumulator += mb_dim*session.run(self.tf_function, feed_dict)
		self.length += mb_dim

	def build(self, *args, **kwargs):
		super(BinaryCrossEntropyLoss, self).make()             
		assert 'labels' in kwargs and 'logits' in kwargs
		self.accumulator = 0
		self.length = 0
		labels, logits = tf.reshape(kwargs['labels'],[-1]), tf.reshape(kwargs['logits'], [-1])
		loss = tf.maximum(logits, 0) - logits*labels + tf.log(1 + tf.exp(-tf.abs(logits)))
		#loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=tf.reshape(labels,[-1]), logits=tf.reshape(logits,[-1]))
		#loss = tf.nn.weighted_cross_entropy_with_logits(labels, logits, 3.22)
		self.tf_function = tf.reduce_mean(loss)

	def reset_accumulator(self, session):
		self.accumulator = 0
		self.length = 0
        
class SoftmaxCrossEntropyLoss(TensorFunction):

	def __call__(self, session):
		result = self.accumulator/max(self.length,1.0)
		self.reset_accumulator(session)
		return result

	def update(self, session, mb_dim, feed_dict):
		self.accumulator += mb_dim*session.run(self.tf_function, feed_dict)
		self.length += mb_dim

	def build(self, *args, **kwargs):
		super(SoftmaxCrossEntropyLoss, self).make()
		assert 'labels' in kwargs and 'logits' in kwargs
		self.accumulator = 0
		self.length = 0
		labels, logits = kwargs['labels'], kwargs['logits']
		loss = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits)
		self.tf_function = tf.reduce_mean(loss)

	def reset_accumulator(self, session):
		self.accumulator = 0
		self.length = 0


class BinaryAccuracy(TensorFunction):

	def __call__(self, session):
		result = session.run(self.accuracy)
		self.reset_accumulator(session)
		return result

	def update(self, session, feed_dict):
		session.run([self.update_op_acc], feed_dict)

	def build(self, *args, **kwargs):
		super(BinaryAccuracy, self).make()             
		assert 'labels' in kwargs and 'logits' in kwargs
		labels, logits = kwargs['labels'], kwargs['logits']

		preds = tf.cast(tf.sigmoid(tf.reshape(logits,[-1])) > 0.5, tf.float32)
		self.accuracy, self.update_op_acc = tf.contrib.metrics.streaming_accuracy(labels, preds, name=self.name)

		self.stream_var = [i for i in tf.local_variables() if i.name.split('/')[0] == self.name]
		self.reset_stream_op = [tf.variables_initializer(self.stream_var)]

		self.tf_function = self.accuracy

	def reset_accumulator(self, session):
		session.run(self.reset_stream_op)

class SoftmaxAccuracy(TensorFunction):

	def __call__(self, session):
		result = session.run(self.accuracy)
		self.reset_accumulator(session)
		return result

	def update(self, session, feed_dict):
		session.run([self.update_op_acc], feed_dict)

	def build(self, *args, **kwargs):
		super(SoftmaxAccuracy, self).make()
		assert 'labels' in kwargs and 'logits' in kwargs
		labels, logits = kwargs['labels'], kwargs['logits']

		preds = tf.argmax(logits, axis=1)
		labels = tf.argmax(labels, axis=1)
		self.accuracy, self.update_op_acc = tf.contrib.metrics.streaming_accuracy(labels, preds, name=self.name)

		self.stream_var = [i for i in tf.local_variables() if i.name.split('/')[0] == self.name]
		self.reset_stream_op = [tf.variables_initializer(self.stream_var)]

		self.tf_function = self.accuracy

	def reset_accumulator(self, session):
		session.run(self.reset_stream_op)
