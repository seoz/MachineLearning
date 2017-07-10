import tensorflow as tf

# X and Y data
x_train = [1, 2, 3]
y_train = [1, 2, 3]

# weight and bias
# https://www.tensorflow.org/versions/master/api_docs/python/tf/Variable
# https://www.tensorflow.org/versions/master/api_docs/python/tf/random_normal
W = tf.Variable(tf.random_normal([1]), name="weight")
b = tf.Variable(tf.random_normal([1]), name="bias")

# hypothesis
hypothesis = W * x_train + b

# cost function
# https://www.tensorflow.org/versions/master/api_docs/python/tf/reduce_mean
# https://www.tensorflow.org/versions/master/api_docs/python/tf/square
cost = tf.reduce_mean(tf.square(hypothesis - y_train))

# random normal test
print(tf.random_normal([1]))
print(tf.random_normal([2]))

# Get the optimizer and minimize it.
# https://www.tensorflow.org/api_docs/python/tf/train/GradientDescentOptimizer
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(cost)

# Launch the graph in the session
# https://www.tensorflow.org/api_docs/python/tf/Session
sess = tf.Session()

# Initialize the global variables in the graph
# https://www.tensorflow.org/api_docs/python/tf/Session#run
sess.run(tf.global_variables_initializer())

# Repeat the training
for step in range(2001):

    # Runs operations and evaluates tensors in fetches.
    # Training happens here.
    sess.run(train)

    # Print the training result
    if step % 20 == 0:
        print(step, sess.run(cost), sess.run(W), sess.run(b))
