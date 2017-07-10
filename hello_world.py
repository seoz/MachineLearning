import tensorflow as tf

sess = tf.Session()

hello = tf.constant("Hello world!")
print("Constant")
print(hello)
print(sess.run(hello))

node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0)
node3 = tf.add(node1, node2)
node4 = node1 + node2
print("node1, 2: ", node1, node2)
print("node3: ", node3)
print("node4: ", node4)
print("sess.run(node1, node2): ", sess.run([node1, node2]))
print("sess.run(node3): ", sess.run(node3))

a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print("Placeholder")
print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, feed_dict={a: [1, 3], b: [2.2, 4.5]}))