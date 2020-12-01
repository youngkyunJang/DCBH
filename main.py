from utils.RetrievalTest import *
from utils.Functions import *

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9

codeLength = 6
numBits = 48

dataset_flag = 1

if(dataset_flag==0):
    DATASET_NAME = 'youtubeface'
    MODEL_DIR = './models'
    TRAIN_SET_PATH = './data/train_vec_folder_32_32'
    TEST_SET_PATH = './data/test_vec_folder_32_32'
    NB_CLASSES = 1595
    print("YoutubeFace")
else:
    DATASET_NAME = 'facescrub'
    MODEL_DIR = './models'
    TRAIN_SET_PATH = './data/train_folder_facescrub'
    TEST_SET_PATH = './data/test_folder_facescrub'
    NB_CLASSES = 530
    print("Facescrub")


# Hyperparameter
batch_size = 256
total_epochs = 2000
quantization_loss_params = 0.1
center_loss_params = 0.0001
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32

x = tf.placeholder(tf.float32, shape=[None, IMAGE_WIDTH, IMAGE_HEIGHT, 3])
label = tf.placeholder(tf.int32, shape=[None, NB_CLASSES])
training_flag = tf.placeholder(tf.bool)

class FHNet():
    def __init__(self, x, training):
        self.training = training
        self.model = self.FH_net(x)

    def FH_net(self, input_x):

        x = conv_layer(input_x, filter=64, kernel=[3, 3], stride=1, layer_name='conv0')
        x = Batch_Normalization(x, training=self.training, scope='batch0')
        x = Relu(x)
        x = conv_layer(x, filter=64, kernel=[3, 3], stride=1, layer_name='conv0-1')
        x = Batch_Normalization(x, training=self.training, scope='batch0-1')
        x = Relu(x)
        x = Max_Pooling(x, pool_size=[2, 2], stride=2)

        x = conv_layer(x, filter=128, kernel=[3, 3], stride=1, layer_name='conv1')
        x = Batch_Normalization(x, training=self.training, scope='batch1')
        x = Relu(x)
        x = conv_layer(x, filter=128, kernel=[3, 3], stride=1, layer_name='conv1-1')
        x = Batch_Normalization(x, training=self.training, scope='batch1-1')
        x = Relu(x)
        x = Max_Pooling(x, pool_size=[2, 2], stride=2)

        x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2')
        x = Batch_Normalization(x, training=self.training, scope='batch2')
        x = Relu(x)
        x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2-1')
        x = Batch_Normalization(x, training=self.training, scope='batch2-1')
        x = Relu(x)
        x = conv_layer(x, filter=256, kernel=[3, 3], stride=1, layer_name='conv2-2')
        x = Batch_Normalization(x, training=self.training, scope='batch2-2')
        x = Relu(x)
        x = Max_Pooling(x, pool_size=[2, 2], stride=2)
        x_branch = Global_Average_Pooling(x)

        x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3')
        x = Batch_Normalization(x, training=self.training, scope='batch3')
        x = Relu(x)
        x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3-1')
        x = Batch_Normalization(x, training=self.training, scope='batch3-1')
        x = Relu(x)
        x = conv_layer(x, filter=512, kernel=[3, 3], stride=1, layer_name='conv3-2')
        x = Batch_Normalization(x, training=self.training, scope='batch3-2')
        x = Relu(x)
        x = Max_Pooling(x, pool_size=[2, 2], stride=2)

        x = Global_Average_Pooling(x)
        x = tf.concat([x, x_branch], 1)
        x = Linear(x, codeLength * numBits, layer_name='linear0')
        x = Batch_Normalization(x, training=self.training, scope='batch4')

        feature = Relu(x)
        x = Slice_Encode(feature, numBits)
        Q_x = Tanh(x)
        Q_loss(Q_x, 0.5)

        x_out = Linear(Q_x, NB_CLASSES, layer_name='linear1')
        x_out_center = Linear(feature, NB_CLASSES, layer_name='linear_center')

        return x_out, x_out_center, Q_x, feature


def run():

    dataset = load_data_split_pickle((TEST_SET_PATH, TRAIN_SET_PATH))
    train_x_search, train_y_search = dataset[0]
    test_x_search, test_y_search = dataset[1]

    train_x_search = train_x_search.reshape((train_x_search.shape[0], 3, IMAGE_WIDTH, IMAGE_HEIGHT))
    test_x_search = test_x_search.reshape((test_x_search.shape[0],3, IMAGE_WIDTH, IMAGE_HEIGHT))
    train_x_search = np.rollaxis(train_x_search, 1, 4)
    test_x_search = np.rollaxis(test_x_search,1, 4)
    train_x_search = deprocess_image(train_x_search)
    test_x_search = deprocess_image(test_x_search)
    print("Train data:", np.shape(train_x_search), np.shape(train_y_search))
    print("Test data :", np.shape(test_x_search), np.shape(test_y_search))

    logits, logits_center, descriptors, feature = FHNet(x=x, training=training_flag).model

    classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits))

    center_classification_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=logits_center))

    quantization_loss = tf.reduce_mean((tf.get_collection("quantization_loss")))*quantization_loss_params

    center_loss, centers_update_op = C_loss(feature, label, NB_CLASSES)
    center_loss = center_loss*center_loss_params

    regularization_var = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    regularization_loss = tf.contrib.layers.apply_regularization(regularizer, regularization_var)

    cost = classification_loss + center_classification_loss + quantization_loss + center_loss + regularization_loss

    with tf.control_dependencies([centers_update_op]):
        optimizer = tf.train.AdamOptimizer()
        train = optimizer.minimize(cost)

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(label, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver(tf.global_variables())

    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        total_parameters = 0
        for variable in tf.trainable_variables():
            # shape is an array of tf.Dimension
            shape = variable.get_shape()
            variable_parameters = 1
            for dim in shape:
                variable_parameters *= dim.value
            total_parameters += variable_parameters
        print("Total params : ",total_parameters)

        train_loss_min = 10.0
        top_MAP = 0.0

        for epoch in range(1, total_epochs + 1):
            if epoch == 1:
                print('Data load')
                db_x, db_y = dataset[0]
                db_num = np.shape(db_x)[0]
                train_x = np.squeeze(db_x)
                train_y = np.squeeze(db_y)
                train_x = train_x.reshape((train_x.shape[0], 3, IMAGE_WIDTH, IMAGE_HEIGHT))
                train_x = np.rollaxis(train_x, 1, 4)
                train_x = deprocess_image(train_x)
                train_y = np.eye(NB_CLASSES)[train_y]

                test_x, test_y = dataset[1]
                test_x = np.squeeze(test_x)
                test_y = np.squeeze(test_y)
                test_x = test_x.reshape((test_x.shape[0], 3, IMAGE_WIDTH, IMAGE_HEIGHT))
                test_x = np.rollaxis(test_x, 1, 4)
                test_x = deprocess_image(test_x)
                test_y = np.eye(NB_CLASSES)[test_y]

                iteration = int(db_num / batch_size) + 1

            pre_index = 0
            train_acc = 0.0
            train_loss = 0.0

            for step in range(1, iteration + 1):
                if pre_index + batch_size < db_num:
                    batch_x = train_x[pre_index: pre_index + batch_size]
                    batch_y = train_y[pre_index: pre_index + batch_size]
                else:
                    batch_x = train_x[pre_index:]
                    batch_y = train_y[pre_index:]
                train_feed_dict = {
                    x: batch_x,
                    label: batch_y,
                    training_flag: True
                }
                _, batch_loss, batch_claLoss, batch_cclaLoss, batch_Qloss, batch_Closs, batch_Rloss = sess.run(
                    [train, cost, classification_loss, center_classification_loss, quantization_loss, center_loss, regularization_loss], feed_dict=train_feed_dict)
                batch_acc = accuracy.eval(feed_dict=train_feed_dict)

                train_loss += batch_loss
                train_acc += batch_acc
                pre_index += batch_size

                if step == iteration:
                    train_loss /= iteration
                    train_acc /= iteration

                    test_acc, test_loss = Evaluate(sess, x, label, training_flag, test_x, test_y, cost, accuracy)

                    line = "epoch: %d/%d / Train - loss: %.4f, acc: %.4f / Test - loss : %.4f, acc : %.4f" % \
                           (epoch, total_epochs, train_loss, train_acc, test_loss, test_acc)
                    print(line)
                    if(epoch==5):
                        saver.save(sess=sess, save_path=MODEL_DIR + '/loss_train')
                        print("Initial Model updated")

            if (train_loss < train_loss_min):
                train_loss_min = train_loss
                saver.save(sess=sess, save_path=MODEL_DIR + '/loss_train')
                print("Model updated(loss_train)")

            if epoch%100==0:
                print("----------------------------------------------------------------------------------------")
                saver.restore(sess, MODEL_DIR+'/loss_train')
                MAP = search(sess, x, training_flag, descriptors, train_x_search, train_y_search, test_x_search, test_y_search)
                MAP_MAX = MAP
                print("Train loss MAP : %f" % (MAP))

                if(MAP_MAX > top_MAP):
                    top_MAP = MAP_MAX

if __name__ == '__main__':
    run()
