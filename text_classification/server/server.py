import tornado.ioloop
import tornado.web

import tensorflow as tf
import numpy as np

import conf
import util

class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r"/poi_classification", MainHandler),
        ]
        settings = {}
        super(Application, self).__init__(handlers, **settings)
        self.sess, self.input_x, self.dropout_keep_prob, self.predictions = self.init_model(conf.MODEL_FILENAME)
        self.parser = util.Util(conf.FEATURE_FILENAME, conf.LABEL_FILENAME)

    def init_model(self, checkpoint_file):
        graph = tf.Graph()
        with graph.as_default():
            sess = tf.Session()
            with sess.as_default():
                # Load the saved meta graph and restore variables
                saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
                saver.restore(sess, checkpoint_file)
        
                # Get the placeholders from the graph by name
                input_x = graph.get_operation_by_name("input_x").outputs[0] 
                #input_y = graph.get_operation_by_name("input_y").outputs[0]
                dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
                # Tensors we want to evaluate
                predictions = graph.get_operation_by_name("output/probabilities").outputs[0]
    
        return sess, input_x, dropout_keep_prob, predictions

class BaseHandler(tornado.web.RequestHandler):
    @property
    def sess(self):
        return self.application.sess

    @property
    def input_x(self):
        return self.application.input_x

    @property
    def dropout_keep_prob(self):
        return self.application.dropout_keep_prob

    @property
    def predictions(self):
        return self.application.predictions

    @property
    def parser(self):
        return self.application.parser

class MainHandler(BaseHandler):
    def predict(self, poi):
        x_pre = self.parser.parse_feature(poi)
        pre_res = self.sess.run(self.predictions, {self.input_x: x_pre, self.dropout_keep_prob: 1.0})[0]
        
        topn_res = []
        index_sorted = np.argsort(-pre_res)
        for i in range(5):
            index = index_sorted[i]
            topn_res.append((self.parser.index2label(index), pre_res[index]))
        return topn_res

    def get(self):
        self.write(conf.RESPONSE_STR % ("王府井", "", ""))

    def post(self):
        poi_name = self.get_body_argument("poi_name")
        info = ""
        if not poi_name:
            info = "<font color=red>bad poi input</font>"
            self.write(conf.RESPONSE_STR % (poi_name, info, ""))
        else:
            topn_res = self.predict(poi_name)
            most_label = topn_res[0][0]
            info = "<table border=\"1\"><tr><th>poi_type</th><th>poi_prob</th></tr>"
            for ele in topn_res: 
                info += "<tr><td>%s</td><td>%s</td></tr>" % ele
            info += "</table>"
            self.write(conf.RESPONSE_STR % (poi_name, most_label, info))

def main():
    http_server = tornado.httpserver.HTTPServer(Application())
    http_server.listen(8090)
    tornado.ioloop.IOLoop.current().start()

if __name__ == "__main__":
    main()
