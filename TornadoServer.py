from io import BytesIO
import base64
import os
import time
from PIL import Image
from predict import predict

import tornado.web
import tornado.websocket
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.gen

from dtlib.web.decos import deco_jsonp
from dtlib.web.tools import get_std_json_res

from tornado.options import define, options
define("port", default=8000, help="run on the given port", type=int)


class Indexhander(tornado.web.RequestHandler):
    def get(self):
        self.render('index.html')

class RecognitionHandler(tornado.web.RequestHandler):
    @deco_jsonp(is_async=False)
    def post(self):
        file_metas = self.request.files['imgdata']  # 提取表单中‘name’为‘imgdata’的文件元数据
        # print(file_metas)
        for meta in file_metas:  # 可能有多个文件,但是本接口中只接收一张图片的情况
            img_file = meta['body']

            stream = BytesIO(img_file)
            img = Image.open(stream)  # 转化成PIL格式
            # img.save('a.jpg')
            # upload_img(img, filename)
            res = predict(img)

            res_dict = dict(
                text=res
            )
            # self.render('result.html', x=res)
            return get_std_json_res(data=res_dict)


class Application(tornado.web.Application):
    def __init__(self):
        handlers = [
            (r'/', Indexhander),
            (r'/result', RecognitionHandler)
        ]
        settings = dict(
            template_path=os.path.join(os.path.dirname(__file__), "templates"),
        )
        tornado.web.Application.__init__(self, handlers, **settings)

if __name__ == '__main__':
    tornado.options.parse_command_line()
    app = Application()
    server = tornado.httpserver.HTTPServer(app)
    server.listen(options.port)
    tornado.ioloop.IOLoop.instance().start()