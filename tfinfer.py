import logging
from collections import namedtuple

log_fmt = '%(asctime)s -  %(name)s - %(levelname)s %(process)d %(funcName)s:%(lineno)d %(message)s'
logging.basicConfig(format=log_fmt, level=logging.INFO)

logger = logging.getLogger(__name__)

Batch = namedtuple('Batch', ['data'])
batch_size = 32

class Singleton(type):
    """
    Singleton Metaclass used to create a MXModel Singleton object
    """
    _instances = {}
    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(Singleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class MXModel(object):
    """
    This is a singleton class that just holds the loaded model in the module object
    We don't want to load the model for every inference when called from the map method
    """
    __metaclass__ = Singleton
    model_loaded = False
    mod = None
    synsets = None
 
    
    def __init__(self, model_url, batch_size):
        model_fname = self.download_model_files(model_url)
        MXModel.mod = self.init_module(model_fname, batch_size)
        MXModel.model_loaded = True


    def download_model_files(self, model):
        """
        Download model files from the given urls to local files    
        """    
        logger.info('download_model_files: model:%s' % (model))
        import tensorflow as tf
        s_fname = mx.test_utils.download(model, overwrite=False)
        return s_fname, p_fname, synset_fname

    def init_module(self, model_fname, batch_size):
        logger.info("initializing model")
        import tensorflow as tf
        return mod


def predict(img_batch, args):
    """
    Run predication on batch of images in 4-D numpy array format and return the top_5 probability along with their classes
    """
    import mxnet as mx
    import numpy as np
    logger.info('predict-args:%s' %(args))
 
    if not MXModel.model_loaded:
        MXModel(args['sym_url'], args['param_url'], args['label_url'], args['batch'])
    
    MXModel.mod.forward(Batch([mx.nd.array(img_batch)]))

    output = MXModel.mod.get_outputs()
    batch_prob = output[0].asnumpy()
    batch_top5 = []
    b = batch_prob.shape[0]

    for out in range(0,b):
        top_5 = []
        prob = batch_prob[out, :]
        prob = np.squeeze(prob)
        a = np.argsort(prob)[::-1]
        for i in a[0:5]:
            top_5.append('probability={:f}, class={}'.format(prob[i], MXModel.synsets[i]))
        batch_top5.append(top_5)

    logger.info('batch_top5:%s' %(batch_top5))
    return batch_top5

def load_images(images):
    """
    Decodes batch of image bytes and returns a 4-D numpy array.
    """
    import numpy as np
    batch = []
    for image in images:
        img_np = readImage(image)
        batch.append(img_np)

    batch_images = np.concatenate(batch)

    logger.info('batch_images.shape:%s'%(str(batch_images.shape)))

    return batch_images

def readImage(img_bytes):
    """
    Decodes an Image bytearray into 3-D numpy array.
    """
    from PIL import Image
    import numpy as np
    import io
    from array import array
    img = io.BytesIO(bytearray(img_bytes))
    # read the bytearray using OpenCV and convert to RGB
    img = Image.open(img)
    img = img.convert('RGB')
    #resize the image to 224x224
    img = img.resize((224, 224), Image.ANTIALIAS)
    # reshape the array from (height, width, channel) to (channel, height, width)
    img = np.swapaxes(img, 0, 2)
    img = np.swapaxes(img, 1, 2)
    # add a new axis to hold a batch of images.
    img = img[np.newaxis, :]
return img