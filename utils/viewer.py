import pyglet


class SimpleImageViewer(object):
    """
    Modified version of gymnasium viewer to chose format (RBG or I = grayscale)
    """

    def __init__(self, display=None):
        self.window = None
        self.isopen = False
        self.display = display

    def imshow(self, arr):
        if self.window is None:
            height, width, channels = arr.shape
            self.window = pyglet.window.Window(width=width, height=height, display=self.display)
            self.width = width
            self.height = height
            self.isopen = True

        nchannels = arr.shape[-1]
        if nchannels == 1:
            _format = "I"
        elif nchannels == 3:
            _format = "RGB"
        else:
            raise ValueError(f"Expected nchannels to be either 1 or 3, got {nchannels}")
        image = pyglet.image.ImageData(self.width, self.height, _format, arr.tobytes())

        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        image.blit(0, 0)
        self.window.flip()

    def close(self):
        if self.isopen:
            self.window.close()
            self.isopen = False

    def __del__(self):
        self.close()
