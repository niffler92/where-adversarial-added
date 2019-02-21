
__all__ = ['occ_center']


class Occlusion:
    def __init__(self, ratio=0.1, args=None, **kwargs):
        self.ratio = ratio
        self.args = args

    def generate(self, images, labels):
        w, h = images.size(-2), images.size(-1)
        w_start = int(w/2 - w*self.ratio/2)
        w_end = int(w/2 + w*self.ratio/2)
        h_start = int(h/2 - h*self.ratio/2)
        h_end = int(h/2 + h*self.ratio/2)

        adv_images = images.clone()
        adv_images[:,:,w_start:w_end,h_start:h_end] = 0
        
        return adv_images


def occ_center(model, args, **kwargs):
    return Occlusion(ratio=0.3, args=args, **kwargs)
