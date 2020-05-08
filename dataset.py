from torch.utils.data import Dataset

class COCO(Dataset):

    def __init__(self, root_dir, transform=None):
        super().__init__()
        self.root_dir = root_dir
        self.tranform = transform
