import unittest
from keras.applications.resnet50 import ResNet50
import app

class TestApp(unittest.TestCase):
    def test_feature_extractor(self):
        url = "https://en.cf-vanguard.com/jsp-material/cardimages/vbt01_001.png"
        model = ResNet50(weights='imagenet', include_top=False, pooling='max')
        features = app.FeaturesFromUrl(model, url)
        self.assertTrue(features.shape == (1,2048))

    def test_flist(self):
        model = ResNet50(weights='imagenet', include_top=False, pooling='max')
        flist,idlist = app.GetFeatureList(model)
        print(len(flist))
        self.assertTrue(len(flist) == 3814)
        self.assertTrue(len(flist[0]) ==2048)
        
if __name__=='__main__':
    unittest.main()
