from sklearn import datasets
import cv2

from sklearn import svm

def Test():
    iris = datasets.load_iris()
    digits = datasets.load_digits()

    clf = svm.SVC(gamma=0.001, C=100.)

    train_images = digits.data[:-1]
    train_lables = digits.target[:-1]

    clf.set_params(kernel='linear').fit(train_images, train_lables)  
    # clf.set_params(kernel='rbf')..fit(train_images, train_lables)  
    # clf.fit(train_images,train_lables)  

    test_image  = digits.data[-1:]
    res = clf.predict(test_image)

    print "shape of image = ",train_images.shape
    print "shape of lables = ",train_lables.shape
    print "shape of test_image = ",test_image.shape

    print "res = ",res

    cv2.namedWindow('test_image',0)
    test_image = test_image.reshape(8,8)/test_image.max()
    cv2.imshow('test_image',test_image)
    cv2.waitKey(0)

if __name__ == '__main__':
    
    Test()