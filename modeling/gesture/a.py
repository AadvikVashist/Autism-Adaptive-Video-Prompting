        
import cv2
import noduro
super_resolution = {"2" : cv2.dnn_superres.DnnSuperResImpl_create(),
                    "3" : cv2.dnn_superres.DnnSuperResImpl_create(),
                    "4" : cv2.dnn_superres.DnnSuperResImpl_create(),
                    }
for i in range(2,5):
    super_resolution[str(i)].readModel(noduro.subdir_path("data/analyzed/upscaling/ESPCN_x" + str(i) + ".pb"))
    super_resolution[str(i)].setModel("espcn", i)
a = noduro.subdir_path("data/analyzed/ESPCN_x" + str(i) + ".pb")
image = cv2.imread("C:/Users/aadvi/Downloads/bayer.png")
im = super_resolution["2"].upsample(image)
im = super_resolution["4"].upsample(im)

cv2.imwrite("C:/Users/aadvi/Downloads/bay.png", im)