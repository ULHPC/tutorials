
# https://towardsdatascience.com/object-detection-with-10-lines-of-code-d6cb4d86f606

from imageai.Detection import ObjectDetection
import os
import sys

def object_recognition(src_image, output_directory):
    execution_path = os.getcwd()
    detector = ObjectDetection()
    detector.setModelTypeAsRetinaNet()
    detector.setModelPath( os.path.join(execution_path , "resnet50_coco_best_v2.0.1.h5"))
    detector.loadModel()
    detections = detector.detectObjectsFromImage(input_image=src_image, output_image_path=os.path.join(output_directory , os.path.basename(src_image)))
    for eachObject in detections:
        print(eachObject["name"] , " : " , eachObject["percentage_probability"] )

def main(argv):
    output_directory = argv[1]
    src_image = argv[0]
    object_recognition(src_image, output_directory)


if __name__ == '__main__':
  main(sys.argv[1:])
