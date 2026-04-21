import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO(r'E:\yolo\ultralytics-main\ultralytics\cfg\models\11\yolo11.yaml')
    model.train(data=r'E:\yolo\ultralytics-main\datasets\data.yaml',
                cache=False,
                imgsz=640,
                epochs=120,
                single_cls=False,
                batch=6,
                close_mosaic=0,
                workers=0,
                device='cpu',
                optimizer='SGD',
                amp=False,
                project='runs/train',
                name='open2exp',
                )








