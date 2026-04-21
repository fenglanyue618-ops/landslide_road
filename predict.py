from ultralytics import YOLO
import os

# 1. 加载你刚刚训练好的最优权重 (注意路径要对)
# 默认通常保存在 runs/detect/UY_Net_Detection/weights/best.pt
model = YOLO('runs/detect/UY_Net_Detection/weights/best.pt')

# 2. 指定你的纯净测试集文件夹路径
test_images_dir = 'UY_Net_Dataset/images/test'

# 3. 运行推理，并强制保存可视化结果 (save=True)
print("🚀 开始对测试集进行逐张推理并保存可视化结果...")
results = model.predict(
    source=test_images_dir,
    conf=0.25,        # 置信度阈值 (低于这个概率的框会被过滤)
    iou=0.45,         # NMS 重叠阈值 (防止同一个滑坡画好几个框)
    save=True,        # 【关键】把画好框的图片保存下来！
    save_txt=True,    # 可选：把预测出来的坐标也保存成 txt
    project='runs/detect',
    name='UY_Net_Test_Results', # 结果保存的文件夹名称
    show=False
)

print(f"✅ 推理完成！请去 runs/detect/UY_Net_Test_Results/ 目录下查看所有画好框的测试集图片。")