import cv2
import pytesseract


# 配置 Tesseract 的路径（如果需要）
# 在 Windows 上可能需要指定 Tesseract 的安装路径
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def ocr_image(image_path):
    # 加载图像
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load image.")
        return

    # 转换为灰度图像
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 图像预处理（可选）：二值化
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 可选：去噪
    denoised = cv2.fastNlMeansDenoising(binary, h=10)

    # 使用 Tesseract 进行 OCR 检测
    text = pytesseract.image_to_string(denoised, lang='eng', config='--psm 6')

    # 打印识别结果
    print("Detected Text:")
    print(text)

    # 显示预处理后的图像（可选）
    cv2.imshow("Processed Image", denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# 示例调用
if __name__ == "__main__":
    # 替换为你的图像路径
    image_path = r"C:\Users\Public\zcj\yolov7\yolov7main\datasets\ball_combine\test_video2\frame_0000.jpg"
    ocr_image(image_path)

