import os
import sys
import cv2

# 支持的图片扩展名（不区分大小写）
IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

def process_image(img_path: str, output_path: str) -> None:
    """
    读取图片，执行灰度→三通道转换，并保存到 output_path。
    """
    img = cv2.imread(img_path)
    if img is None:
        print(f"⚠️ 无法读取图片: {img_path}")
        return

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_3channel = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    cv2.imwrite(output_path, img_3channel)
    print(f"✅ 已处理: {img_path} → {output_path}")

def process_folder(input_folder: str, output_folder: str = None, overwrite: bool = False) -> None:
    """
    递归遍历 input_folder，处理所有图片。
    - 若指定 output_folder，则在其中保持原目录结构输出；
    - 若未指定，且 overwrite=True，则直接覆盖原文件（谨慎使用）；
    - 若未指定，且 overwrite=False（默认），则在原文件同目录生成 _gray 后缀的新文件。
    """
    if output_folder:
        os.makedirs(output_folder, exist_ok=True)
        
    print(111)

    for root, dirs, files in os.walk(input_folder):
        print(222)
        for file in files:
            if not file.lower().endswith(IMAGE_EXTENSIONS):
                continue

            img_path = os.path.join(root, file)

            if output_folder:
                # 输出到指定目录，保持相对路径
                rel_path = os.path.relpath(root, input_folder)
                out_dir = os.path.join(output_folder, rel_path)
                os.makedirs(out_dir, exist_ok=True)
                out_path = os.path.join(out_dir, file)
            else:
                if overwrite:
                    out_path = img_path  # 直接覆盖
                else:
                    # 添加 _gray 后缀
                    base, ext = os.path.splitext(file)
                    new_file = base + "_gray" + ext
                    out_path = os.path.join(root, new_file)

            process_image(img_path, out_path)

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python script.py <输入文件夹> [输出文件夹]")
        print("  示例1: python script.py ./images          # 在每张图片旁生成 _gray 版本")
        print("  示例2: python script.py ./images ./output # 所有处理后图片保存到 ./output 并保留目录结构")
        sys.exit(1)

    input_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else None

    # 如需直接覆盖原图，可将 overwrite 设为 True（非常危险，请备份！）
    print(input_dir, output_dir)
    process_folder(input_dir, output_dir, overwrite=False)