import cv2
import numpy as np
import subprocess

def adb_screenshot(filename="screen.png"):
    """通过ADB截取模拟器屏幕并保存到本地"""
    try:
        # 使用adb命令截屏并导出
        command = f"adb exec-out screencap -p > {filename}"
        subprocess.run(command, shell=True, check=True)
        print(f"截图已保存至: {filename}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"截图失败: {e}")
        return False

# 使用示例
adb_screenshot("current_screen.png")