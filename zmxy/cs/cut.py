import random
import cv2
import pytesseract
import numpy as np
import subprocess
import time
import logging
import easyocr
from typing import Tuple, Union

# 配置日志输出
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

jiedian = 1  # 通过的节点数
quanjugengticishu = 0  # 已经更替的次数,默认0
zj = 0  # 终劫标记

emulator_ip = "127.0.0.1"
emulator_port = "16384"
adb_connect_cmd = f"adb connect {emulator_ip}:{emulator_port}"


def init():
    try:
        result = subprocess.run(adb_connect_cmd, shell=True, capture_output=True, text=True)
        print("ADB Output:", result.stdout)
        print("ADB Error:", result.stderr)
    except Exception as e:
        print("Error:", e)



def click_button_by_image(template_path: str,
                          threshold: float = 0.8,
                          adb_path: str = "adb",
                          max_retry: int = 3,
                          device_id: str = None) -> bool:
    """
    通过图像匹配自动点击屏幕按钮

    Args:
        template_path (str): 预存的按钮模板图片路径
        threshold (float): 匹配阈值(0-1)，默认0.8
        adb_path (str): adb命令路径，默认系统环境变量中的adb
        max_retry (int): 最大重试次数，默认3次
        device_id (str): 设备ID（用于多设备连接场景）

    Returns:
        bool: 是否成功找到并点击按钮
    """

    def capture_screen() -> np.ndarray:
        """通过ADB截取屏幕"""
        try:
            # 构建adb命令
            cmd = [adb_path]
            if device_id:
                cmd.extend(["-s", device_id])
            cmd.extend(["exec-out", "screencap", "-p"])

            # 执行截屏命令
            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=5  # 设置5秒超时
            )

            if result.returncode != 0:
                logger.error(f"截屏失败: {result.stderr.decode().strip()}")
                return None

            # 转换图像数据
            img_array = np.frombuffer(result.stdout, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)

        except Exception as e:
            logger.error(f"截屏异常: {str(e)}")
            return None

    def match_template(screen: np.ndarray) -> tuple:
        """执行模板匹配"""
        try:
            template = cv2.imread(template_path)
            if template is None:
                logger.error(f"无法读取模板图片: {template_path}")
                return None, 0.0

            # 灰度化提升匹配速度
            screen_gray = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

            cv2.imshow("Screen", screen_gray)
            cv2.waitKey(0)
            cv2.imshow("Screen", template_gray)
            cv2.waitKey(0)

            res = cv2.matchTemplate(screen_gray, template_gray, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            return max_loc, max_val

        except Exception as e:
            logger.error(f"模板匹配异常: {str(e)}")
            return None, 0.0

    def perform_click(center_x: int, center_y: int) -> bool:
        """执行点击操作"""
        try:
            cmd = [adb_path]
            if device_id:
                cmd.extend(["-s", device_id])
            cmd.extend(["shell", "input", "tap", str(center_x), str(center_y)])

            result = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=3
            )

            if result.returncode == 0:
                return True
            logger.error(f"点击失败: {result.stderr.decode().strip()}")
            return False

        except Exception as e:
            logger.error(f"点击异常: {str(e)}")
            return False

    # 主逻辑
    for attempt in range(1, max_retry + 1):
        logger.info(f"尝试第 {attempt} 次匹配...")

        # 1. 截取屏幕
        screen = capture_screen()
        if screen is None:
            continue

        # 2. 执行模板匹配
        max_loc, max_val = match_template(screen)
        if max_loc is None:
            continue

        logger.debug(f"最高匹配值: {max_val:.2f}")

        # 3. 判断是否超过阈值
        if max_val < threshold:
            logger.info(f"匹配值{max_val:.2f}低于阈值{threshold}")
            time.sleep(1)  # 等待界面稳定
            continue

        # 4. 计算点击坐标
        template_img = cv2.imread(template_path)
        h, w = template_img.shape[:2]
        center_x = max_loc[0] + w // 2
        center_y = max_loc[1] + h // 2

        # 5. 执行点击
        if perform_click(center_x, center_y):
            logger.info(f"成功点击坐标 ({center_x}, {center_y})")
            return True

        time.sleep(0.5)  # 点击失败后短暂等待

    logger.warning(f"未找到按钮，已达到最大重试次数{max_retry}")
    return False


# 使用示例
# if __name__ == "__main__":
#     # 配置调试模式显示详细信息
#     logger.setLevel(logging.DEBUG)
#
#     success = click_button_by_image(
#         template_path="sl.png",
#         threshold=0.30,
#         max_retry=5
#     )
#
#     print(f"操作结果: {'成功' if success else '失败'}")


def preprocess_image(img):
    """图像预处理：降噪、锐化、缩放"""
    # 高斯模糊降噪
    img = cv2.GaussianBlur(img, (3, 3), 0)
    # 锐化
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    # 缩放提高分辨率（针对小文字）
    img = cv2.resize(img, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    return img


def ocr_and_click(target_text, region=None, lang='chi_sim'):
    """
    识别屏幕中指定区域的文字，并点击目标文本
    - target_text: 要识别的目标文字（如"确定"）
    - region: 截屏区域 (x, y, width, height)，若为None则全屏识别
    - lang: OCR语言（英文'eng'，中文'chi_sim'）
    """
    # 1. 截取屏幕
    img = capture_screen(region)

    # # 2. 转换为灰度图并增强对比度
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    # # 在OCR前调用预处理
    # gray = preprocess_image(gray)

    # cv2.imshow("Screen", gray)
    # cv2.waitKey(0)
    # 3. OCR识别文字和位置
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)

    print(data)
    # 4. 遍历识别结果，寻找目标文本
    for i in range(len(data['text'])):
        if data['conf'][i] > 60:  # 置信度阈值（0-100）
            if target_text in data['text'][i]:
                # 获取文字位置
                x = data['left'][i]
                y = data['top'][i]
                w = data['width'][i]
                h = data['height'][i]

                # 计算中心坐标（如果是区域截图，需加上区域偏移）
                click_x = x + w // 2
                click_y = y + h // 2
                if region:
                    click_x += region[0]
                    click_y += region[1]

                # 执行点击
                subprocess.run(f"adb shell input tap {click_x} {click_y}", shell=True)
                return True
    return False


def capture_screen(region=None):
    """截取屏幕（全屏或指定区域）"""
    # 通过ADB截屏
    subprocess.run("adb exec-out screencap -p > screen.png", shell=True)
    img = cv2.imread("screen.png")

    # 裁剪区域
    if region:
        x, y, w, h = region
        img = img[y:y + h, x:x + w]
    return img


def adb_screenshot_to_cv():
    """直接获取截图并转换为OpenCV格式（无文件写入）"""
    try:
        # 获取二进制截图数据
        result = subprocess.run(
            ["adb", "exec-out", "screencap", "-p"],
            stdout=subprocess.PIPE,
            check=True
        )
        # 转换为numpy数组
        img_array = np.frombuffer(result.stdout, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
    except Exception as e:
        print(f"截图失败: {e}")
        return None


def adb_click(x, y, delay_range=(0.2, 0.3)):
    """点击屏幕坐标 (x, y)，并添加随机延迟防检测"""
    # 执行ADB点击命令
    subprocess.run(f"adb shell input tap {x} {y}", shell=True)
    # 添加随机延迟（例如0.1~0.5秒）
    time.sleep(random.uniform(*delay_range))


# """造梦西游游戏应用位置"""
# adb_click(302, 652)
#
#
# """破虚左1,y范围200-400,x400-600"""
# adb_click(520, 330)
#
#
# """中,x范围800-1000"""
# adb_click(950, 340)
#
#
# """右2,x1300-1500"""
# adb_click(1404, 336)


# print(ocr_and_click("烦", region=(345, 460, 1200, 200), lang='chi_sim'))

#
#
# # 使用示例
# image = adb_screenshot_to_cv()
# if image is not None:
#     cv2.imshow("Screen", image)
#     cv2.waitKey(0)


# # 初始化阅读器（中文+英文）
# reader = easyocr.Reader(['ch_sim', 'en'])
#
# # 读取图像
# image = capture_screen(region=(345, 460, 1200, 200))
#
# # 执行OCR识别
# results = reader.readtext(image)
#
# # 输出结果
# for (bbox, text, prob) in results:
#     print(f'坐标: {bbox}, 文本: {text}, 置信度: {prob:.2f}')
def preproimcess(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]


def find_text_position(target_text: str,
                       region: Tuple[int, int, int, int] = None,
                       confidence_threshold: float = 0.2) -> Union[Tuple[int, int], None]:
    """
    通过OCR查找指定文本在屏幕上的坐标位置，返回置信度最高的结果

    Args:
        target_text (str): 要查找的文本内容
        region (Tuple[int, int, int, int]): 截图区域 (x, y, width, height)
        confidence_threshold (float): 最低置信度阈值 (0-1)

    Returns:
        Tuple[int, int] | None: 置信度最高的坐标 (x,y)，未找到返回None
    """
    reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)

    try:
        image = capture_screen(region)
        results = reader.readtext(image)

        max_prob = confidence_threshold  # 初始化为最低阈值
        best_position = None

        for (bbox, text, prob) in results:
            # 双重验证 + 置信度比较
            if target_text in text and prob >= max_prob:
                # 更新最高置信度和坐标
                max_prob = prob
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]
                best_position = (
                    int(sum(x_coords) / 4),
                    int(sum(y_coords) / 4)
                )

                return best_position  # 可能为None

    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"截屏失败: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"OCR识别异常: {str(e)}")


def safe_click(x: int, y: int, max_offset: int = 5):
    """带随机偏移的点击"""
    x += random.randint(-max_offset, max_offset)
    y += random.randint(-max_offset, max_offset)
    # print(x, y)
    adb_click(x, y)


# safe_click(1800, 80)  # 测试


# def rujie():
#     safe_click(952, 779)
#
#
def gengti():
    safe_click(1800, 770)


def adb_long_click(x: int, y: int, duration_ms: int):
    """ADB 长按操作
    Args:
        x, y (int): 点击坐标
        duration_ms (int): 持续时间（毫秒）
        :param duration_ms: 时间
        :param y: dy
        :param x: dx
    """
    subprocess.run(
        ["adb", "shell", "input", "swipe",
         str(x), str(y), str(x), str(y),
         str(duration_ms)]
    )


#
# def querengengti():
#     safe_click(1100, 500)

#
# positions = find_text_position("烦恼障", region=None)
# print("检测到的坐标:", positions)
#
# # 判断坐标是否存在，并进行相应操作
# if positions:
#     # 如果找到目标，点击检测到的坐标
#     adb_click(*positions)
#     print(f"已点击目标坐标: {positions}")
# else:
#     # 如果未找到目标，点击备用坐标（例如 (100, 200)）1100,500
#     backup_position = (1800, 770)  # 修改为你需要的备用坐标
#     adb_click(*backup_position)
#     time.sleep(1)
#     adb_click(1100, 500)
#     print(f"未找到目标，已点击更替: {backup_position}")
# adb_click(1796,74)


# ---------- 基础工具函数 ----------
def dengdai_shijian(ms):
    """更替的等待时间秒记"""
    time.sleep(ms)


def dianji_queren():
    """通关成功,点击确认"""
    safe_click(956, 680)


def zaici_tiaozhan():
    """再次挑战"""
    safe_click(1135, 677)
    safe_click(1135, 677)


def queren_gengti():
    """确认更替"""
    safe_click(1100, 500)


def xuanze_diyi_guan():
    """选择第一关"""
    safe_click(513, 345)


def panduan_zhandoucg():
    """截图判断战斗成功情况"""
    re = find_text_position("奖励")
    if re:
        print(f"战斗成功{re}")
    else:
        print("战斗失败")
    return re


# panduan_zhandoucg()  # 测试


def zhixing_zhandou():
    """战斗点击技能"""
    safe_click(991, 791)  # 光暗
    safe_click(122, 576)  # 真武
    safe_click(666, 788)  # 化身
    adb_long_click(349, 763, 4500)  # 右移
    safe_click(164, 764)  # 左移一下
    safe_click(164, 764)  # 左移一下
    safe_click(164, 764)  # 左移一下
    safe_click(1798, 374)  # 无双
    safe_click(115, 428)  # 剑阵
    safe_click(115, 428)  # 剑阵
    safe_click(1261, 656)  # 法宝
    safe_click(1152, 792)  # 时空
    safe_click(1152, 792)  # 时空
    # safe_click(1302, 792)  # 风眼
    # time.sleep(0.1)
    # safe_click(1302, 792)  # 风眼
    safe_click(828, 798)  # 娃娃
    safe_click(828, 798)  # 娃娃
    time.sleep(0.4)
    safe_click(1471, 682)  # 树人
    safe_click(1471, 682)  # 树人
    safe_click(991, 791)  # 光暗
    time.sleep(0.4)
    adb_long_click(1468, 791, 500)  # 强力击
    time.sleep(0.4)
    adb_long_click(1468, 791, 500)  # 强力击
    time.sleep(0.4)
    adb_long_click(1468, 791, 500)  # 强力击
    time.sleep(0.4)
    adb_long_click(1468, 791, 500)  # 强力击
    time.sleep(2)
    # 截图进行对比查看是否通关成功
    return panduan_zhandoucg()


def ru_jie():
    """入劫"""
    global jiedian
    safe_click(952, 779)  # 入劫按钮
    dengdai_shijian(4.1)  # 等待5秒加载
    print(f"进入节点{jiedian}")
    # zhandou_jieguo = zhixing_zhandou()  # 战斗点击技能
    while True:  # 战斗点击技能
        # print("准备挑战")
        result = zhixing_zhandou()  # 获取 zhixing_zhandou() 的返回值
        if result is not None:  # 如果返回值不是 None，跳出循环
            break
        zaici_tiaozhan()  # 再次挑战
        dengdai_shijian(4.1)  # 加载等待

    if result:
        # 战斗成功分支
        dianji_queren()  # 点击确认
        jiedian += 1
        print(f"下一个节点为{jiedian}")
        dengdai_shijian(5)  # 加载等待
        # jiedian = 1
        # quanjugengticishu = 0
        # dengdai_shijian(6)  # 加载等待
    # if zhandou_jieguo:
    #     # 战斗成功分支
    #     dianji_queren()  # 点击确认
    #     jiedian += 1
    #     dengdai_shijian(6)  # 加载等待
    #
    # else:
    #     # 战斗失败分支
    #     while zhixing_zhandou() is None:
    #         zaici_tiaozhan()  # 再次挑战
    #         dengdai_shijian(4.5)  # 加载等待
    #         zhixing_zhandou()  # 点击战斗技能


# zhixing_zhandou() 战斗测试


def ocr_shibie_fannaozhang():
    """OCR识别烦恼障"""
    position = find_text_position("烦", region=None)
    position1 = find_text_position("恼", region=None)
    if position:
        print("识别成功")
        return position
    if position1:
        print("识别成功")
        return position1


# ocr_shibie_fannaozhang()

def chuli_shibie_chenggong(po):
    """处理识别成功,点击烦恼障"""
    adb_click(*po)
    ru_jie()  # 入劫


def chuli_shibie_shibai():
    """处理识别失败"""
    global quanjugengticishu

    if quanjugengticishu == 0:  # 免费次数用完了
        print("免费次数用完了,选择第一关进入")
        xuanze_diyi_guan()  # 选择第一关进入
        ru_jie()  # 入劫

    else:
        gengti()  # 更替
        dengdai_shijian(1)  # 等待1秒
        queren_gengti()  # 确认更替
        quanjugengticishu -= 1
        print(f"免费更替第{quanjugengticishu - 3}次")


def zhongjie():
    global jiedian
    print("进入终劫")
    ru_jie()


def tianmo():
    global jiedian, quanjugengticishu
    print("准备进入天魔")
    safe_click(91, 767)  # 挑战天魔按钮
    time.sleep(1)
    safe_click(1793, 711)  # 进入天魔按钮
    dengdai_shijian(4.4)  # 等待5秒加载

    while True:  # 战斗点击技能
        # print("准备挑战")
        result = zhixing_zhandou()  # 获取 zhixing_zhandou() 的返回值
        if result is not None:  # 如果返回值不是 None，跳出循环
            break
        zaici_tiaozhan()  # 再次挑战按钮
        dengdai_shijian(4.5)  # 加载等待

    if result:
        # 战斗成功分支
        safe_click(1179, 683)  # 前往新一层
        safe_click(1179, 683)  # 前往新一层
        jiedian -= 6
        quanjugengticishu = 3
        dengdai_shijian(5)  # 加载等待

    # else:
    #     # 战斗失败分支
    #     while zhixing_zhandou() is None:
    #         print("准备再次挑战")
    #         zaici_tiaozhan()  # 再次挑战
    #         dengdai_shijian(4.5)  # 加载等待
    #         zhixing_zhandou()  # 点击战斗技能


def zhu_xunhuan():
    """主循环,过图模式,从下一直往上打直到打不过"""
    while True:
        if jiedian == 6:
            zhongjie()
        if jiedian == 7:
            tianmo()
        po = ocr_shibie_fannaozhang()
        if po:
            # OCR识别烦恼障 → 识别成功
            chuli_shibie_chenggong(po)
        else:
            # OCR识别烦恼障 → 识别失败
            chuli_shibie_shibai()


def cccooo(count):
    """重复挑战多少次,传入度牒数量,直到度牒用完"""
    global jiedian, quanjugengticishu
    for i in range(count+1):
        if jiedian == 6:
            zhongjie()
            jiedian -= 6
            quanjugengticishu = 3
        po = ocr_shibie_fannaozhang()
        if po:
            # OCR识别烦恼障 → 识别成功
            chuli_shibie_chenggong(po)
        else:
            # OCR识别烦恼障 → 识别失败
            chuli_shibie_shibai()


if __name__ == '__main__':
    jiedian = 1  # 当前节点
    quanjugengticishu = 3  # 能使用几次更替
    # zhu_xunhuan()  # 过图
    cccooo(45)  # 用度牒重复刷舍利子












