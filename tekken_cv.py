import os
import cv2
import pytesseract
import numpy as np
import imutils
import subprocess
import argparse

# Install opencv from https://stackoverflow.com/a/58991547 if using PyCharm
# Install pytesseract exe from https://github.com/UB-Mannheim/tesseract/wiki
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

rank_matcher_data = {}
mouse_first_x = -1
mouse_first_y = -1


def scrape_stream(stream, queue):
    streamlink = subprocess.Popen(f"streamlink {stream}  best -O", stdout=subprocess.PIPE)
    ffmpeg = subprocess.Popen("ffmpeg -i pipe:0 -r 0.25 -pix_fmt bgr24 -loglevel quiet -vcodec rawvideo -an -sn -f image2pipe pipe:1",
                              stdin=streamlink.stdout, stdout=subprocess.PIPE, bufsize=1920 * 1080 * 3)

    cooldown = 0
    while True:
        raw_image = ffmpeg.stdout.read(1920 * 1080 * 3)
        image = np.fromstring(raw_image, dtype='uint8')  # convert read bytes to np
        image = image.reshape((1080, 1920, 3))

        # skip those right after the card has been detected to avoid detecting it multiple times
        if cooldown > 0:
            cooldown -= 1
            continue

        region = cv2.threshold(cv2.GaussianBlur(cv2.inRange(cv2.cvtColor(
            image[49:86, 786:1132], cv2.COLOR_BGR2GRAY), 190, 230), (3, 3), 0), 0, 255, cv2.THRESH_BINARY_INV)[1]
        text = pytesseract.image_to_string(region, config='--psm 7')

        if text == "Match-Up Win Rate (World)\n\f":
            cooldown = 8
            print("Card detected!")
            queue.put(read_image(image))


def read_image(image):
    # ROI = image[y1:y2, x1:x2]
    # sharpened = cv2.filter2D(image, -1,  np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]]))
    # image_red_channel = image[:, :, 2]
    print("Reading image...", end='', flush=True)

    results = {}

    # Forsen card
    forsen_card_og_vertex = np.float32([[84, 33], [426, 62], [82, 165], [427, 188]])
    forsen_card_warp_vertex = np.float32([[0, 0], [360, 0], [0, 135], [360, 135]])
    forsen_card_warp_matrix = cv2.getPerspectiveTransform(forsen_card_og_vertex, forsen_card_warp_vertex)
    forsen_card = cv2.warpPerspective(image, forsen_card_warp_matrix, (360, 135))
    results['forsen_name'] = transcribe(threshold_white(forsen_card[0:42, 45:358]))
    results['forsen_wins'] = transcribe(threshold_white(forsen_card[68:127, 258:357]), correct_numbers=True, correct_wins=True)
    results['forsen_rank'] = rank_detection(forsen_card[59:134, 2:181])

    # Enemy card
    enemy_card_og_vertex = np.float32([[1499, 64], [1844, 33], [1500, 186], [1842, 166]])
    enemy_card_warp_vertex = np.float32([[0, 0], [360, 0], [0, 135], [360, 135]])
    enemy_card_warp_matrix = cv2.getPerspectiveTransform(enemy_card_og_vertex, enemy_card_warp_vertex)
    enemy_card = cv2.warpPerspective(image, enemy_card_warp_matrix, (360, 135))
    results['enemy_name'] = transcribe(threshold_white(enemy_card[2:44, 4:315]))
    results['enemy_wins'] = transcribe(threshold_white(enemy_card[73:122, 1:107]), correct_numbers=True, correct_wins=True)
    results['enemy_rank'] = rank_detection(enemy_card[60:134, 186:357])

    # Win rates
    results['forsen_mu_wr_world'] = transcribe(threshold_red(image[49:87, 622:715]), correct_numbers=True)
    results['forsen_mu_wr_personal'] = transcribe(threshold_red(image[104:141, 638:732]), correct_numbers=True)
    results['forsen_stage_wr'] = transcribe(threshold_red(image[158:196, 665:758]), correct_numbers=True)
    results['enemy_mu_wr_world'] = transcribe(threshold_blue(image[52:87, 1209:1299]), correct_numbers=True)
    results['enemy_mu_wr_personal'] = transcribe(threshold_blue(image[105:142, 1187:1276]), correct_numbers=True)
    results['enemy_stage_wr'] = transcribe(threshold_blue(image[158:196, 1163:1253]), correct_numbers=True)

    # Prowess
    results['forsen_prowess'] = transcribe(threshold_red(image[401:438, 98:250]))
    results['enemy_prowess'] = transcribe(threshold_blue(image[402:435, 1660:1822]))

    # Top stats
    stat_og_vertex = np.float32([[12, 0], [60, 0], [0, 50], [50, 50]])
    stat_warp_vertex = np.float32([[0, 0], [50, 0], [0, 50], [50, 50]])
    stat_warp_matrix = cv2.getPerspectiveTransform(stat_og_vertex, stat_warp_vertex)
    results['forsen_first_stat_letter'] = transcribe(gray_threshold_blur(red_background_to_black(
        cv2.warpPerspective(image[526:570, 105:200], stat_warp_matrix, (86, 44)))), correct_stat_grade=True)
    results['forsen_second_stat_letter'] = transcribe(gray_threshold_blur(red_background_to_black(
        cv2.warpPerspective(image[588:632, 155:250], stat_warp_matrix, (86, 44)))), correct_stat_grade=True)
    results['forsen_third_stat_letter'] = transcribe(gray_threshold_blur(red_background_to_black(
        cv2.warpPerspective(image[648:692, 200:295], stat_warp_matrix, (86, 44)))), correct_stat_grade=True)
    results['enemy_first_stat_letter'] = transcribe(gray_threshold_blur(blue_background_to_black(
        cv2.warpPerspective(image[525:569, 1715:1810], stat_warp_matrix, (86, 44)))), correct_stat_grade=True)
    results['enemy_second_stat_letter'] = transcribe(gray_threshold_blur(blue_background_to_black(
        cv2.warpPerspective(image[589:633, 1675:1770], stat_warp_matrix, (86, 44)))), correct_stat_grade=True)
    results['enemy_third_stat_letter'] = transcribe(gray_threshold_blur(blue_background_to_black(
        cv2.warpPerspective(image[650:694, 1625:1720], stat_warp_matrix, (86, 44)))), correct_stat_grade=True)

    results['forsen_first_stat_name'] = transcribe(threshold_white(image[524:575, 237:524]), correct_stat_name=True)
    results['forsen_second_stat_name'] = transcribe(threshold_white(image[587:638, 282:569]), correct_stat_name=True)
    results['forsen_third_stat_name'] = transcribe(threshold_white(image[648:697, 330:617]), correct_stat_name=True)
    results['enemy_first_stat_name'] = transcribe(threshold_white(image[523:572, 1397:1684]), correct_stat_name=True)
    results['enemy_second_stat_name'] = transcribe(threshold_white(image[585:635, 1350:1637]), correct_stat_name=True)
    results['enemy_third_stat_name'] = transcribe(threshold_white(image[646:698, 1304:1591]), correct_stat_name=True)

    # Previous matches
    forsen_previous_matches_coords = [[24, 55], [79, 52], [133, 49], [186, 45], [238, 42], [289, 39], [339, 35], [388, 32], [436, 28], [484, 26]]
    forsen_previous = threshold_gold(image[870:945, 118:625])
    results['forsen_previous_wr'] = sum(forsen_previous[coords[1], coords[0]] == 255 for coords in forsen_previous_matches_coords) / 10.0
    enemy_previous_matches_coords = [[20, 26], [65, 29], [112, 32], [159, 35], [207, 39], [256, 42], [306, 45], [339, 42]]
    enemy_previous = threshold_gold(image[863:926, 1183:1527])
    results['enemy_previous_wr'] = sum(enemy_previous[coords[1], coords[0]] == 255 for coords in enemy_previous_matches_coords) / 8.0

    # TODO Map, might be easier to match the image
    # map_og_vertex = np.float32([[12, 0], [377, 0], [0, 26], [365, 26]])
    # map_warp_vertex = np.float32([[0, 0], [365, 0], [0, 26], [365, 26]])
    # map_warp_matrix = cv2.getPerspectiveTransform(map_og_vertex, map_warp_vertex)
    # cv2.warpPerspective(image[905:931, 777:1141], map_warp_matrix, (353, 26))

    print("Done")
    return results


def gray_threshold_blur(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def threshold_white(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = cv2.inRange(image, 180, 240)
    image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def threshold_blue(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image, (102, 175, 180), (107, 220, 255))
    image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def threshold_red(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(image, (176, 230, 200), (180, 255, 255))
    mask2 = cv2.inRange(image, (0, 230, 200), (2, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)
    image = cv2.threshold(mask, 0, 255, cv2.THRESH_BINARY_INV)[1]
    image = cv2.GaussianBlur(image, (3, 3), 0)
    return image


def red_background_to_black(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask1 = cv2.inRange(image_hsv, (120, 50, 10), (180, 255, 52))
    mask2 = cv2.inRange(image_hsv, (0, 50, 10), (2, 255, 52))
    mask = cv2.bitwise_or(mask1, mask2)
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    image = apply_brightness_contrast(image, 20, 20)
    image[np.where(mask != [0])] = [0, 0, 0]
    return image


def blue_background_to_black(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(image_hsv, (102, 115, 10), (113, 240, 89))
    mask = cv2.GaussianBlur(mask, (3, 3), 0)
    image = apply_brightness_contrast(image, 20, 20)
    image[np.where(mask != [0])] = [0, 0, 0]
    return image


def threshold_gold(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image = cv2.GaussianBlur(image, (9, 9), 0)
    mask = cv2.inRange(image, (17, 45, 70), (60, 205, 255))
    return mask


def apply_brightness_contrast(image, brightness=0, contrast=0):
    if brightness != 0:
        if brightness > 0:
            shadow = brightness
            highlight = 255
        else:
            shadow = 0
            highlight = 255 + brightness
        alpha_b = (highlight - shadow) / 255
        gamma_b = shadow
        buf = cv2.addWeighted(image, alpha_b, image, 0, gamma_b)
    else:
        buf = image.copy()

    if contrast != 0:
        f = 131 * (contrast + 127) / (127 * (131 - contrast))
        alpha_c = f
        gamma_c = 127 * (1 - f)
        buf = cv2.addWeighted(buf, alpha_c, buf, 0, gamma_c)
    return buf


def prepare_rank_matcher_data():
    print("Preparing rank matcher...", end='', flush=True)
    for rank in range(1, 36):
        rank_template = cv2.cvtColor(cv2.imread(f"rank_images/{rank}.png"), cv2.COLOR_BGR2GRAY)
        sift = cv2.SIFT_create()
        rank_matcher_data[rank] = sift.detectAndCompute(rank_template, None)
    print("Done")


def rank_detection(rank_card):
    # https://docs.opencv.org/master/dc/dc3/tutorial_py_matcher.html
    rank_card = cv2.cvtColor(rank_card, cv2.COLOR_BGR2GRAY)

    highest_match = (0, 0)
    for rank in range(1, 36):
        sift = cv2.SIFT_create()

        kp1, des1 = sift.detectAndCompute(rank_card, None)
        kp2, des2 = rank_matcher_data[rank]

        flann_index_kdtree = 1
        index_params = dict(algorithm=flann_index_kdtree, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)

        good_matches = 0
        for k, (m, n) in enumerate(matches):
            if m.distance < 0.7 * n.distance:
                good_matches += 1

        if good_matches > highest_match[1]:
            highest_match = (rank, good_matches)
    return highest_match[0]


def transcribe(image, correct_stat_grade=False, correct_stat_name=False, correct_numbers=False, correct_wins=False):
    text = pytesseract.image_to_string(imutils.resize(image, width=400), config='--psm 7')[:-2]
    if correct_stat_grade:
        text = text.replace('5', 'D').replace('8', 'B').replace('Cc', 'C').replace('Ss', 'S')
    if correct_stat_name:
        stat_names = ['Combo Damage', 'Defensive Ability', 'Throw Countering', 'Agility', 'Rage Usage',
                      'Aggressiveness', 'Side Stepping', 'Punishment Proficiency', 'Tenacity']
        for name in stat_names:
            if name in text:
                text = name
                break
    if correct_numbers:
        text = text.replace('O', '0').replace('|', '1').replace('l', '1').replace('I', '1')
    if correct_wins:
        if text[-1] == 'W':
            text = text[:-1]
            if text[-1] == ' ':
                text = text[:-1]
    return text


def on_mouse_show_roi(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        global mouse_first_x
        global mouse_first_y
        if mouse_first_x == -1:
            mouse_first_x = x
            mouse_first_y = y
        else:
            print(f"[{mouse_first_y}:{y}, {mouse_first_x}:{x}]")
            mouse_first_x = -1


def on_mouse_show_coordinates(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(f"[{x},{y}]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--stream')
    group.add_argument('--folder')
    args = parser.parse_args()

    prepare_rank_matcher_data()
    if args.stream:
        print(f"Scraping stream with URL: {args.stream}")
        scrape_stream(args.stream)
    else:
        print(f"Reading images from folder: {args.folder}\n")
        for image_file in os.scandir(args.folder):
            print(f"Image: {image_file.name}")
            read_image(cv2.imread(image_file.path))
