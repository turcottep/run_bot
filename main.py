import os
import time
import cv2
import mss
import numpy as np
import pynput
from pynput.keyboard import Key
import pytesseract
# from keras.models import Sequential, load_model


keys_pressed = {
    "Key.up": False,
    "Key.left": False,
    "Key.right": False,
}

run = True


def main():
    print("AI ON")
    keyboard_controller = pynput.keyboard.Controller()
    mouse_controller = pynput.mouse.Controller()

    # listen for key presses

    def on_press(key):
        # print("\nkey pressed", key, end="\n\n")

        if key == Key.esc:
            print("esc pressed")
            listener.stop()
            global run
            run = False
            return False

        for key_from_dict in keys_pressed:
            if key_from_dict == str(key):
                keys_pressed[key_from_dict] = True

    def on_release(key):
        # print("\nkey released", key, end="\n")

        for key_from_dict in keys_pressed:
            if key_from_dict == str(key):
                keys_pressed[key_from_dict] = False

    listener = pynput.keyboard.Listener(
        on_press=on_press,
        on_release=on_release)
    listener.start()

    # main loop
    millis_old = time.time()
    i = 0

    predictions = [
        "left", "left_up", "noop", "right", "right_up", "up"
    ]

    # model = load_model("model_v4.h5")
    image_resolution = 256

    left_light_image = cv2.imread("light_small.png")

    fps = 0
    fps_alpha = 0.9

    with mss.mss() as sct:
        monitor_1024 = {"top": -1050, "left": 1300, "width": 1024, "height": 1024}
        monitor_full = {"top": -1080, "left": 860, "width": 1920, "height": 1080}
        monitor_score = {"top": -1040, "left": 930, "width": 350, "height": 85}
        score = 0

        image = None
        alive = False
        # get current date with milliseconds
        run_name = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        print("run_name", run_name)
        # create folder for this run
        os.mkdir("detection/" + run_name)
        while run:
            elapsed = time.time() - millis_old
            millis_old = time.time()

            # mouse position
            # print("mouse position", mouse_controller.position, elapsed, end="\r")

            # if (i % 2 == 0):
            #     keyboard_controller.press(Key.up)
            #     keyboard_controller.press(Key.left)

            # else:
            #     keyboard_controller.release(Key.up)
            #     keyboard_controller.release(Key.left)

            # if i > 0:
            #     name = "training/"
            #     if (keys_pressed["Key.left"]):
            #         name += "left"
            #         if (keys_pressed["Key.up"]):
            #             name += "_up"
            #     elif (keys_pressed["Key.right"]):
            #         name += "right"
            #         if (keys_pressed["Key.up"]):
            #             name += "_up"
            #     elif (keys_pressed["Key.up"]):
            #         name += "up"
            #     else:
            #         name += "noop"
            #     name += "/" + str(random.randint(0, 100000000000000000)) + ".png"
            #     cv2.imwrite(name, image)

            image = np.array(sct.grab(monitor_full))
            # cv2.imwrite("last.png", image)
            # image = np.array(

            # sct.shot(mon=2, output="last.png")

            # cv2.imshow("image", image)
            # cv2.waitKey(1)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # # get average color rgb
            # area_for_average_color = image[100:200, 100:200]
            # average_color = np.average(area_for_average_color, axis=(0, 1))
            # average_color_rgb = [average_color[2], average_color[1], average_color[0]]

            # dead_color_rgb = [187, 132, 255]
            # alive_color_rgb = [47, 100, 83]

            # if np.allclose(average_color_rgb, dead_color_rgb, atol=20):
            #     alive = False
            #     # time.sleep(1)

            # if np.allclose(average_color_rgb, alive_color_rgb, atol=20):
            #     alive = True

            # get location of left light
            # light_location_result = cv2.matchTemplate(image, left_light_image, cv2.TM_SQDIFF_NORMED)
            # _, _, light_loc_top_left, _ = cv2.minMaxLoc(light_location_result)
            # light_bottom_right = (light_loc_top_left[0] + left_light_image.shape[1], light_loc_top_left[1] + left_light_image.shape[0])
            # cv2.rectangle(image, light_loc_top_left, light_bottom_right, 255, 2)

            # cv2.imshow("image", image)
            # cv2.waitKey(1)

            # # move mouse to light slowly
            # light_pos_screen = (light_loc_top_left[0] + 1300, light_loc_top_left[1] - 1050)
            # # mouse_controller.position = light_pos_screen
            # mouse_controller.move(light_pos_screen[0] - mouse_controller.position[0], light_pos_screen[1] - mouse_controller.position[1])

            # cv2.imwrite("detection/" + run_name + "/" + run_name + str(i) + ".png", image)

            # text = pytesseract.image_to_string(image)
            # text_clean = text.replace("\n", "")
            # text_clean = text_clean.replace(" ", "")
            # # keep only numbers
            # text_clean = ''.join(filter(str.isdigit, text_clean))
            # if text_clean != "":
            #     new_score = int(text_clean)
            #     if new_score > score:
            #         if score < 1_000_000:
            #             score = new_score
            fps = (1 / elapsed) * (1 - fps_alpha) + fps * fps_alpha

            print("time: ", format(elapsed, ".3f"), "s", "fps: ", format(fps, ".0f"), end="\n\n")

            # image = cv2.resize(image, (image_resolution, image_resolution))
            # image = np.array(image)
            # image = image.reshape(1, image_resolution, image_resolution, 3)
            # image = image / 255
            # millis_after_image_processing = time.time()

            # prediction = model.predict(image, verbose=0)
            # millis_after_prediction = time.time()
            # prediction_text = predictions[np.argmax(prediction)]

            # print("AI input: ", prediction_text, "               ", end="\r")

            # if prediction_text == "left":
            #     keyboard_controller.press(keyboard_controller._Key.left)
            #     keyboard_controller.release(keyboard_controller._Key.right)
            #     keyboard_controller.release(keyboard_controller._Key.up)
            # elif prediction_text == "left_up":
            #     keyboard_controller.press(keyboard_controller._Key.left)
            #     keyboard_controller.press(keyboard_controller._Key.up)
            #     keyboard_controller.release(keyboard_controller._Key.right)
            # elif prediction_text == "noop":
            #     keyboard_controller.release(keyboard_controller._Key.left)
            #     keyboard_controller.release(keyboard_controller._Key.right)
            #     keyboard_controller.release(keyboard_controller._Key.up)
            # elif prediction_text == "right":
            #     keyboard_controller.press(keyboard_controller._Key.right)
            #     keyboard_controller.release(keyboard_controller._Key.left)
            #     keyboard_controller.release(keyboard_controller._Key.up)
            # elif prediction_text == "right_up":
            #     keyboard_controller.press(keyboard_controller._Key.right)
            #     keyboard_controller.press(keyboard_controller._Key.up)
            #     keyboard_controller.release(keyboard_controller._Key.left)
            # elif prediction_text == "up":
            #     keyboard_controller.press(keyboard_controller._Key.up)
            #     keyboard_controller.release(keyboard_controller._Key.left)
            #     keyboard_controller.release(keyboard_controller._Key.right)

            # time.sleep(0.01)
            i += 1

    # exit
    cv2.destroyAllWindows()
    sct.close()


if __name__ == "__main__":
    main()
