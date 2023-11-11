
import multiprocessing
import random
import time
import cv2
import mss
import numpy as np

fps = 0
fps_alpha = 0.9

start_time = time.time()

display_time = 1/60


def GRABMSS_screen(q):
    print("GRABMSS_screen started...")
    global fps, start_time
    with mss.mss() as sct:

        # monitor_1024 = {"top": -1050, "left": 1300, "width": 1024, "height": 1024}
        # monitor_720 = {"top": -900, "left": 1100, "width": 1280, "height": 720}
        monitor = sct.monitors[1]
        top = monitor["top"]
        left = monitor["left"]
        monitor_360 = {"top": top, "left": left, "width": 640, "height": 360}

        while True:
            # print("GRABMSS_screen loop...")
            # Get raw pixels from the screen, save it to a Numpy array
            img = np.array(sct.grab(monitor_360))
            # To get real color we do this:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            elapsed_time = time.time() - start_time

            # wait so that total time is 1/60 second
            if elapsed_time < display_time:
                time.sleep(display_time - elapsed_time)

            # if current_time >= display_time:
            fps = (1 / elapsed_time) * (1 - fps_alpha) + fps * fps_alpha
            print("time: ", format(elapsed_time, ".3f"), "s", "fps: ", format(fps, ".0f"), "  ", end="\n")
            # fps = 0
            start_time = time.time()

            q.put_nowait(img)
            q.join()


def SHOWMSS_screen(q, name):
    while True:
        if not q.empty():
            img = q.get()
            q.task_done()
            # To get real color we do this:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # # Display the picture
            # # cv2.imshow("image", img)
            # # Display the picture in grayscale
            # fps += 1

            file_name = "test/" + str(random.randint(0, 100000000000000000)) + ".png"
            cv2.imwrite(file_name, img)
            # time.sleep(1)
            # # Press "q" to quit
            # if cv2.waitKey(25) & 0xFF == ord("q"):
            #     cv2.destroyAllWindows()
            #     break


if __name__ == "__main__":
    # Queue
    q = multiprocessing.JoinableQueue()

    # creating new processes
    producer = multiprocessing.Process(target=GRABMSS_screen, args=(q, ))
    consumers = []
    for i in range(2):
        consumer = multiprocessing.Process(target=SHOWMSS_screen, args=(q, i))
        consumers.append(consumer)

    # starting our processes
    producer.start()
    for consumer in consumers:
        consumer.start()

    # wait until processes are finished
    producer.join()
    for consumer in consumers:
        consumer.join()
