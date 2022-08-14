import cv2
import br_config
from br_removeBG import bg_remove
from br_getImage import getIMage
from br_barcode_reader import barcode


def main():
    cam = cv2.VideoCapture(br_config.capture)

    while cam.isOpened:
        ret, frame = cam.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        copy_frame = frame.copy()

        try:
            info = barcode(img=frame)

            if info is not None:
                getIMage(image_url=info)
        except:
            pass

        try:
            background_img = cv2.imread(br_config.file_path+br_config.filename+br_config.img_type)
            output_frame = bg_remove(img=copy_frame, imgBG=background_img)

            cv2.imshow('Result', output_frame)
        except:
            pass

        cv2.imshow('Copy frame', copy_frame)

        if cv2.waitKey(16) == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()