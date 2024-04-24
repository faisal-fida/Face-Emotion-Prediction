import time
from helpers import detect_single_face, predict, b64_to_image


with open("cb-base64-string.txt", "r") as f:
    IMAGE = b64_to_image(f.read())


def main():
    time_start = time.time()
    face_image = detect_single_face(IMAGE)
    prediction, avg_time = predict(face_image)
    time_end = time.time()
    avg_time.append(time_end - time_start)
    print(f"Prediction: {prediction}, Time: {round(sum(avg_time) / len(avg_time), 4)}")

    # Save the face image
    # file_name = f"pred/{prediction}_{time.time()}.png"
    # cv2.imwrite(file_name, face_image)


if __name__ == "__main__":
    for _ in range(200):
        main()
