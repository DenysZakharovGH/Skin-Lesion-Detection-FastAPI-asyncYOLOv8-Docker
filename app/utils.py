import cv2

from config import COLORS


def draw_detection(img, box, label, conf, class_name):
    x1, y1, x2, y2 = map(int, box)
    color = COLORS.get(class_name, (0, 255, 0))

    cv2.rectangle(
        img,
        (int(x1), int(y1)),
        (int(x2), int(y2)),
        color,
        2
    )

    # Draw label background
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(
        img,
        (int(x1), int(y1) - h - 6),
        (int(x1) + w, int(y1)),
        color,
        -1
    )

    # Draw label text
    cv2.putText(
        img,
        label,
        (int(x1), int(y1) - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 0),
        1
    )

    return img