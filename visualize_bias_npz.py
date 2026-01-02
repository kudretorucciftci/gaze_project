import numpy as np
import cv2

PROFILE_PATH = "implicit_bias.npz"
SCALE = 600   # pencere boyutu

data = np.load(PROFILE_PATH)
bx = data["bx"]
by = data["by"]

# magnitude (toplam etki)
mag = np.sqrt(bx**2 + by**2)

def normalize(img):
    img = img - img.min()
    if img.max() > 0:
        img = img / img.max()
    return img

# normalize
bx_n = normalize(bx)
by_n = normalize(by)
mag_n = normalize(mag)

# resize
def up(img):
    return cv2.resize(img, (SCALE, SCALE), interpolation=cv2.INTER_NEAREST)

bx_img = up(bx_n)
by_img = up(by_n)
mag_img = up(mag_n)

# color maps
bx_c = cv2.applyColorMap((bx_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
by_c = cv2.applyColorMap((by_img * 255).astype(np.uint8), cv2.COLORMAP_JET)
mag_c = cv2.applyColorMap((mag_img * 255).astype(np.uint8), cv2.COLORMAP_INFERNO)

# labels
cv2.putText(bx_c, "Bias X (Horizontal)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.putText(by_c, "Bias Y (Vertical)", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
cv2.putText(mag_c, "Bias Magnitude", (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

cv2.imshow("Bias X", bx_c)
cv2.imshow("Bias Y", by_c)
cv2.imshow("Bias Magnitude", mag_c)

print("Bias analysis running.")
print("Close windows or press ESC to exit.")

while True:
    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()

