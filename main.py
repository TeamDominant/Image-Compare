import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

print("Расчет для Замка")

# Загрузка изображений
original = cv2.imread('castle.jpg', cv2.IMREAD_GRAYSCALE)
processed = cv2.imread('flt_castle.jpg', cv2.IMREAD_GRAYSCALE)

# 1. Расчёт ПОСШ (PSNR - сигнал/шум)
mse = np.mean((original - processed) ** 2)
max_pixel = 255.0
psnr = 10 * np.log10(max_pixel**2 / mse)

# 2. Норма Минковского
p = 2  # Параметр нормы (например, Евклидова норма)
minkowski_norm = np.sum(np.abs(original - processed) ** p) ** (1 / p)

# 3. МСП
ssim_index, _ = ssim(original, processed, full=True)

# 4. Средний контраст
mean_contrast = np.std(processed)

# 5. Среднеквадратичное отклонение (СКО)
std_dev = np.std(processed)

# 6. Коэффициент восстановимости изображения (Cri)
cri = psnr / np.max(psnr)

# Вывод результатов
print(f"ПОСШ: {psnr:.6f}")
print(f"Норма Минковского (p={p}): {minkowski_norm:.6f}")
print(f"Мера структурного подобия: {ssim_index:.6f}")
print(f"Средний контраст: {mean_contrast:.6f}")
print(f"СКО (Стандартное отклонение): {std_dev:.6f}")
print(f"Коэффициент восстановимости изображения (Cri): {cri:.6f}")
