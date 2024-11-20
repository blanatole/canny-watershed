import tkinter as tk
from tkinter import filedialog, messagebox, Scrollbar, Canvas
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from scipy import ndimage
from scipy.ndimage import convolve
from scipy import misc

def load_image():
    global img, segmented_img
    try:
        file_path = filedialog.askopenfilename(
            filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff *.tif")]
        )
        if file_path:
            # Kiểm tra kích thước file (dưới 10MB)
            file_size = os.path.getsize(file_path)
            if file_size > 10 * 1024 * 1024:
                messagebox.showerror("Lỗi", "Kích thước ảnh vượt quá 10MB. Vui lòng chọn ảnh nhỏ hơn.")
                return
            
            # Đọc ảnh màu
            img = cv2.imread(file_path)
            if img is None:
                messagebox.showerror("Lỗi", "Không thể tải ảnh. Vui lòng chọn một tệp ảnh hợp lệ.")
                return
            img_color_copy = img.copy()
            
            # Chuyển đổi sang thang độ xám
            def rgb2gray(rgb):
                r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
                gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
                return gray
            
            # Phát hiện cạnh bằng Canny
            class cannyEdgeDetector:
                def __init__(self, img, sigma=1, kernel_size=5, weak_pixel=75, strong_pixel=255, lowthreshold=0.05, highthreshold=0.15):
                    self.img = img
                    self.img_smoothed = None
                    self.gradientMat = None
                    self.thetaMat = None
                    self.nonMaxImg = None
                    self.thresholdImg = None
                    self.weak_pixel = weak_pixel
                    self.strong_pixel = strong_pixel
                    self.sigma = sigma
                    self.kernel_size = kernel_size
                    self.lowThreshold = lowthreshold
                    self.highThreshold = highthreshold
                
                def gaussian_kernel(self, size, sigma=1):
                    ax = np.linspace(-(size // 2), size // 2, size)
                    xx, yy = np.meshgrid(ax, ax)
                    kernel = np.exp(-(xx**2 + yy**2) / (2. * sigma**2))
                    return kernel / np.sum(kernel)
                
                def sobel_filters(self, img):
                    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.float32)
                    Ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.float32)

                    Ix = convolve(img, Kx)
                    Iy = convolve(img, Ky)

                    G = np.hypot(Ix, Iy)
                    G = G / G.max() * 255
                    theta = np.arctan2(Iy, Ix)
                    return (G, theta)
                

                def non_max_suppression(self, img, D):
                    M, N = img.shape
                    Z = np.zeros((M,N), dtype=np.int32)
                    angle = D * 180. / np.pi
                    angle[angle < 0] += 180


                    for i in range(1,M-1):
                        for j in range(1,N-1):
                            try:
                                q = 255
                                r = 255

                            #angle 0
                                if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                                    q = img[i, j+1]
                                    r = img[i, j-1]
                                #angle 45
                                elif (22.5 <= angle[i,j] < 67.5):
                                    q = img[i+1, j-1]
                                    r = img[i-1, j+1]
                                #angle 90
                                elif (67.5 <= angle[i,j] < 112.5):
                                    q = img[i+1, j]
                                    r = img[i-1, j]
                                #angle 135
                                elif (112.5 <= angle[i,j] < 157.5):
                                    q = img[i-1, j-1]
                                    r = img[i+1, j+1]

                                if (img[i,j] >= q) and (img[i,j] >= r):
                                    Z[i,j] = img[i,j]
                                else:
                                    Z[i,j] = 0


                            except IndexError as e:
                                pass

                    return Z

                def threshold(self, img):

                    highThreshold = img.max() * self.highThreshold;
                    lowThreshold = highThreshold * self.lowThreshold;

                    M, N = img.shape
                    res = np.zeros((M,N), dtype=np.int32)

                    weak = np.int32(self.weak_pixel)
                    strong = np.int32(self.strong_pixel)

                    strong_i, strong_j = np.where(img >= highThreshold)
                    zeros_i, zeros_j = np.where(img < lowThreshold)

                    weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))

                    res[strong_i, strong_j] = strong
                    res[weak_i, weak_j] = weak

                    return (res)

                def hysteresis(self, img):

                    M, N = img.shape
                    weak = self.weak_pixel
                    strong = self.strong_pixel

                    for i in range(1, M-1):
                        for j in range(1, N-1):
                            if (img[i,j] == weak):
                                try:
                                    if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
                                        or (img[i, j-1] == strong) or (img[i, j+1] == strong)
                                        or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
                                        img[i, j] = strong
                                    else:
                                        img[i, j] = 0
                                except IndexError as e:
                                    pass

                    return img
                
                def detect(self):
                    gaussian_k = self.gaussian_kernel(self.kernel_size, self.sigma)
                    self.img_smoothed = convolve(self.img, gaussian_k)

                    self.gradientMat, self.thetaMat = self.sobel_filters(self.img_smoothed)

                    self.nonMaxImg = self.non_max_suppression(self.gradientMat, self.thetaMat)

                    self.thresholdImg = self.threshold(self.nonMaxImg)

                    img_final = self.hysteresis(self.thresholdImg)


                    return img_final.astype(np.uint8)
                
            gray = rgb2gray(img_color_copy)
                
            detector = cannyEdgeDetector(gray, sigma=1.4, kernel_size=5, lowthreshold=0.09, highthreshold=0.17, weak_pixel=100)
            img_final = detector.detect()
            
            # Morphological operations
            kernel = np.ones((3,3), np.uint8)
            closing = cv2.morphologyEx(img_final, cv2.MORPH_CLOSE, kernel, iterations=2)
            sure_bg = cv2.dilate(closing, kernel, iterations=3)
            dist_transform = cv2.distanceTransform(closing, cv2.DIST_L2, 5)
            ret, sure_fg = cv2.threshold(dist_transform, 0.1 * dist_transform.max(), 255, 0)
            sure_fg = np.uint8(sure_fg)
            unknown = cv2.subtract(sure_bg, sure_fg)
            
            # Tìm markers
            ret, markers = cv2.connectedComponents(sure_fg)
            markers = markers + 1
            markers[unknown == 255] = 0
            
            # Áp dụng Watershed
            markers = cv2.watershed(img_color_copy, markers)
            img_color_copy[markers == -1] = [0, 0, 255]
            
            # Phân vùng ảnh bằng màu ngẫu nhiên
            segmented_image = np.zeros_like(img_color_copy)
            for marker in np.unique(markers):
                if marker == -1 or marker == 1:
                    continue
                segmented_image[markers == marker] = np.random.randint(0, 255, 3)
            
            segmented_img = segmented_image
            
             # Hiển thị ảnh gốc
            display_original = convert_cv_to_image(img)
            display_original_tk = ImageTk.PhotoImage(image=display_original)
            canvas_original.configure(scrollregion=canvas_original.bbox("all"))
            canvas_original.create_image(0, 0, anchor="nw", image=display_original_tk)
            canvas_original.image = display_original_tk  # Giữ tham chiếu
            
            # Hiển thị ảnh đã phân vùng
            display_segmented = convert_cv_to_image(segmented_img)
            display_segmented_tk = ImageTk.PhotoImage(image=display_segmented)
            canvas_segmented.configure(scrollregion=canvas_segmented.bbox("all"))
            canvas_segmented.create_image(0, 0, anchor="nw", image=display_segmented_tk)
            canvas_segmented.image = display_segmented_tk  # Giữ tham chiếu

    except Exception as e:
        messagebox.showerror("Lỗi", f"Đã xảy ra lỗi: {e}")

def convert_cv_to_image(cv_img):
    """Chuyển đổi ảnh từ OpenCV sang PIL Image."""
    cv_img_rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_img_rgb)
    return pil_img

def toggle_fullscreen(event=None):
    """Chuyển đổi chế độ toàn màn hình."""
    global fullscreen
    fullscreen = not fullscreen
    root.attributes("-fullscreen", fullscreen)

def exit_fullscreen(event=None):
    """Thoát chế độ toàn màn hình."""
    global fullscreen
    fullscreen = False
    root.attributes("-fullscreen", False)

# Tạo cửa sổ chính
root = tk.Tk()
root.title("Ứng Dụng Phân Vùng Ảnh với Tkinter")
root.attributes("-fullscreen", True)  # Bật chế độ toàn màn hình
root.bind("<F11>", toggle_fullscreen)    # F11 để chuyển đổi toàn màn hình
root.bind("<Escape>", exit_fullscreen)    # Esc để thoát toàn màn hình
root.resizable(True, True)

fullscreen = True

# Nút tải ảnh
btn_load = tk.Button(root, text="Tải Ảnh", command=load_image, font=("Arial", 14))
btn_load.pack(pady=10)

# Khung hiển thị ảnh gốc và ảnh đã phân vùng với thanh cuộn
frame = tk.Frame(root)
frame.pack(fill="both", expand=True)

# Hàm tạo Canvas với Scrollbars
def create_scrollable_canvas(parent):
    canvas = Canvas(parent, bg="grey")
    h_scroll = Scrollbar(parent, orient="horizontal", command=canvas.xview)
    v_scroll = Scrollbar(parent, orient="vertical", command=canvas.yview)
    canvas.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)
    
    h_scroll.pack(side="bottom", fill="x")
    v_scroll.pack(side="right", fill="y")
    canvas.pack(side="left", fill="both", expand=True)
    return canvas

# Tạo Canvas cho ảnh gốc
canvas_original = create_scrollable_canvas(frame)
canvas_original_label = tk.Label(frame, text="Ảnh gốc sẽ hiển thị ở đây", font=("Arial", 12))
canvas_original_label.place(relx=0.5, rely=0.5, anchor="center")

# Tạo Canvas cho ảnh đã phân vùng
canvas_segmented = create_scrollable_canvas(frame)
canvas_segmented_label = tk.Label(frame, text="Ảnh phân vùng sẽ hiển thị ở đây", font=("Arial", 12))
canvas_segmented_label.place(relx=0.5, rely=0.5, anchor="center")

# Điều chỉnh layout khi thay đổi kích thước cửa sổ
def on_resize(event):
    # Hiển thị nhãn khi không có ảnh
    if not hasattr(canvas_original, 'image'):
        canvas_original_label.place(relx=0.5, rely=0.5, anchor="center")
    else:
        canvas_original_label.place_forget()
    
    if not hasattr(canvas_segmented, 'image'):
        canvas_segmented_label.place(relx=0.5, rely=0.5, anchor="center")
    else:
        canvas_segmented_label.place_forget()

root.bind("<Configure>", on_resize)

# Chạy ứng dụng
root.mainloop()