"""
Introduction:
Các bạn vào trang web sau: https://quickdraw.withgoogle.com/
Các bạn hãy chơi thử game trong vòng 1-2 phút
Đây là game QuickDraw, được phát triển dựa trên bộ dataset nổi tiếng cùng tên của Google. Các bạn có thể vào link sau
https://quickdraw.withgoogle.com/data
để có cái nhìn tổng quát về bộ dataset này. Các bạn có thể click vào 1 icon bất kì để xem các bản vẽ của category tương ứng

Pre-requisites:
Các bạn hãy vào link sau đây
https://console.cloud.google.com/storage/browser/quickdraw_dataset/full/numpy_bitmap?pageState=(%22StorageObjectListTable%22:(%22f%22:%22%255B%255D%22))&prefix=&forceOnObjectsSortingFiltering=false
và tự chọn cho mình 10 categories mà các bạn thích (hint: Đừng chọn category nào quá phức tạp)
Sau đó các bạn hãy download các file tương ứng của các categories này (click vào mũi tên đi xuống ở dòng tương ứng)
Các file được download sẽ có tên như sau: full_numpy_bitmap_{category}.npy. Sau đó các bạn hãy để các file .npy này
vào chung với folder của script python này. Giờ các bạn có thể chuyển sang phần tiếp theo - phần lập trình
"""
# %%
import numpy as np
from matplotlib import pyplot as plt
import os 

DATA_DIR = "./data"
# avg_images = dict()
avg_images = []
class_names = []
fig, axes = plt.subplots(5, 2)
axes = axes.ravel()

# Class đã chọn: 
# Angel
# Apple
# Axe 
# Banana
# Baseball bat
# Basketball
# Book
# Boomerang
# Butterfly
# Cat

for index, file in enumerate(os.listdir("./data")):
    file_path = os.path.join("./data", file)
    class_name = file_path.split('.')[1].split('_')[-1]
    images = np.load(file_path).astype(np.float32)
    avg_image = np.average(images, axis=0)
    # avg_images[class_name] = avg_image
    avg_images.append(avg_image)
    class_names.append(class_name)
    # print(class_name)
    # plt.subplot(5,2,index+1)
    axes[index].title.set_text(class_name)
    axes[index].imshow(np.reshape(avg_image, (28, 28)))
plt.show()

file_path = os.path.join("./data", "full_numpy_bitmap_angel.npy")
images = np.load(file_path).astype(np.float32)
test_images = images[-10:]
# %%
plt.imshow(np.reshape(test_images[0], (28, 28)))
plt.show()

# print(np.array(avg_images).shape)
scores = [np.dot(test_images[0], np.array(image)) for image in avg_images]
max_index = np.argmax(scores)
print(f"Class with maximum score is {class_names[max_index]} with score {scores[max_index]}")
# max_score = 0
# max_class = ""
# for key, value in avg_images.items():
#     score = np.dot(test_images[0], value)
#     if max_score < score:
#         max_score = score
#         max_class = key
#     print(f"Class: {key}, score: {score}")
# print(f"Max score: {max_score}, class: {max_class}")
# class_name = file_path.split('.')[1].split('_')[-1]
# Bước 1: Các bạn hãy chọn 1 trong số các file .npy mà các bạn vừa tải về ở trên, và thay đổi đường dẫn tương ứng phía dưới
# file_path = "./full_numpy_bitmap_cat.npy"  # <= Các bạn thay từ bicycle bằng tên tương ứng của category các bạn chọn nhé
# images = np.load(file_path).astype(np.float32)  # Load toàn bộ các ảnh của category này vào biến images
# print(images.shape)
# train_images = images[:-10]  # Lấy tất cả ảnh, ngoại trừ 10 ảnh cuối ra làm bộ training.
# test_images = images[-10:]  # Giữ 10 ảnh cuối làm bộ test

# #TODO Bước 2: Các bạn hãy tính ra 1 bức ảnh trung bình của bộ training. Bức ảnh này sẽ có kích thước 28x28 pixel (hint: 784=28*28)
# print(train_images.shape)
# avg_image = np.average(train_images, axis=0)
# # print(avg_image.shape)
# # Bước 3: Các bạn sẽ visualize bức ảnh trung bình các bạn vửa tính được ở bước 2 bằng 2 dòng sau. Các bạn thử
# # xem các bạn có nhận ra được category mà các bạn chọn bằng cách nhìn vào bức ảnh trung bình này không nhé
# # %%
# plt.imshow(np.reshape(avg_image, (28, 28)))
# plt.show()

# # Bước 4: Các bạn chọn 1 index bất kì từ 0 đến 9. Ví dụ mình chọn index = 4
# # Sau đó các bạn hãy tính tích vô hướng (dot product) của bức ảnh test này với bức ảnh trung bình các bạn tính được ở trên
# index = 4  # Các bạn có thể thay đổi index tùy ý
# test_image = test_images[index]
# #TODO Các bạn tính tích vô hướng (dot product) của bức ảnh test và ảnh trung bình ở dòng dưới đây
# # (các bạn có thể code trên nhiều hơn 1 dòng)
# score = np.dot(test_image, avg_image)
# print(score)

# Bước 5: Các bạn hãy lặp lại bước 1 đến 3 cho tất cả các categories còn lại (chú ý tại bước 1 các bạn không cần phân
# ra train với test images nữa nhé, coi như là dùng tất cả cho train). Sau đó các bạn hãy tính tích vô hướng của từng ảnh
# trung bình của ảnh test các bạn chọn ở bước 4 với từng bức ảnh trung bình này.
#
# Cuối cùng các bạn xem là liệu trong 10 score này, score tương ứng với tích vô hướng của ảnh test này với
# ảnh trung bình của category của chính nó có phải là score lớn nhất không nhé. Các bức ảnh trung bình mà các bạn tính ra
# có thể xem như là weight cho từng category mà các bạn vừa học ở bài 1 (tất nhiên là weight của mô hình sau khi đã
# train xong)

# Bước 6 (optional): Các bạn thử visualize 10 weight (avg_image) này trong cùng 1 ảnh kích thước 2x5 hoặc 5x2 để so sánh xem,
# weight của các categories nào dễ nhìn và weight nào không nhé


# %%
