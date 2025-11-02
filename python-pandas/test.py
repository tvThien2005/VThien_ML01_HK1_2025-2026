import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
df = pd.read_csv("sinhvien.txt",sep =",",header=None,names=["MSV","Họ và Tên","Khoa","Chuyên ngành","GPA"])
# In ra MSV và Chuyên ngành
# print(df[["MSV", "Chuyên ngành"]])
# # In ra 3 dòng đầu tiên
# print(df.head(5))
# # In ra các sinh viên có GPA >= 3.6
# hoc_sinh_sx = df[df.GPA >= 3.6]
# hoc_sinh_sx.to_csv("hocSinhGioi.csv")
# thêm cột LoaiBang nhận giá trị Xuất sắc, Giỏi, Khá
#  tương ứng với GPA >= 3.6, GPA >= 3.2, GPA < 3.2
df["LoaiBang"] = np.where(df.GPA >= 3.6, "Xuất sắc",np.where(df.GPA >= 3.2, "Giỏi","khá"))
print(df)
# In ra thống kê mô tả của cột GPA
print(df.describe())
# df["GPA"].plot()
# plt.show()
loai_bang_counts = df["LoaiBang"].value_counts()
plt.figure(figsize=(6,6))
plt.pie(loai_bang_counts, labels=loai_bang_counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Tỉ lệ các loại bằng theo GPA")
plt.axis('equal')  # Đảm bảo hình tròn
plt.show()