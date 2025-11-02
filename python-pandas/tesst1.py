import pandas as pd
from pandas import set_option
import matplotlib.pyplot as plt
import numpy as np  
import seaborn as sns
from pandas.plotting import scatter_matrix
df = pd.read_csv("train.csv",sep =",")
# print(df.tail(5))
# print(df.describe())

# sự tương quan giữa các thuộc tính (Correlations Between Attributes)
# 1 → tương quan dương tuyệt đối (cùng tăng/giảm).

#-1 → tương quan âm tuyệt đối (một tăng, một giảm).

# 0 → gần như không có mối liên hệ tuyến tính.
def correlation_between_attributes():

    numeric_df = df.select_dtypes(include=['number'])
    correlation = numeric_df.corr(method='pearson')
    set_option('display.width', 100)
    set_option('display.precision', 3)

    print(correlation)

# độ lệch của các thuộc tính (Skewness of Attributes)
# Skewness = 0 → phân phối gần đối xứng (giống phân phối chuẩn).
# Skewness > 0 → phân phối lệch phải (right-skewed, có đuôi dài bên phải).

# Skewness < 0 → phân phối lệch trái (left-skewed, có đuôi dài bên trái).
def skewwness_of_attributes():
    numeric_df = df.select_dtypes(include=['number'])
    skewness = numeric_df.skew()
    print(skewness)

# biểu đồ(Histograms)

# biểu đồ tròn(Pie Charts)
def circile():
    loai_bang_Survived = df["Survived"].value_counts()
    labels = ["chết" if x == 0 else "sống" for x in loai_bang_Survived.index]
    plt.figure(figsize=(6,6))
    plt.pie(loai_bang_Survived, labels=labels, autopct='%1.1f%%', startangle=90)
    plt.title("Tỉ lệ sống sót")
    plt.axis('equal')  # Đảm bảo hình tròn
    plt.show()
# Univariate Plots biểu dồ đơn biến
def univariate_plots():
    df.hist(figsize=(12,10))
    plt.tight_layout()
    plt.show()


# biểu đồ mật độ (Density Plots)
def density_plots():
    df.plot(kind='density', subplots=True, layout=(3,3), sharex=False, figsize=(12,10))
    plt.tight_layout()
    plt.show()

# biểu đồ hộp (Box and Whisker Plots)
def box_and_whisker_plots():
    df.plot(kind='box', subplots=True, layout=(3,3), sharex=False, sharey=False, figsize=(12,10))
    plt.tight_layout() # Là hàm của matplotlib. Tự chỉnh lề và khoảng cách giữa các subplot để tránh chồng lấn giữa tiêu đề, trục, nhãn.
    plt.show()


# biểu đồ đa biến (Multivariate Plots)
def Multivariate_Plots():
    numeric_df = df.select_dtypes(include=['number'])
     # Tính ma trận tương quan
    correlations = numeric_df.corr()
    fig = plt.figure(figsize=(10,8))
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, cmap='coolwarm', vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,len(correlations.columns),1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)    
    ax.set_xticklabels(correlations.columns, rotation=90)
    ax.set_yticklabels(correlations.columns)
    plt.tight_layout()
    plt.show()

# ma trận biểu đồ phân tán (Scatterplot Matrix)
def Scatterplot_Matrix():
    numeric_df = df.select_dtypes(include=['number'])  # chỉ lấy cột số
    plt.figure(figsize=(12,10))
    scatter_matrix(numeric_df)
    plt.tight_layout()
    plt.show()
 
# Scatterplot_Matrix()
# Multivariate_Plots()
# box_and_whisker_plots()
# density_plots()
# univariate_plots()
# circile()
# correlation_between_attributes()
#skewwness_of_attributes()

