# ❤️ Heart Disease Classification and Prediction

Dự án thuộc môn học: **Nhập môn Trí tuệ Nhân tạo - IT3160**

## 👥 Nhóm thực hiện

| Họ và tên         | MSSV     |
| ----------------- | -------- |
| Lê Hải Anh        | 20236019 |
| Lương Hương Giang | 20236027 |
| Dương Gia Huy     | 20236035 |
| Nguyễn Nhật Linh  | 20236040 |
| Nguyễn Quang Linh | 20236041 |

---

# 📌 1. Giới thiệu dự án

Bệnh tim mạch là một trong những nguyên nhân gây tử vong hàng đầu trên thế giới. Việc phát hiện sớm nguy cơ mắc bệnh tim giúp bác sĩ đưa ra các biện pháp can thiệp kịp thời, nâng cao hiệu quả điều trị và giảm tỷ lệ tử vong.

Dự án này xây dựng một hệ thống ứng dụng trí tuệ nhân tạo nhằm dự đoán nguy cơ mắc bệnh tim dựa trên các chỉ số sức khỏe của bệnh nhân. Hệ thống sử dụng các thuật toán Machine Learning để hỗ trợ đánh giá nguy cơ, từ đó cung cấp thông tin tham khảo cho người dùng và nhân viên y tế.

> ⚠️ Kết quả dự đoán chỉ mang tính chất hỗ trợ tham khảo và không thay thế cho chẩn đoán của bác sĩ chuyên khoa.

---

# 🎯 2. Mục tiêu

* Xây dựng mô hình dự đoán nguy cơ mắc bệnh tim từ dữ liệu sức khỏe.
* So sánh hiệu quả của nhiều thuật toán Machine Learning.
* Lựa chọn mô hình có hiệu suất tốt nhất để triển khai.
* Xây dựng ứng dụng Web hỗ trợ người dùng nhập dữ liệu và nhận kết quả dự đoán trực quan.

---

# 📊 3. Bộ dữ liệu

## Nguồn dữ liệu

Heart Disease Dataset từ Kaggle.

* Số lượng mẫu: khoảng 303 bệnh nhân.
* Số lượng thuộc tính: 14 thuộc tính.

## Biến đầu vào (Features)

### Thông tin cá nhân

| Thuộc tính | Mô tả     |
| ---------- | --------- |
| age        | Tuổi      |
| sex        | Giới tính |

### Triệu chứng lâm sàng

| Thuộc tính | Mô tả                      |
| ---------- | -------------------------- |
| cp         | Loại đau ngực              |
| exang      | Đau thắt ngực khi vận động |

### Chỉ số sinh lý và xét nghiệm

| Thuộc tính | Mô tả               |
| ---------- | ------------------- |
| trestbps   | Huyết áp khi nghỉ   |
| chol       | Nồng độ Cholesterol |
| fbs        | Đường huyết lúc đói |

### Hoạt động tim

| Thuộc tính | Mô tả                    |
| ---------- | ------------------------ |
| thalach    | Nhịp tim tối đa đạt được |

### Điện tim và kiểm tra chuyên sâu

| Thuộc tính | Mô tả                        |
| ---------- | ---------------------------- |
| restecg    | Kết quả điện tâm đồ khi nghỉ |
| oldpeak    | ST Depression                |
| slope      | Độ dốc của đoạn ST           |

### Tình trạng mạch máu và máu

| Thuộc tính | Mô tả                          |
| ---------- | ------------------------------ |
| ca         | Số lượng mạch máu lớn bị hẹp   |
| thal       | Kết quả xét nghiệm Thalassemia |

---

## Biến đầu ra (Target)

| Giá trị | Ý nghĩa      |
| ------- | ------------ |
| 0       | Nguy cơ thấp |
| 1       | Nguy cơ cao  |

---

# ⚙️ 4. Quy trình thực hiện

## 1. Tiền xử lý dữ liệu

Các bước xử lý dữ liệu bao gồm:

* Kiểm tra dữ liệu thiếu.
* Xử lý giá trị khuyết trong các thuộc tính:

  * `ca`
  * `thal`
* Mã hóa dữ liệu phân loại.
* Chuẩn hóa dữ liệu bằng:

  * `StandardScaler`

---

## 2. Phân tích dữ liệu (EDA)

Nhóm tiến hành:

* Phân tích phân phối độ tuổi.
* Phân tích phân phối cholesterol.
* Khảo sát sự mất cân bằng dữ liệu.
* Xây dựng ma trận tương quan.
* Đánh giá mức độ ảnh hưởng của từng thuộc tính.

Các thuộc tính có ảnh hưởng đáng kể:

* `cp`
* `thalach`
* `oldpeak`
* `ca`
* `thal`

---

## 3. Huấn luyện mô hình

Dữ liệu được chia:

* 80% Train Set
* 20% Test Set

Đồng thời sử dụng:

* K-Fold Cross Validation

Các mô hình được thử nghiệm:

### Logistic Regression

* Mô hình phân loại tuyến tính.
* Dễ triển khai.
* Làm mô hình baseline.

### Decision Tree

* Dễ giải thích.
* Trực quan hóa tốt.
* Có nguy cơ overfitting.

### Random Forest

* Mô hình tổ hợp từ nhiều cây quyết định.
* Độ chính xác cao.
* Giảm overfitting.

### K-Nearest Neighbors (KNN)

* Dựa trên khoảng cách giữa các điểm dữ liệu.
* Hiệu quả với dữ liệu nhỏ.

### Support Vector Machine (SVM)

* Tìm siêu phẳng phân tách tối ưu.
* Hoạt động tốt trên tập dữ liệu kích thước vừa và nhỏ.

---

## 4. Đánh giá mô hình

Các chỉ số đánh giá:

* Accuracy
* Precision
* Recall
* F1-Score

Trong bài toán y tế, nhóm ưu tiên:

> Recall cao nhằm giảm khả năng bỏ sót các bệnh nhân có nguy cơ mắc bệnh tim.

Kết quả đạt độ chính xác khoảng:

**80% - 90%**

---

# 🌲 5. Mô hình được lựa chọn cuối cùng

Sau quá trình thử nghiệm và đánh giá, nhóm lựa chọn **Random Forest** là mô hình chính thức của hệ thống.

## Lý do lựa chọn

### Độ chính xác cao

Random Forest đạt độ chính xác cao nhất hoặc nằm trong nhóm cao nhất so với các mô hình còn lại.

### Giảm Overfitting

So với Decision Tree đơn lẻ, Random Forest ổn định hơn nhờ sử dụng nhiều cây quyết định.

### Recall tốt

Khả năng phát hiện các trường hợp có nguy cơ mắc bệnh tim tốt hơn, phù hợp với yêu cầu của bài toán y tế.

### Khả năng tổng quát hóa cao

Mô hình hoạt động ổn định trên dữ liệu kiểm thử và dữ liệu mới.

### Hỗ trợ đánh giá mức độ quan trọng của thuộc tính

Random Forest cho phép xác định các yếu tố ảnh hưởng lớn nhất tới dự đoán như:

* cp
* thalach
* oldpeak
* ca
* thal

---

## Nguyên lý hoạt động

Random Forest là một phương pháp Ensemble Learning sử dụng nhiều cây quyết định.

Quy trình:

1. Tạo nhiều tập dữ liệu ngẫu nhiên bằng Bootstrap Sampling.
2. Huấn luyện một cây quyết định trên mỗi tập dữ liệu.
3. Mỗi lần phân tách chỉ sử dụng một tập con thuộc tính ngẫu nhiên.
4. Các cây đưa ra dự đoán độc lập.
5. Kết quả cuối cùng được xác định bằng Majority Voting.

Nhờ đó mô hình:

* Giảm phương sai.
* Tăng độ ổn định.
* Tăng khả năng dự đoán chính xác.

---

# 🏗️ 6. Kiến trúc hệ thống

```text
User Input
     │
     ▼
Flask Web Application
     │
     ▼
Data Preprocessing
     │
     ▼
Random Forest Model
     │
     ▼
Prediction Result
```

Người dùng nhập các thông tin sức khỏe thông qua giao diện Web, dữ liệu được tiền xử lý và đưa vào mô hình Random Forest để dự đoán nguy cơ mắc bệnh tim.

---

# 💻 7. Cách chạy dự án

## Bước 1: Clone repository

```bash
git clone https://github.com/haianh020505/heart-disease-classification.git
cd heart-disease-classification
```

## Bước 2: Cài đặt thư viện

```bash
pip install -r requirements_web.txt
```

## Bước 3: Chạy ứng dụng Flask

```bash
python app.py
```

## Bước 4: Truy cập ứng dụng

Mở trình duyệt và truy cập:

```text
http://127.0.0.1:5000
```



