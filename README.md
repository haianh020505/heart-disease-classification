# BÁO CÁO DỰ ÁN: PHÂN LOẠI VÀ DỰ ĐOÁN NGUY CƠ MẮC BỆNH TIM

Dự án thuộc môn học: **Nhập môn Trí tuệ Nhân tạo - IT3160**

### 👥 Nhóm thực hiện
1. **Lê Hải Anh** - 20236019
2. **Lương Hương Giang** - 20236027
3. **Dương Gia Huy** - 20236035
4. **Nguyễn Nhật Linh** - 20236040
5. **Nguyễn Quang Linh** - 20236041

---

## 📌 1. Bối cảnh & Mục tiêu
* **Bối cảnh**: Bệnh tim mạch là nguyên nhân tử vong hàng đầu toàn cầu. Phát hiện sớm giúp can thiệp kịp thời, giảm tỷ lệ tử vong và hỗ trợ bác sĩ chẩn đoán lâm sàng.
* **Bài toán**: Phân loại nhị phân (Binary Classification) để dự đoán nguy cơ mắc bệnh tim (Có / Không).
* **Mục tiêu**: Tự động hóa dự đoán nguy cơ dựa trên 13 chỉ số sức khỏe, hỗ trợ ra quyết định lâm sàng (không thay thế bác sĩ chuyên khoa).

---

## 📊 2. Dữ liệu & Các thuộc tính (Input/Output)
* **Dataset**: ~303 mẫu và 14 thuộc tính chính (Nguồn: Kaggle Heart Disease Dataset).
* **Input (13 chỉ số chia làm 6 nhóm)**:
  1. *Thông tin cá nhân*: Tuổi (`age`), Giới tính (`sex`).
  2. *Triệu chứng lâm sàng*: Loại đau ngực (`cp`), Đau ngực khi vận động (`exang`).
  3. *Chỉ số sinh lý & xét nghiệm*: Huyết áp nghỉ (`trestbps`), Cholesterol (`chol`), Đường huyết đói (`fbs`).
  4. *Hoạt động tim*: Nhịp tim tối đa (`thalach`).
  5. *Điện tim & kiểm tra chuyên sâu*: Kết quả ECG nghỉ (`restecg`), ST Depression (`oldpeak`), Slope ST (`slope`).
  6. *Tình trạng mạch máu & máu*: Số mạch máu lớn bị hẹp (`ca`), Thalassemia (`thal`).
* **Output**:
  * `0`: Nguy cơ thấp (Không mắc bệnh tim)
  * `1`: Nguy cơ cao (Có nguy cơ mắc bệnh tim)

---

## ⚙️ 3. Quy trình thực hiện (Pipeline)
1. **Tiền xử lý dữ liệu**: Xử lý dữ liệu khuyết thiếu (`ca`, `thal`), mã hóa thuộc tính phân loại (categorical encoding) và chuẩn hóa dữ liệu (`StandardScaler`).
2. **Phân tích dữ liệu (EDA)**: Đánh giá phân phối độ tuổi, nồng độ cholesterol, ma trận tương quan và xác định mức độ quan trọng của các thuộc tính (như `cp`, `thalach`, `oldpeak`).
3. **Huấn luyện mô hình**: Chia tập dữ liệu (80% Train / 20% Test) kết hợp K-Fold Cross-Validation. Sử dụng các mô hình:
   * **Logistic Regression** (Phân loại xác suất, baseline tốt)
   * **Decision Tree** (Cây quyết định dễ giải thích trực quan)
   * **Random Forest** (Tránh overfitting, cho độ chính xác cao nhất)
   * **KNN** (Dựa trên khoảng cách lân cận)
   * **SVM** (Tối ưu hóa siêu phẳng phân tách dữ liệu nhỏ)
4. **Đánh giá & Chọn mô hình**: Đánh giá dựa trên Accuracy, Precision, Recall và F1-score. Ưu tiên mô hình có chỉ số **Recall** cao để tránh bỏ sót bệnh nhân nguy cơ cao. Độ chính xác đạt khoảng **80% - 90%**.

---

## 💻 4. Cách chạy dự án
1. **Cài đặt thư viện**:
   ```bash
   pip install -r requirements_web.txt
   ```
2. **Khởi chạy ứng dụng Web (Flask)**:
   ```bash
   python app.py
   ```
3. Truy cập vào địa chỉ [http://127.0.0.1:5000](http://127.0.0.1:5000) trên trình duyệt để sử dụng ứng dụng web.
