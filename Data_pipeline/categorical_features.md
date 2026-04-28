# Phân loại feature categorical (heart.csv)

Dựa trên dữ liệu trong `heart.csv` (các cột: `age, sex, chest_pain, restbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, target`), các **feature categorical** là:

| Feature | Nhóm categorical | Phân loại chi tiết |
|---|---|---|
| `sex` | Categorical | **Boolean** (2 giá trị: 0/1) |
| `fbs` | Categorical | **Boolean** (fasting blood sugar > 120: 0/1) |
| `exang` | Categorical | **Boolean** (exercise induced angina: 0/1) |
| `target` | Categorical (label) | **Boolean** (0/1) |
| `chest_pain` | Categorical | **Nominal** (4 nhóm loại đau ngực, không có thứ tự tự nhiên) |
| `restecg` | Categorical | **Nominal** (3 nhóm kết quả ECG, không có thứ tự tự nhiên) |
| `thal` | Categorical | **Nominal** (normal/fixed defect/reversible defect/unknown) |
| `slope` | Categorical | **Ordinal** (các mức slope có tính thứ tự) |
| `ca` | Categorical rời rạc | **Ordinal** (số mạch máu lớn: 0–4, có thứ tự tăng dần) |

## Các feature không phải categorical

`age`, `restbps`, `chol`, `thalach`, `oldpeak` được xem là **numeric** (liên tục hoặc rời rạc có bản chất định lượng).
