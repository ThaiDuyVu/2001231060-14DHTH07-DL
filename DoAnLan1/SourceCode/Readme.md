# AI Model Training & Demo Project (Streamlit + PyTorch)

## Giới thiệu
Đây là một dự án AI end-to-end bao gồm:
- Huấn luyện mô hình (training)
- Đánh giá mô hình (evaluation)
- Triển khai demo giao diện web (Streamlit)
- Hệ thống người dùng và bình luận dạng mạng xã hội đơn giản

Mô hình được huấn luyện bằng PyTorch và triển khai trực tiếp trong ứng dụng Streamlit.

---

## Cấu trúc thư mục dự án
project/
│
├── train.py # Code huấn luyện mô hình + lưu model
├── demo.py # Giao diện Streamlit demo + tích hợp model
├── evaluate_model.py # Kiểm tra độ chính xác mô hình bằng dữ liệu test
│
├── best_model.pt # Model đã train tốt nhất (PyTorch)
│
├── test_data.json # Dữ liệu test mẫu
├── label_map.json # Ánh xạ nhãn (label encoding/decoding)
│
├── users.json # Lưu thông tin người dùng đăng nhập
├── comments.json # Lưu bình luận người dùng (mini social system)
│
└── README.md # Tài liệu hướng dẫn dự án

---

## Mô tả từng thành phần

### 1. `train.py`
- Chứa toàn bộ pipeline huấn luyện mô hình
- Bao gồm:
  - Load dữ liệu
  - Tiền xử lý dữ liệu
  - Xây dựng mô hình
  - Train & validate
  - Lưu model tốt nhất ra `best_model.pt`

 Mục đích: tạo ra model AI phục vụ cho demo

---

### 2. `demo.py`
- Xây dựng giao diện web bằng **Streamlit**
- Chức năng chính:
  - Đăng nhập / quản lý user (`users.json`)
  - Dự đoán bằng model `best_model.pt`
  - Hiển thị kết quả AI
  - Hệ thống bình luận giống mạng xã hội (`comments.json`)
  
Mục đích: demo ứng dụng AI chạy thực tế

---

### 3. `evaluate_model.py`
- Dùng để đánh giá độ chính xác mô hình
- Hoạt động:
  - Load model từ `best_model.pt`
  - Load dữ liệu từ `test_data.json`
  - So sánh dự đoán với label thật
  - Tính accuracy / report kết quả

 Mục đích: kiểm tra chất lượng model sau training

---

### 4. `test_data.json`
- Bộ dữ liệu test được tạo sẵn
- Dùng trong evaluation
- Format thường gồm:

5. label_map.json
Ánh xạ giữa label và tên lớp
6. users.json
Lưu thông tin người dùng đăng ký/đăng nhập
7. best_model.pt
Model đã huấn luyện xong (PyTorch)
8. comments.json
Lưu bình luận người dùng trong giao diện demo