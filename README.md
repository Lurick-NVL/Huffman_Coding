# Huffman_Coding Ngô Lực // Khoa CNTT // Mạng Máy Tính & Truyền Thông
📋 Yêu cầu hệ thống
Python 3.x đã được cài đặt

Thư viện tkinter(thường đã có sẵn khi cài đặt Python)

🚀 Cách chạy chương trình

Bước 1: Sao chép hoặc tải kho lưu trữ về máy

git clone https://github.com/Lurick-NVL/Huffman_Coding.git
cd Huffman_Coding

Bước 2: Chạy chương trình

python main.py

💡 Hướng dẫn sử dụng giao diện
Chương trình có đồ họa giao diện (GUI) với 2 tab chính:

Tab NÉN (Nén)

Chọn nguồn tệp : Bấm vào nút "Chọn tệp" ở dòng "Tệp văn bản nguồn (. txt)" để chọn tệp .txtcần nén

Choose file target : Tên file target ( .huff) sẽ tự động được gợi ý hoặc bạn có thể tự động chọn

Chọn tân kiến ​​nén :

Nén Huffman tĩnh : Chỉ dùng thuật toán Huffmancoding

Nén Huffman + LZ77 : Kết hợp Huffman và LZ77 để nén tốt hơn (khuyến nghị)

Xem kết quả : Sau khi nén, thông tin về gốc kích thước tệp, nén tệp và nén tỷ lệ sẽ hiển thị

Tab GIẢI NÉN (Decompress)

Chọn nén file : Click "Chọn file" ở dòng "Nguồn nén file (.huff)" để chọn file .huffcần giải nén

Choose file target : Giải nén file tên ( .txt) sẽ tự động được gợi ý

Nhấp vào "Giải nén (Giải nén)" : Chương trình sẽ tự động nhận dạng định dạng nhận diện (HF2/HFZ/legacy) và giải nén

Xem kết quả : Thông tin về nén file kích thước và giải nén file sau sẽ hiển thị

📁 Tệp cấu trúc

Huffman_Coding/
├── main.py                    # File chính chạy GUI
├── huffman_compress.py        # Module xử lý nén
├── huffman_decompress. py      # Module xử lý giải nén
└── README.md

⚙️ Các tính năng

✅ Nén file văn bản với 2 phương pháp (Huffman tĩnh và Huffman + LZ77)

✅ Giải nén file.huff

✅ Hiển thị quá trình nén/giải nén theo thời gian thực

✅ Thống kê tỷ lệ nén, kích thước tệp

✅ Giao diện thân thiện, dễ sử dụng

🔍 Lưu ý

Nén nguồn tệp phải là bản văn tệp ( .txt)

Tên tệp đích và nguồn tệp không được trùng lặp

Chương trình sẽ xác định khi tệp ghi đè tồn tại
