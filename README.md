# Báo cáo Assignment 2: Xây dựng Agent Giao dịch Chứng khoán

## 1. Đặt Vấn Đề

Mục tiêu của assignment là xây dựng một agent giao dịch tự động có khả năng học hỏi và đưa ra quyết định thông minh trên thị trường chứng khoán.

Để thành công cho bài toán với mã IBM, theo em, agent cần phải học cách trả lời một số câu hỏi:

-   **Thời điểm hành động:** Khi nào là thời điểm tối ưu để Mua, Bán, hay Giữ vị thế?
-   **Quy mô vị thế:** Nên giao dịch với khối lượng bao nhiêu? Một phần nhỏ hay toàn bộ tài sản?
-   **Quản lý rủi ro:** Khi nào nên chấp nhận rủi ro cao (sử dụng đòn bẩy, bán khống) và khi nào cần phòng thủ để bảo toàn vốn?
-   **Nhận diện xu hướng:** Làm thế nào để phân biệt giữa những biến động nhiễu ngắn hạn và một xu hướng thực sự của thị trường?

*Trong quá trình thực hiện, em nhận thấy rằng agent đang học chưa được "tốt cho lắm" do có một số hàm em chưa cải thiện được (ví dụ: transaction fee được tính 3%).*

## 2. Giải Thuật Sử Dụng

Em sử dụng thuật toán Proximal Policy Optimization (PPO). Tuy nhiên, em đã có một cải tiến quan trọng so với giải thuật PPO gốc:

-   **Giải thuật gốc:** PPO tiêu chuẩn sử dụng một mạng nơ-ron đơn giản (MLP - Multi-Layer Perceptron), khiến agent bị "mất trí nhớ" (memoryless). Nó chỉ có thể ra quyết định dựa trên thông tin của một thời điểm duy nhất.
-   **Cải tiến:** Em sử dụng **Recurrent Proximal Policy Optimization (RecurrentPPO)** từ thư viện `sb3-contrib`, kết hợp với kiến trúc `MlpLstmPolicy`. Việc tích hợp lớp LSTM (Long Short-Term Memory) đã trang bị cho agent một "bộ nhớ", cho phép nó ghi nhớ và phân tích một chuỗi các sự kiện trong quá khứ. Điều này cực kỳ quan trọng, giúp agent có khả năng nhận diện xu hướng và động lượng của thị trường, một yếu tố mà PPO gốc không thể làm được.

## 3. Xây Dựng Không Gian Hành Động (Action Space)

Em nhận thấy rằng trong trading thì việc chỉ cắm đầu mua và bán sẽ là không đủ mà ở đó sẽ còn những action có thể nhận đến. Dưới đây là dựa trên một số nghiên cứu về action em có thể thực hiện:

Không gian hành động định nghĩa tất cả các lựa chọn mà agent có thể thực hiện ở mỗi bước. Để mô phỏng gần hơn với một nhà giao dịch thực tế, agent được trang bị một bộ 16 hành động rời rạc:

-   **Hành động 0: `HOLD`** - Không thực hiện bất kỳ giao dịch nào, giữ nguyên vị thế hiện tại.
-   **Hành động 1-5: `BUY`** - Mua vào với 5 mức độ khác nhau, sử dụng từ 24% đến 53% lượng tiền mặt hiện có.
-   **Hành động 6-10: `SELL / SELL SHORT`** - Bán ra cổ phiếu đang nắm giữ hoặc thực hiện bán khống nếu không có cổ phiếu. Khối lượng bán dựa trên 5 mức độ, từ 22% đến 73% tổng giá trị tài sản.
-   **Hành động 11-15: `LEVERAGED BUY`** - Mua sử dụng đòn bẩy, với 5 mức đòn bẩy từ 1.5x đến 3.0x so với vốn chủ sở hữu.

## 4. Xây Dựng Không Gian Quan Sát (Observation Space)

Em sử dụng thư viện `yfinance` để nhập dữ liệu nguồn từ trang `finance.yahoo.com`, với dữ liệu được sử dụng cho IBM trong vòng 1 năm gần nhất từ thời điểm em bắt đầu lần đầu training (20/05/2024 - 20/05/2025).

Đây là toàn bộ dữ liệu mà agent nhận được để đưa ra quyết định. Không gian quan sát được thiết kế để cung cấp một cái nhìn toàn diện, bao gồm 10 yếu tố sau:

1.  **Tỷ lệ tiền mặt:** `cash / init_cash`
2.  **Số lượng cổ phiếu sở hữu:** `stock_owned` (số dương là mua, số âm là bán khống)
3.  **Giá đóng cửa (Scaled):** `Close`
4.  **Khối lượng giao dịch (Scaled):** `Volume`
5.  **Chỉ báo RSI (Scaled):** Đo lường trạng thái quá mua/quá bán.
6.  **Chỉ báo EMA (Scaled):** Thể hiện xu hướng giá gần đây.
7.  **Chỉ báo MACD (Scaled):** Đo lường động lượng và xu hướng.
8.  **Chỉ báo BB_percent (Scaled):** Cho biết vị trí của giá so với dải Bollinger.
9.  **Vị trí thời gian:** `step_idx / max_steps`
10. **Vốn chủ sở hữu đỉnh (Peak Equity):** `peak_equity / init_cash`

*Tất cả các dữ liệu số đều được chuẩn hóa bằng `StandardScaler` để giúp mạng nơ-ron học hiệu quả hơn.*

## 5. Xây Dựng Hàm Thưởng (Reward Function)

Hàm thưởng trong phiên bản v17 được thiết kế để dạy cho agent về kỷ luật quản lý rủi ro. Điều này em xây dựng dựa trên việc nhận thấy rằng agent cần phải vừa khám phá nhưng phải tránh rủi ro, vừa có thể reward nhiều hơn trong mỗi lần nhận tiền dương và đồng thời có thể học cách bảo toàn vốn do cơ chế hoạt động từ đề bài.

-   **Phần thưởng cốt lõi:** Dựa trên sự thay đổi giá trị vốn chủ sở hữu (equity) sau mỗi bước.
    ```
    reward = (equity - self.prev_total_asset) / self.init_cash
    ```
-   **Hình phạt sụt giảm từ đỉnh (Drawdown Penalty):** Đây là cơ chế phạt chính. Agent bị phạt nặng nếu để mất đi số tiền lời đã kiếm được. Mức phạt tỷ lệ thuận với mức độ sụt giảm so với đỉnh cao nhất mà danh mục từng đạt được.
    ```
    drawdown = (self.peak_equity - equity) / self.peak_equity
    reward -= drawdown * 0.5
    ```
    Cơ chế này buộc agent phải học cách bảo toàn vốn thay vì chỉ theo đuổi lợi nhuận.

## 6. Xây Dựng Hàm Kết Thúc (Terminated/Done Condition)

Một lượt chơi (episode) sẽ kết thúc khi một trong hai điều kiện sau xảy ra:

1.  **Hết dữ liệu:** Agent đã đi hết chuỗi thời gian trong bộ dữ liệu huấn luyện.
    ```python
    if self.step_idx >= self.max_steps:
        done = True
    ```
2.  **Phá sản:** Vốn chủ sở hữu của agent giảm xuống dưới ngưỡng tối thiểu (`total_asset_min`). Khi điều này xảy ra, agent sẽ nhận một hình phạt rất lớn.
    ```python
    if equity < self.total_asset_min:
        done = True
        reward = -1
    ```

## 7. Kết Quả Đạt Được

Quá trình huấn luyện mô hình v17 (với 1 triệu timesteps) đã cho thấy những kết quả rất đặc trưng:

![Biểu đồ thể hiện hiệu suất huấn luyện theo thời gian thực](https://github.com/wise-warlock/REL_Trading/blob/main/Figure_7.png)

-   **Cumulative Reward (ep_rew_mean):** Chỉ số này vẫn ở mức âm, cho thấy chiến lược tổng thể của agent vẫn chưa có lợi nhuận một cách nhất quán. Mặc dù có những giao dịch thắng lớn, nhưng những giao dịch thua lỗ còn lớn hơn.
-   **Total Assets (Vốn chủ sở hữu):** Biểu đồ huấn luyện cho thấy một hành vi giao dịch cực kỳ biến động. Agent đã học được cách sử dụng đòn bẩy để tạo ra những cú đột phá, đẩy giá trị danh mục lên tới $800,000. Tuy nhiên, nó hoàn toàn thiếu khả năng quản lý rủi ro, dẫn đến những cú sụt giảm thảm khốc ngay sau đó.

Sau khi để 80% tập cho training, với 20% cho tập test, kết quả chỉ nhận được **$177,027**.

**Kết luận:** Agent đã học được cách "tấn công" để tìm kiếm lợi nhuận lớn nhưng chưa học được cách "phòng thủ" để bảo vệ thành quả. Nó giống một người chơi cờ bạc hơn là một nhà giao dịch có kỷ luật. Điều này em nhận thấy là do agent chưa thể normalize hàm và khá là "nhiễu" trong việc ổn định kết quả assets.

## 8. Minh Chứng Kết Quả

-   **Source Code:** Toàn bộ logic được triển khai trong tệp python.
-   **History Log:** Các tệp log được tạo ra bởi TensorBoard và các thông báo in ra trong quá trình huấn luyện cung cấp chi tiết về các chỉ số học tập (loss, ep_rew_mean, explained_variance, v.v.).
-   **Biểu đồ huấn luyện:** Hình ảnh trực quan về đường cong vốn chủ sở hữu trong quá trình huấn luyện là minh chứng rõ ràng nhất cho hành vi "bùng nổ và sụp đổ" của agent, cho thấy sự cần thiết phải cải tiến hơn nữa về mặt quản lý rủi ro.
