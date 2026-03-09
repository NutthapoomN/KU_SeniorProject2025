df = readtable("D:\Senior Project\Data\22-12-2025.xlsx");
Time = df{:,1};
Hour = df{:,2};
Entry = df{:,5};
Exit = df{:,6};
Exit = abs(Exit);
People = df{:,7};
Light = df{:,8};
Door = df{:,9};

hourly_matrix = reshape(People, 3600, 24);
hourly_avg = mean(hourly_matrix, 1);

minute_matrix = reshape(People, 60, 1440);
minute_avg = mean(minute_matrix, 1);

%% Plot 
figure(1);
bar(0:23, hourly_avg, 'FaceColor', [0 0.45 0.74]);
grid on;
xlabel('Time [Hour-by-Hour]');
ylabel('Quantity');
title('Average Occupancy per Hour');
xticks(0:23); % แสดงเลขชั่วโมงครบทุกแท่ง

figure(2);
bar(1:1440, minute_avg, 'EdgeColor', 'none', 'FaceColor', [0.85 0.33 0.1]);
grid on;
xlabel('Time [Minute-by-Minute]');
ylabel('Quantity');
title('Average Occupancy per Minute');

% 1. เตรียมข้อมูล
Light = reshape(Light, 1, []); 
Time_Hours = Time / 3600; % แปลงวินาทีเป็นชั่วโมง

figure('Color', [1 1 1]); 
hold on;

% 2. วาดกราฟโดยใช้แกน X เป็นชั่วโมง
imagesc(Time_Hours, [0 1], Light); 
axis xy; 

% 3. ตั้งค่าสี: [OFF (Gray), ON (#129EA6)]
colormap([0.2 0.2 0.2; 18/255, 158/255, 166/255]); 

% 4. ปรับแต่งแกน X ให้โชว์ทุกๆ 1 หรือ 2 ชั่วโมง (ตามความเหมาะสม)
max_hour = max(Time_Hours);
xticks(0:2:max_hour); % แสดงตัวเลขทุกๆ 2 ชั่วโมง (ปรับเลข 2 เป็น 1 ได้ถ้าต้องการละเอียดขึ้น)

% 5. ตกแต่งความสวยงาม (ปรับ XColor/YColor เป็นสีดำ 'k' เพื่อให้ตัดกับพื้นหลังขาว)
set(gca, 'Color', [1 1 1], ... 
         'XColor', 'k', 'YColor', 'k', ...
         'YTick', [], 'Box', 'off', 'FontSize', 12);

xlabel('Time (Hours)', 'Color', 'k');
title('System Operating Status (On/Off)', 'Color', 'k');
axis tight;
