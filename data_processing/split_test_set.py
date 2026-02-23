import torch
import pickle
import os
import shutil

# --- 配置路径 ---
train_path = "/home/lizijia/Gen-JEPA/vae/data/pusht_noise/train"
test_path = "/home/lizijia/Gen-JEPA/vae/data/pusht_noise/test"

# 创建测试集目录结构
os.makedirs(os.path.join(test_path, "obses"), exist_ok=True)

# --- 1. 读取长度表并确定切分点 ---
with open(os.path.join(train_path, "seq_lengths.pkl"), "rb") as f:
    lengths = pickle.load(f)

num_test_videos = 1000  # 想要分出的视频数量
split_idx_in_vids = len(lengths) - num_test_videos

# 确定 Tensor 的切分物理索引
total_frames = sum(lengths)
test_frames_count = sum(lengths[split_idx_in_vids:])
split_idx_in_frames = total_frames - test_frames_count

# --- 2. 移动并重命名视频文件 ---
print(f"开始移动视频并重新编号 (格式: episode_XXX.mp4)...")
for i in range(num_test_videos):
    old_idx = split_idx_in_vids + i
    new_idx = i

    # 根据你的描述 episode_007.mp4 似乎是3位索引。
    # 如果实际是更多位（如7位），请将 :03d 改为 :07d
    src_filename = f"episode_{old_idx:03d}.mp4" 
    dst_filename = f"episode_{new_idx:03d}.mp4"

    src_vid = os.path.join(train_path, "obses", src_filename)
    dst_vid = os.path.join(test_path, "obses", dst_filename)
    
    if os.path.exists(src_vid):
        shutil.move(src_vid, dst_vid)
    else:
        # 如果找不到，尝试匹配不带补零的格式，增加鲁棒性
        print(f"警告: 未找到 {src_vid}，请检查位数是否为3位 (03d)")

# --- 3. 截取并保存所有 .pth 文件 ---
files_to_process = ["abs_actions.pth", "rel_actions.pth", "states.pth", "tokens.pth", "velocities.pth"]
for file_name in files_to_process:
    full_path = os.path.join(train_path, file_name)
    if not os.path.exists(full_path):
        continue
    
    print(f"正在加载并切分 {file_name}...")
    data = torch.load(full_path)
    
    # 统一切分逻辑：无论是 Tensor 还是 List，切片语法 [:] 都是通用的
    train_data = data[:split_idx_in_vids]
    test_data = data[split_idx_in_vids:]
    
    # 获取长度用于打印 (兼容 Tensor 的 .shape 和 List 的 len)
    t_len = train_data.shape[0] if torch.is_tensor(train_data) else len(train_data)
    v_len = test_data.shape[0] if torch.is_tensor(test_data) else len(test_data)
    
    # 保存
    torch.save(train_data, os.path.join(train_path, file_name))
    torch.save(test_data, os.path.join(test_path, file_name))
    print(f"✅ 已完成 {file_name}: Train 数量={t_len}, Test 数量={v_len}")
    
# --- 4. 更新并保存 seq_lengths.pkl ---
train_lengths = lengths[:split_idx_in_vids]
test_lengths = lengths[split_idx_in_vids:]

with open(os.path.join(train_path, "seq_lengths.pkl"), "wb") as f:
    pickle.dump(train_lengths, f)
with open(os.path.join(test_path, "seq_lengths.pkl"), "wb") as f:
    pickle.dump(test_lengths, f)

print("🎉 数据集物理切分成功！")
print(f"Train 剩余视频: {len(train_lengths)}, Test 新增视频: {len(test_lengths)}")