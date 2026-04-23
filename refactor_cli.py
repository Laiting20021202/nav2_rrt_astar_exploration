import os
from google import genai
from google.genai import types

# 1. 初始化
api_key = os.environ.get("GOOGLE_API_KEY")
if not api_key:
    print("錯誤：請先執行 export GOOGLE_API_KEY='你的金鑰'")
    exit()

client = genai.Client(api_key=api_key)

# 2. 自動搜尋檔案路徑 (不分大小寫並顯示搜尋過程)
def find_file(name):
    print(f"正在搜尋 {name} ...")
    for root, dirs, files in os.walk("."):
        if name in files:
            p = os.path.join(root, name)
            print(f"  -> 找到目標路徑: {p}")
            return p
    return None

target_names = ["exploration_coordinator.py", "frontier_scoring.py", "exploration_manager.yaml"]
files_to_fix = [find_file(f) for f in target_names if find_file(f) is not None]

# 移除重複項 (避免 find_file 執行兩次)
files_to_fix = list(set([f for f in files_to_fix if f is not None]))

def refactor():
    if not files_to_fix:
        print("\n[錯誤]：完全找不到任何目標檔案！")
        print("請確認你是在專案根目錄下，且執行過 ls -R 能看到 src 資料夾。")
        return

    for file_path in files_to_fix:
        print(f"\n🚀 正在重構: {file_path} ...")
        with open(file_path, "r") as f:
            original_code = f.read()

        prompt = f"""
        你現在是一位 ROS 2 機器人導航專家。請修改這份代碼。
        目標：將專案從『邊境探索』完全轉向為『精確前往 Goal Pose』。
        
        修改重點要求：
        1. 提高 w_goal 權重（設為 2.5 以上）。
        2. 強化 w_visit (memory penalty)，確保機器人絕對不走回頭路。
        3. 修改 plan_on_known_free 邏輯：若路徑規劃失敗，立即嘗試『樂觀路徑規劃』（allow_unknown=True）。
        4. 降低 w_info（資訊增益）權重。
        
        檔案路徑：{file_path}
        內容：
        {original_code}
        
        請直接回傳修改後的完整內容，不要包含 Markdown 標記 (如 ```python) 或解釋。
        """

        try:
            response = client.models.generate_content(
                model="gemini-1.5-pro",
                config=types.GenerateContentConfig(temperature=0.0),
                contents=prompt
            )
            
            new_code = response.text.strip()
            # 徹底移除可能的 Markdown 標籤
            for tag in ["```python", "```yaml", "```"]:
                new_code = new_code.replace(tag, "")
            
            with open(file_path, "w") as f:
                f.write(new_code.strip())
            print(f"✅ 成功！{file_path} 已更新。")
            
        except Exception as e:
            print(f"❌ 處理 {file_path} 時出錯: {e}")

if __name__ == "__main__":
    refactor()
