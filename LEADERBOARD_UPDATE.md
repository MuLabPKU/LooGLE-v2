# Leaderboard 数据更新指南

## 如何更新 Leaderboard 数据

### 方法 1: 直接编辑 HTML 文件

打开 `index.html`，找到 JavaScript 部分的 `leaderboardData` 对象（大约在第 400 行），更新数据：

```javascript
const leaderboardData = {
    overall: [
        { rank: 1, model: "模型名称", overall: 87.5, finance: 89.2, law: 86.8, code: 88.5, game: 85.5 },
        // 添加更多模型...
    ],
    finance: [...],
    law: [...],
    code: [...],
    game: [...]
};
```

### 方法 2: 从 JSON 文件加载（推荐）

1. 创建一个 `leaderboard.json` 文件，格式如下：

```json
{
    "overall": [
        {
            "rank": 1,
            "model": "GPT-4o",
            "overall": 87.5,
            "finance": 89.2,
            "law": 86.8,
            "code": 88.5,
            "game": 85.5
        }
    ],
    "finance": [...],
    "law": [...],
    "code": [...],
    "game": [...]
}
```

2. 在 `index.html` 的 JavaScript 部分，取消注释以下代码：

```javascript
// fetch('leaderboard.json')
//     .then(response => response.json())
//     .then(data => {
//         leaderboardData = data;
//         renderLeaderboard('overall');
//     });
```

### 方法 3: 从评估结果自动生成

你可以修改 `evaluate.py` 脚本，让它输出符合 leaderboard 格式的 JSON 文件：

```python
# 在 evaluate.py 中添加
def generate_leaderboard_json(results_dir, output_file='leaderboard.json'):
    # 读取所有评估结果文件
    # 汇总数据并生成 leaderboard.json
    pass
```

## 数据格式说明

- `rank`: 排名（整数）
- `model`: 模型名称（字符串）
- `overall`: 总体准确率（浮点数，0-100）
- `finance`: Finance 领域准确率（浮点数，0-100）
- `law`: Law 领域准确率（浮点数，0-100）
- `code`: Code 领域准确率（浮点数，0-100）
- `game`: Game 领域准确率（浮点数，0-100）

## 部署

直接将 `index.html` 放在项目根目录，然后：

1. **GitHub Pages**: 推送到 GitHub，在仓库设置中启用 GitHub Pages
2. **本地服务器**: 使用 `python -m http.server` 或 `npx serve`
3. **其他静态托管**: 上传到任何静态网站托管服务

