"""贪吃蛇 - 使用 tkinter 实现，无需第三方依赖。

操作:
    方向键 / WASD  控制方向
    空格 / P       暂停 / 继续
    M              切换穿墙模式
    R              重新开始
    Esc            退出

最高分会自动保存到 snake_highscore.json。
"""

import json
import os
import random
import tkinter as tk

# 网格与画布参数
COLS, ROWS = 30, 20
CELL = 25
WIDTH, HEIGHT = COLS * CELL, ROWS * CELL
INIT_SPEED_MS = 150          # 初始移动间隔(毫秒)
MIN_SPEED_MS = 70            # 最快移动间隔
SPEEDUP_STEP = 4             # 每吃多少个食物加速一次

COLORS = {
    "bg": "#1e1e2e",
    "snake": "#a6e3a1",
    "head": "#89dceb",
    "food": "#f38ba8",
    "text": "#cdd6f4",
    "over": "#f38ba8",
}

# 最高分存档
HIGHSCORE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "snake_highscore.json")


def load_high_score():
    """读取历史最高分，失败时返回 0。"""
    try:
        with open(HIGHSCORE_FILE, "r", encoding="utf-8") as f:
            return int(json.load(f).get("high_score", 0))
    except (OSError, ValueError, json.JSONDecodeError, AttributeError):
        return 0


def save_high_score(score):
    """保存最高分，忽略写入错误。"""
    try:
        with open(HIGHSCORE_FILE, "w", encoding="utf-8") as f:
            json.dump({"high_score": score}, f)
    except OSError:
        pass

# 方向向量
DIRS = {"Up": (0, -1), "Down": (0, 1), "Left": (-1, 0), "Right": (1, 0)}
# 按键 -> 方向
KEY_MAP = {
    "Up": "Up", "w": "Up", "W": "Up",
    "Down": "Down", "s": "Down", "S": "Down",
    "Left": "Left", "a": "Left", "A": "Left",
    "Right": "Right", "d": "Right", "D": "Right",
}


class SnakeLogic:
    """与界面无关的游戏逻辑，便于测试。"""

    def __init__(self):
        self.wall_mode = False          # 穿墙模式（默认关闭）
        self.high_score = load_high_score()
        self.reset()

    def reset(self):
        """初始化游戏状态。"""
        self.snake = [(COLS // 2, ROWS // 2), (COLS // 2 - 1, ROWS // 2),
                      (COLS // 2 - 2, ROWS // 2)]
        self.direction = "Right"
        self.pending_dir = "Right"
        self.score = 0
        self.food_eaten = 0
        self.speed = INIT_SPEED_MS
        self.game_over = False
        self.paused = False
        self.new_record = False
        self.food = None
        self.spawn_food()

    def spawn_food(self):
        """在空白格子上随机生成食物。"""
        free = [(x, y) for x in range(COLS) for y in range(ROWS)
                if (x, y) not in self.snake]
        self.food = random.choice(free) if free else None

    def handle_key(self, keysym):
        """处理方向键；返回 True 表示方向被更新。"""
        new_dir = KEY_MAP.get(keysym)
        if not new_dir or self.game_over:
            return False
        # 不允许直接掉头
        opposite = {"Up": "Down", "Down": "Up", "Left": "Right", "Right": "Left"}
        if new_dir == opposite[self.direction]:
            return False
        self.pending_dir = new_dir
        return True

    def toggle_pause(self):
        """暂停 / 继续，游戏结束后无效。"""
        if not self.game_over:
            self.paused = not self.paused
        return self.paused

    def toggle_wall(self):
        """切换穿墙模式。"""
        self.wall_mode = not self.wall_mode
        return self.wall_mode

    def _end_game(self):
        """结束游戏并更新最高分。"""
        self.game_over = True
        if self.score > self.high_score:
            self.high_score = self.score
            self.new_record = True
            save_high_score(self.high_score)

    def step(self):
        """推进一帧。"""
        if self.game_over or self.paused:
            return

        self.direction = self.pending_dir
        dx, dy = DIRS[self.direction]
        head = self.snake[0]
        new_head = (head[0] + dx, head[1] + dy)

        # 穿墙模式下从另一侧出现；否则撞墙结束
        if self.wall_mode:
            new_head = (new_head[0] % COLS, new_head[1] % ROWS)
            hit_wall = False
        else:
            hit_wall = not (0 <= new_head[0] < COLS and 0 <= new_head[1] < ROWS)
        # 撞到自己（尾巴即将移开的位置不算）-> 游戏结束
        hit_self = new_head in self.snake[:-1]
        if hit_wall or hit_self:
            self._end_game()
            return

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 10
            self.food_eaten += 1
            if self.food_eaten % SPEEDUP_STEP == 0:
                self.speed = max(MIN_SPEED_MS, self.speed - 10)
            self.spawn_food()
            if self.food is None:  # 蛇占满全屏，胜利
                self._end_game()
        else:
            self.snake.pop()


class SnakeGame(SnakeLogic):
    """tkinter 界面包装。"""

    def __init__(self, root):
        super().__init__()
        self.root = root
        root.title("贪吃蛇")
        root.resizable(False, False)

        self.canvas = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg=COLORS["bg"],
                                highlightthickness=0)
        self.canvas.pack()
        self.score_label = tk.Label(root, text="得分: 0", font=("Consolas", 14),
                                    fg=COLORS["text"], bg=COLORS["bg"])
        self.score_label.pack(fill="x")

        root.bind("<KeyPress>", self.on_key)
        root.focus_set()

        self.draw()

    def on_key(self, event):
        if event.keysym in ("r", "R"):
            self.reset()
            self.score_label.config(text=f"得分: {self.score}  最高: {self.high_score}")
            self.draw()
            return
        if event.keysym == "Escape":
            self.root.destroy()
            return
        if event.keysym in ("space", "p", "P"):
            self.toggle_pause()
            self.draw()
            return
        if event.keysym in ("m", "M"):
            self.toggle_wall()
            self.draw()
            return
        self.handle_key(event.keysym)

    def step(self):
        super().step()
        self.draw()
        if self.game_over:
            tip = "新纪录! " if self.new_record else ""
            self.score_label.config(
                text=f"{tip}游戏结束! 得分: {self.score}  最高: {self.high_score}  (按 R 重新开始)")
        else:
            mode = "穿墙" if self.wall_mode else "普通"
            self.score_label.config(
                text=f"得分: {self.score}  最高: {self.high_score}  模式: {mode}")

    def draw(self):
        """重绘画面。"""
        self.canvas.delete("all")
        # 网格线
        for i in range(1, COLS):
            self.canvas.create_line(i * CELL, 0, i * CELL, HEIGHT, fill="#313244")
        for i in range(1, ROWS):
            self.canvas.create_line(0, i * CELL, WIDTH, i * CELL, fill="#313244")

        # 食物
        if self.food is not None:
            fx, fy = self.food
            self.canvas.create_oval(fx * CELL + 3, fy * CELL + 3,
                                    (fx + 1) * CELL - 3, (fy + 1) * CELL - 3,
                                    fill=COLORS["food"], outline=COLORS["food"])

        # 蛇
        for i, (x, y) in enumerate(self.snake):
            color = COLORS["head"] if i == 0 else COLORS["snake"]
            pad = 1 if i == 0 else 2
            self.canvas.create_rectangle(x * CELL + pad, y * CELL + pad,
                                         (x + 1) * CELL - pad, (y + 1) * CELL - pad,
                                         fill=color, outline=color)

        # 穿墙模式提示
        if self.wall_mode and not self.game_over:
            self.canvas.create_text(8, 8, anchor="nw", text="穿墙模式",
                                    font=("Consolas", 12), fill=COLORS["food"])

        # 暂停提示
        if self.paused and not self.game_over:
            self.canvas.create_text(WIDTH // 2, HEIGHT // 2,
                                    text="PAUSED", font=("Consolas", 36, "bold"),
                                    fill=COLORS["text"])
            self.canvas.create_text(WIDTH // 2, HEIGHT // 2 + 40,
                                    text="按空格继续", font=("Consolas", 16),
                                    fill=COLORS["text"])

        # 结束提示
        if self.game_over:
            title = "新纪录!" if self.new_record else "GAME OVER"
            self.canvas.create_text(WIDTH // 2, HEIGHT // 2,
                                    text=title, font=("Consolas", 36, "bold"),
                                    fill=COLORS["over"])
            self.canvas.create_text(WIDTH // 2, HEIGHT // 2 + 40,
                                    text="按 R 重新开始", font=("Consolas", 16),
                                    fill=COLORS["text"])

    def run(self):
        self._tick()

    def _tick(self):
        self.step()
        self.root.after(self.speed if not self.game_over else 200, self._tick)


if __name__ == "__main__":
    root = tk.Tk()
    SnakeGame(root).run()
    root.mainloop()
