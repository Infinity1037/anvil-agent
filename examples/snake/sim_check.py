"""临时脚本：模拟贪吃蛇，检查不变量与边界情况。"""
import random
import snake

random.seed(42)

# 1) 随机长时间游玩，检查不变量
g = snake.SnakeLogic()
invariant_errors = []
for i in range(50000):
    # 随机按键
    if random.random() < 0.3:
        g.handle_key(random.choice(list(snake.KEY_MAP)))
    g.step()
    if g.game_over:
        # 检查结束时状态一致
        break
    # 不变量
    if len(set(g.snake)) != len(g.snake):
        invariant_errors.append(("overlap", i, g.snake))
        break
    if g.food is not None and g.food in g.snake:
        invariant_errors.append(("food_on_snake", i, g.food, g.snake))
        break
    if g.food is not None and not (0 <= g.food[0] < snake.COLS and 0 <= g.food[1] < snake.ROWS):
        invariant_errors.append(("food_outside", i, g.food))
        break
print("随机游玩模拟: 结束于第", i, "步, game_over =", g.game_over,
      "score =", g.score, "蛇长 =", len(g.snake))
print("不变量错误:", invariant_errors if invariant_errors else "无")

# 2) 吃食物后长度与分数
g2 = snake.SnakeLogic()
g2.food = (g2.snake[0][0] + 1, g2.snake[0][1])
g2.step()
print("吃食物后: score =", g2.score, "长度 =", len(g2.snake), "新食物 =", g2.food)

# 3) 蛇追尾(尾巴即将离开)不死亡
g3 = snake.SnakeLogic()
g3.snake = [(3, 3), (2, 3), (2, 4), (3, 4)]
g3.direction = "Up"
g3.pending_dir = "Up"
g3.food = (10, 10)
g3.step()
print("追尾(尾巴移开): game_over =", g3.game_over, "头部 =", g3.snake[0])

# 4) 一帧内连续按两个键(Up 然后 Down)，检查是否出现非法转向
g4 = snake.SnakeLogic()
g4.handle_key("Up")
g4.handle_key("Down")
g4.step()
print("连按 Up+Down 后方向 =", g4.direction)

# 5) 一帧内 Up 然后 Left（Left 是 Right 的反向，应被拒绝）
g5 = snake.SnakeLogic()
g5.handle_key("Up")
accepted = g5.handle_key("Left")
g5.step()
print("连按 Up+Left: Left 被接受 =", accepted, "最终方向 =", g5.direction)

# 6) 蛇占满全屏时 spawn_food
g6 = snake.SnakeLogic()
g6.snake = [(x, y) for x in range(snake.COLS) for y in range(snake.ROWS)]
g6.food = None
g6.spawn_food()
print("全屏时食物 =", g6.food)

# 7) 暂停时按键不应移动
g7 = snake.SnakeLogic()
g7.paused = True
head = g7.snake[0]
g7.step()
print("暂停移动: 头不变 =", g7.snake[0] == head)
