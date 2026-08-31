"""snake.py 游戏逻辑的无界面测试。"""
import os
import unittest

import snake


class SnakeLogicTest(unittest.TestCase):
    def test_initial_state(self):
        g = snake.SnakeLogic()
        self.assertEqual(len(g.snake), 3)
        self.assertEqual(g.direction, "Right")
        self.assertEqual(g.score, 0)
        self.assertFalse(g.game_over)
        self.assertNotIn(g.food, g.snake)

    def test_move_forward(self):
        g = snake.SnakeLogic()
        head = g.snake[0]
        g.step()
        self.assertEqual(g.snake[0], (head[0] + 1, head[1]))

    def test_eat_food(self):
        g = snake.SnakeLogic()
        g.food = (g.snake[0][0] + 1, g.snake[0][1])
        before = len(g.snake)
        g.step()
        self.assertEqual(g.score, 10)
        self.assertEqual(len(g.snake), before + 1)  # 吃到了，不弹尾巴

    def test_hit_wall(self):
        g = snake.SnakeLogic()
        g.snake = [(0, 0), (1, 0), (2, 0)]
        g.direction = "Left"
        g.pending_dir = "Left"
        g.food = (10, 10)
        g.step()
        self.assertTrue(g.game_over)

    def test_hit_self(self):
        g = snake.SnakeLogic()
        g.snake = [(5, 5), (4, 5), (4, 6), (5, 6), (6, 6), (6, 5)]
        g.direction = "Left"
        g.pending_dir = "Left"
        g.food = (10, 10)
        g.step()
        self.assertTrue(g.game_over)

    def test_no_reverse(self):
        g = snake.SnakeLogic()
        self.assertFalse(g.handle_key("Left"))  # 与 Right 相反，应被忽略
        self.assertEqual(g.pending_dir, "Right")
        self.assertTrue(g.handle_key("Up"))
        self.assertEqual(g.pending_dir, "Up")

    def test_handle_key_wasd(self):
        g = snake.SnakeLogic()
        g.handle_key("d")
        self.assertEqual(g.pending_dir, "Right")
        g.handle_key("s")
        self.assertEqual(g.pending_dir, "Down")

    def test_food_not_on_snake(self):
        g = snake.SnakeLogic()
        g.spawn_food()
        self.assertNotIn(g.food, g.snake)

    def test_speedup(self):
        g = snake.SnakeLogic()
        g.food_eaten = snake.SPEEDUP_STEP - 1
        before = g.speed
        g.food = (g.snake[0][0] + 1, g.snake[0][1])
        g.step()
        self.assertEqual(g.food_eaten, snake.SPEEDUP_STEP)
        self.assertLess(g.speed, before)

    def test_pause(self):
        g = snake.SnakeLogic()
        head = g.snake[0]
        g.paused = True
        g.step()
        self.assertEqual(g.snake[0], head)  # 暂停时蛇不动
        g.paused = False
        g.step()
        self.assertNotEqual(g.snake[0], head)  # 恢复后前进

    def test_toggle_pause(self):
        g = snake.SnakeLogic()
        self.assertTrue(g.toggle_pause())
        self.assertTrue(g.paused)
        self.assertFalse(g.toggle_pause())
        self.assertFalse(g.paused)

    def test_toggle_wall(self):
        g = snake.SnakeLogic()
        self.assertFalse(g.wall_mode)
        self.assertTrue(g.toggle_wall())
        self.assertTrue(g.wall_mode)

    def test_wall_mode_wrap(self):
        g = snake.SnakeLogic()
        g.snake = [(0, 0), (1, 0), (2, 0)]
        g.direction = "Left"
        g.pending_dir = "Left"
        g.food = (10, 10)
        g.wall_mode = True
        g.step()
        self.assertFalse(g.game_over)
        self.assertEqual(g.snake[0], (snake.COLS - 1, 0))  # 从右侧出现

    def test_high_score_saved(self):
        import tempfile
        tmp = tempfile.TemporaryDirectory()
        old_path = snake.HIGHSCORE_FILE
        snake.HIGHSCORE_FILE = os.path.join(tmp.name, "hs.json")
        try:
            g = snake.SnakeLogic()
            self.assertEqual(g.high_score, 0)
            g.snake = [(0, 0), (1, 0), (2, 0)]
            g.direction = "Left"
            g.pending_dir = "Left"
            g.score = 50
            g.food = (10, 10)
            g.step()  # 撞墙结束
            self.assertTrue(g.game_over)
            self.assertEqual(g.high_score, 50)
            self.assertTrue(g.new_record)
            g2 = snake.SnakeLogic()
            self.assertEqual(g2.high_score, 50)  # 重新加载仍保留
        finally:
            snake.HIGHSCORE_FILE = old_path
            tmp.cleanup()


if __name__ == "__main__":
    unittest.main()
