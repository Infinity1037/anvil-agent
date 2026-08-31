"""todo.py 的单元测试。"""
import os
import tempfile
import unittest

import todo


class TodoListTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = os.path.join(self.tmp.name, "test.json")
        self.todos = todo.TodoList(self.path)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add(self):
        item = self.todos.add("买牛奶")
        self.assertEqual(item["text"], "买牛奶")
        self.assertFalse(item["done"])
        self.assertEqual(self.todos.pending(), [item])

    def test_id_increment(self):
        a = self.todos.add("任务 A")
        b = self.todos.add("任务 B")
        self.assertEqual(b["id"], a["id"] + 1)

    def test_persistence(self):
        self.todos.add("写周报", due="2025-06-01")
        self.todos.add("买菜")
        self.todos.set_done(1)
        again = todo.TodoList(self.path)
        self.assertEqual(len(again.items), 2)
        self.assertTrue(again.find(1)["done"])
        self.assertEqual(again.find(1)["due"], "2025-06-01")

    def test_done_undo(self):
        self.todos.add("任务")
        self.todos.set_done(1)
        self.assertEqual(len(self.todos.pending()), 0)
        self.assertEqual(len(self.todos.done_items()), 1)
        self.todos.set_done(1, done=False)
        self.assertEqual(len(self.todos.pending()), 1)
        self.assertIsNone(self.todos.find(1)["finished"])

    def test_remove(self):
        self.todos.add("任务")
        removed = self.todos.remove(1)
        self.assertIsNotNone(removed)
        self.assertIsNone(self.todos.find(1))
        self.assertIsNone(self.todos.remove(999))  # 不存在的编号

    def test_clear_done(self):
        self.todos.add("任务 A")
        self.todos.add("任务 B")
        self.todos.set_done(1)
        n = self.todos.clear_done()
        self.assertEqual(n, 1)
        self.assertEqual(len(self.todos.items), 1)

    def test_clear_all(self):
        self.todos.add("任务 A")
        self.todos.add("任务 B")
        self.assertEqual(self.todos.clear_all(), 2)
        self.assertEqual(self.todos.items, [])

    def test_stats(self):
        self.todos.add("任务 A")
        self.todos.add("任务 B")
        self.todos.set_done(1)
        self.assertEqual(self.todos.stats(), (2, 1, 1))

    def test_bad_file_recovers(self):
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("not json{{{")
        t = todo.TodoList(self.path)
        self.assertEqual(t.items, [])
        self.assertEqual(t._next_id, 1)


if __name__ == "__main__":
    unittest.main()
