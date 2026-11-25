# pairwise-document-shadow-flatten

## 领域

游戏、图形与多媒体

## 主题

文档阴影压平

## 说明

低通分离阴影、文字保护与分区增益。

初始状态下 `imaging/document_shadow_flatten.py` 中所有函数抛出 `NotImplementedError`，测试全部失败。
修复实现后运行：

```bash
python -m pytest -q
```

期望全部通过。
