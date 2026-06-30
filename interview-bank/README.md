# 项目面试题库独立版

这个目录是完全静态页面，不依赖后端、登录、React、Vite 或 Docker。

## 打开方式

直接用浏览器打开：

```text
interview-bank/index.html
```

也可以在文件管理器中双击 `index.html`。

## 文件说明

- `index.html`：页面结构
- `styles.css`：页面样式
- `app.js`：搜索、筛选、随机抽题、折叠答案等交互
- `questions.js`：面试题数据

## 维护题库

后续新增或修改题目，直接编辑 `questions.js` 中的 `window.interviewQuestions` 数组即可。
