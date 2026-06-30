# 医疗智能问答助手前端

React + Vite + TypeScript + Ant Design。

## 启动

先启动 FastAPI 后端：

```bash
uv run uvicorn app.main:app --reload --port 8000
```

再启动前端：

```bash
npm.cmd run dev -- --host 127.0.0.1 --port 5173
```

访问：

```text
http://127.0.0.1:5173/
```

## 接口地址

默认请求：

```text
http://localhost:8000/api/v1
```

如需覆盖，创建 `frontend/.env.local`：

```env
VITE_API_BASE_URL=http://localhost:8000/api/v1
```

## 当前功能

- 登录 / 注册
- 会话列表 / 会话详情
- SSE 流式聊天
- RAG 引用来源展示
- 管理员文档上传入库
- 管理员索引内置 `medical_new_2.json`
